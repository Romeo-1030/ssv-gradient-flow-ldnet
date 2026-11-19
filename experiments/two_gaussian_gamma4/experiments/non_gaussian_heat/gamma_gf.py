"""Two CUDA treatments: fixed global clock + GF versus two learned clocks + GF.

Both fix alpha=beta=1/2. The two-chart model additionally has early-frame
overlap coefficient one. An independent IC-to-gamma predictor is pretrained
and frozen before latent/GF training; no per-test-IC optimization, future
field labels, or PDE right-hand side is used.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import platform
import time

import numpy as np
import torch
from torch import nn

from experiment import ROOT, loss, metrics
from two_gaussian_data import make_data, fingerprint, boundary_report, PROTOCOL
from moving_charts import overlap_loss, state_hash
from original_scale_model import EnergyDensity, ORIGINAL_SOURCE
from original_scale import (OBS, make_gf_indices, validation_loss, save_json,
                            cpu_state, sync, diagnostics)
from gamma_gf_model import METHODS, GammaGFModel, raw_gf_terms


def make_energy(seed):
    # Match the original global-model RNG consumption, and give BOTH treatments
    # identical initial energy parameters regardless of their expert count.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = GammaGFModel('global_fixed_gf')
        energy = EnergyDensity()
    return energy


def gamma_report(model, y, ix):
    report = {}
    with torch.no_grad():
        for split, ids in ix.items():
            g = model.gamma_values(y[ids, 0]).cpu().numpy()
            if not np.isfinite(g).all() or not (g > 0).all():
                raise RuntimeError(f'Invalid positive clock in {split}')
            report[split] = {'ids': ids.tolist(), 'values': g.tolist(),
                             'mean': g.mean(0).tolist(), 'std': g.std(0).tolist(),
                             'min': g.min(0).tolist(), 'max': g.max(0).tolist()}
    return report


def evaluate_test(model,y,ix,x,t,out,method,seed,device):
    """Held-out ICs; future labels enter only here, after checkpoint selection."""
    ids=ix['test']
    model.eval()
    with torch.no_grad():
        sync(device)
        start=time.perf_counter()
        pred,parts=model(y[ids,0],t,x,components=True)
        sync(device)
        seconds=time.perf_counter()-start
        if not torch.isfinite(pred).all():
            raise RuntimeError(f'Nonfinite prediction: {method}, seed{seed}')
        pc,yc=pred.cpu(),y[ids].cpu()
        record=metrics(pc,yc,x.cpu(),t.cpu())
        record['initial']=record['curve'][0]
        record['prediction_seconds']=seconds
        record['diagnostics']=diagnostics(model,pred,parts,y[ids],x,t)
        squared=(pc.double()-yc.double()).square().sum(-1)
        per_ic=(squared/yc.double().square().sum(-1).clamp_min(1e-12)).sqrt()
        record['per_ic_long']=per_ic[:,t.cpu()>2].mean(-1).tolist()
        record['per_ic_final']=per_ic[:,-1].tolist()
        # The inherited 'observed' metric also samples t=.125,.375. Name the
        # three actually labelled instants explicitly to avoid ambiguity.
        record['label_times_error']=float(np.asarray(record['curve'])[list(OBS)].mean())
        np.savez_compressed(out/f'{method}_{seed}_test.npz',prediction=pc.numpy(),
                            components=parts.cpu().numpy(),ids=ids)
    return record


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use .venv-cuda')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists() and not args.resume:
        raise FileExistsError('Existing run is preserved: choose a new --out or --resume')
    data=make_data()
    xx,tt,yy,splits=(data[k] for k in ('x','t','u','split'))
    if tt[0] != 0 or not np.all(np.diff(tt) > 0):
        raise ValueError('Increasing rollout grid beginning at zero required')
    x, t, y = [torch.tensor(a, dtype=torch.float32, device=device) for a in (xx,tt,yy)]
    ix = {k: np.flatnonzero(splits == k) for k in ('train','valid','test')}
    gf_indices = make_gf_indices(t)
    gamma_path=ROOT/args.gamma_source
    gamma_checkpoint=torch.load(gamma_path,map_location='cpu',weights_only=True)
    if gamma_checkpoint['config']['data_fingerprint']!=fingerprint(data):
        raise ValueError('Gamma calibration and latent training must use identical data')
    gamma_state=gamma_checkpoint['state']
    gamma_source_hash=hashlib.sha256(gamma_path.read_bytes()).hexdigest()
    source_names = ('gamma_gf.py','gamma_gf_model.py','original_scale.py',
                    'original_scale_model.py','experiment.py','moving_charts.py','two_stage.py',
                    'pretrain_gamma.py','two_gaussian_data.py')
    config = vars(args) | {
        'model_description': 'Original-size backbone; fixed heat exponents; constant global or two IC-conditioned clocks',
        'latent': 16, 'width': 128, 'depth': 3, 'condition_dim': 16, 'ic_samples': 192,
        'fourier_frequencies': 16, 'fourier_coefficients': 33,
        'alpha': .5, 'beta': .5, 'gamma_pretraining_initial': 1.,
        'gamma_learning': 'Independent IC192 -> Linear16 -> Tanh -> Linear2 -> softplus; pretrained then fully frozen',
        'gamma_objective': 'Three-frame chart-weighted initial-template alignment; approximate coordinate fitting',
        'gamma_constraints': 'positive softplus; predictor independent of trainable latent encoder',
        'gamma_source_sha256': gamma_source_hash,
        'gamma_calibration_selection': gamma_checkpoint['selection'],
        'gamma_calibration_shared_across_seeds': True,
        'centers': {'global_fixed_gf': [0.], 'charts2_gamma_gf': [-4.,4.]},
        'gate_width': 4., 'gates': 'fixed physical Gaussian POU',
        'supervised_times': [0.,.25,.5], 'overlap_times': [0.,.25,.5],
        'gf_times': t[gf_indices].cpu().tolist(), 'gf_time_points': len(gf_indices),
        'gf_collocation_per_chart': 24, 'gf_sampling_coordinates': 'each transformed xi in [-24,24]',
        'gf_aggregation': 'mean over chart residuals; one shared energy density; not GF on blended output',
        'gf_schedule': 'weight zero through100; detached mean raw normalization at101; ramp to lambda_gf at200',
        'energy_initialization': 'same seed, original global-model RNG consumption, matched across methods',
        'checkpoint_selection': 'minimum per-IC initial-L2-normalized early validation error; every10 steps',
        'optimizer': {'name':'AdamW','lr':.001,'weight_decay':1e-6,'clip_norm':5,
                      'clip_scope':'trainable backbone only; frozen SSV and energy excluded',
                      'scheduler':'CosineAnnealingLR','eta_min':1e-5},
        'split_counts': {k:len(v) for k,v in ix.items()}, 'domain':[-24,24],
        'boundary':PROTOCOL['reference'], 'data_protocol':PROTOCOL,
        'data_fingerprint':fingerprint(data),'boundary_check':boundary_report(data),
        'parameter_matching':False, 'gf_compute_matching':False,
        'torch':str(torch.__version__), 'python':platform.python_version(),
        'cuda_runtime':torch.version.cuda, 'threads':1, 'tf32':False,
        'deterministic_algorithms':True, 'synchronized_gpu_timing':True,
        'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        'device_memory_bytes':torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},
        'original_train_sha256':hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest(),
    }
    if (out/'config.json').exists():
        old = json.loads((out/'config.json').read_text())
        for key in ('steps','seeds','methods','overlap_lambda','lambda_gf','source_sha256',
                    'original_train_sha256','gamma_source_sha256','data_fingerprint','torch','device_name'):
            if old[key] != config[key]:
                raise ValueError(f'Resume mismatch: {key}')
    else:
        save_json(out/'config.json', config)
        np.savez_compressed(out/'data.npz', **data)
        snapshot=out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in source_names:
            (snapshot/name).write_bytes((ROOT/name).read_bytes())
        (snapshot/'upstream_train.py').write_bytes(ORIGINAL_SOURCE.read_bytes())
        (out/'gamma_predictor.pt').write_bytes(gamma_path.read_bytes())
        (out/'gamma_pretraining.json').write_bytes((gamma_path.parent/'results.json').read_bytes())
        (out/'gamma_pretraining_config.json').write_bytes((gamma_path.parent/'config.json').read_bytes())
    records = json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    completed = {(r['method'],r['seed']) for r in records}
    for seed in range(args.seeds):
        for method in args.methods:
            if (method,seed) in completed:
                continue
            torch.manual_seed(seed)
            model = GammaGFModel(method,gamma_state=gamma_state)
            energy = make_energy(seed)
            init = {'initial_state_sha256':state_hash(model.state_dict()),
                    'initial_encoder_sha256':state_hash(model.encoder.state_dict()),
                    'initial_first_expert_parameters_sha256':state_hash(dict(model.experts[0].named_parameters())),
                    'initial_energy_sha256':state_hash(energy.state_dict())}
            model, energy = model.to(device), energy.to(device)
            if device.type=='cuda':
                torch.cuda.manual_seed_all(seed)
            ic = y[ix['train'],0]
            with torch.no_grad():
                initial_gamma = model.gamma_values(ic).clone()
                if model.charts==1:
                    torch.testing.assert_close(initial_gamma,torch.ones_like(initial_gamma),rtol=0,atol=0)
            ssv_hash=state_hash(model.ssv.state_dict()) if model.ssv is not None else None
            frozen = model.get_scales().clone()
            torch.testing.assert_close(frozen, torch.full_like(frozen,.5), rtol=0,atol=0)
            variables=[p for p in model.parameters() if p.requires_grad]+list(energy.parameters())
            opt=torch.optim.AdamW(variables, lr=.001, weight_decay=1e-6)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
            weights=model.weights(x)[:,None].expand(-1,5,-1).detach()
            best, selected_step, best_state, best_energy = float('inf'), None, None, None
            history, gf_scale = [], None
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start=time.perf_counter()
            print('START',method,seed,'parameters',sum(p.numel() for p in variables),flush=True)
            for step in range(1,args.steps+1):
                model.train()
                pred,parts=model(ic,t[:5],x,components=True)
                data_value=loss(pred,y[ix['train'],:5],list(OBS))
                ov=overlap_loss(parts,weights,ic) if model.charts==2 else data_value.new_zeros(())
                gf_terms=raw_gf_terms(model,energy,ic,t,gf_indices)
                raw=gf_terms.mean()
                if step==101:
                    gf_scale=max(float(raw.detach()),1e-12)
                gf_weight=0. if step<=100 else args.lambda_gf*min((step-100)/100,1.)
                normalized=raw/gf_scale if gf_weight else raw.new_zeros(())
                objective=data_value+args.overlap_lambda*ov+gf_weight*normalized
                opt.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
                scheduler.step()
                if step==1 or step%10==0 or step==args.steps:
                    model.eval()
                    with torch.no_grad():
                        valid=validation_loss(model(y[ix['valid'],0],t[:5],x),y[ix['valid'],:5])
                        gamma=model.gamma_values(ic)
                    values=torch.stack((objective.detach(),data_value.detach(),ov.detach(),raw.detach(),normalized.detach(),valid)).cpu().tolist()
                    if not all(np.isfinite(values)) or not torch.isfinite(gamma).all() or not (gamma>0).all():
                        raise RuntimeError(f'Nonfinite optimization or invalid gamma: {method},seed{seed},step{step}')
                    valid_value=values[-1]
                    h=dict(zip(('train','data','overlap','raw_gf','normalized_gf','valid'),values))
                    h.update(step=step,gf_weight=gf_weight,gf_per_chart=gf_terms.detach().cpu().tolist(),
                             gamma_mean=gamma.mean(0).cpu().tolist(),gamma_min=gamma.amin(0).cpu().tolist(),
                             gamma_max=gamma.amax(0).cpu().tolist())
                    history.append(h)
                    if valid_value<best:
                        best,selected_step=valid_value,step
                        best_state,best_energy=cpu_state(model),cpu_state(energy)
                if step%500==0:
                    sync(device)
                    elapsed=time.perf_counter()-start
                    save_json(out/'progress.json',{'method':method,'seed':seed,'step':step,
                              'elapsed_seconds':elapsed,'completed_runs':len(records),'gamma_mean':h['gamma_mean']})
                    print('TRAIN',method,seed,step,'early_valid',valid_value,'gamma',h['gamma_mean'],
                          'seconds',round(elapsed,1),flush=True)
            sync(device)
            seconds=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
            model.load_state_dict(best_state)
            energy.load_state_dict(best_energy)
            torch.testing.assert_close(model.get_scales(),frozen,rtol=0,atol=0)
            with torch.no_grad():
                torch.testing.assert_close(model.gamma_values(ic),initial_gamma,rtol=0,atol=0)
            if model.ssv is not None:
                assert state_hash(model.ssv.state_dict())==ssv_hash
                assert all(not p.requires_grad and p.grad is None for p in model.ssv.parameters())
            record=init | {'method':method,'seed':seed,'charts':model.charts,'width':128,
                'parameters':sum(p.numel() for p in variables),
                'backbone_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'energy_parameters':sum(p.numel() for p in energy.parameters()),
                'frozen_gamma_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
                'steps':args.steps,'selected_step':selected_step,'seconds':seconds,'peak_gpu_memory_bytes':peak,
                'scales':frozen.cpu().tolist(),'scales_unchanged':True,'gamma_unchanged':True,
                'gamma_predictor_sha256':ssv_hash,'gamma_before_latent_training':initial_gamma.cpu().tolist(),
                'overlap_lambda':args.overlap_lambda if model.charts==2 else 0.,
                'lambda_gf':args.lambda_gf,'gf_scale':gf_scale,'history':history}
            record['test']=evaluate_test(model,y,ix,x,t,out,method,seed,device)
            record['gamma']=gamma_report(model,y,ix)
            torch.save({'model':best_state,'energy':best_energy,'method':method,'seed':seed},out/f'{method}_{seed}.pt')
            records.append(record)
            save_json(out/'results.json',records)
            print('DONE',method,seed,'test_long',record['test']['long'],
                  'seconds',round(seconds,1),'peak_MiB',round(peak/2**20,1),flush=True)
    save_json(out/'progress.json',{'status':'complete','runs':len(records)})
    print('COMPLETE',len(records),'runs',out,flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--overlap-lambda',type=float,default=1.)
    parser.add_argument('--lambda-gf',type=float,default=1e-4)
    parser.add_argument('--gamma-source',default='runs/two_gaussian_gamma_calibration/gamma_predictor.pt')
    parser.add_argument('--out',default='runs/two_gaussian_main_cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.steps,args.seeds)<1 or min(args.overlap_lambda,args.lambda_gf)<0 or len(set(args.methods))!=len(args.methods):
        parser.error('Positive steps/seeds, nonnegative penalties and unique methods required')
    run(args)
