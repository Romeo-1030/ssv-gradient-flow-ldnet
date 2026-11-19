"""Paired two-chart experiment: early-only vs early+future consistency, no GF.

Both arms use the same full 49-step rollout and identical initialization. Only
the weight of the separately averaged future consistency term changes (0/1).
Training and checkpoint selection receive the three observed fields only.
The independent, previously calibrated gamma predictor remains fully frozen.
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
from gamma_gf_model import GammaGFModel
from moving_charts import overlap_loss, state_hash
from original_scale import OBS, cpu_state, diagnostics, save_json, sync
from original_scale_model import ORIGINAL_SOURCE
from two_gaussian_data import make_data, fingerprint, boundary_report, PROTOCOL


METHODS=('early_only','early_future')
FUTURE_WEIGHTS={'early_only':0.,'early_future':1.}


class FrozenTwoChartModel(GammaGFModel):
    """The existing two-chart backbone contains no energy/GF network itself."""
    def __init__(self,gamma_state=None):
        super().__init__('charts2_gamma_gf',gamma_state=gamma_state)
        self.method='two_charts_no_gf'


def future_indices(t):
    return torch.nonzero(t>.5,as_tuple=False).flatten().cpu().tolist()


def objective_terms(model,ic,observed,x,t,future_weight):
    """No future targets are accepted by this interface; ordinary backprop only."""
    if observed.shape!=(len(ic),len(OBS),len(x)):
        raise ValueError('Only the three labelled fields are accepted')
    indices=future_indices(t)
    if not indices:
        raise ValueError('Full rollout with at least one future time is required')
    pred,parts=model(ic,t,x,components=True)
    weights=model.weights(x)[:,None].expand(-1,len(t),-1).detach()
    data=loss(pred[:,list(OBS)],observed)
    early=overlap_loss(parts,weights,ic,indices=OBS)
    future=overlap_loss(parts,weights,ic,indices=indices)
    total=data+early
    if future_weight:
        total=total+future_weight*future
    return total,{'data':data,'early_consistency':early,'future_consistency':future}


def observed_validation(model,ic,observed,x,t):
    pred=model(ic,t[:5],x)[:,list(OBS)]
    norm=observed[:,0].square().sum(-1).clamp_min(1e-6)
    return ((pred-observed).square().sum(-1)/norm[:,None]).sqrt().mean()


def train_one(args,method,seed,gamma_state,train_ic,train_observed,valid_ic,valid_observed,x,t,out):
    """This function cannot access training/validation/test future labels."""
    device=x.device
    torch.manual_seed(seed)
    model=FrozenTwoChartModel(gamma_state)
    init_hash=state_hash(model.state_dict())
    encoder_hash=state_hash(model.encoder.state_dict())
    model=model.to(device)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==138866
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad)==3122
    assert not any('energy' in name for name,_ in model.named_parameters())
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=1e-5)
    scale_before=model.get_scales().clone()
    torch.testing.assert_close(scale_before,torch.full_like(scale_before,.5),rtol=0,atol=0)
    gamma_hash=state_hash(model.ssv.state_dict())
    with torch.no_grad():
        gamma_before=model.gamma_values(train_ic).clone()
    fw=FUTURE_WEIGHTS[method]
    history=[]
    best,best_state,selected_step=float('inf'),None,None
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start=time.perf_counter()
    print('START',method,seed,'trainable_parameters',sum(p.numel() for p in params),flush=True)
    previous_log_seconds=0.
    for step in range(1,args.steps+1):
        model.train()
        objective,terms=objective_terms(model,train_ic,train_observed,x,t,fw)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(params,5.)
        optimizer.step()
        scheduler.step()
        if step==1 or step%10==0 or step==args.steps:
            model.eval()
            with torch.no_grad():
                valid=observed_validation(model,valid_ic,valid_observed,x,t)
            values={k:float(v.detach()) for k,v in terms.items()}
            values.update(step=step,train=float(objective.detach()),valid=float(valid),future_weight=fw)
            if not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError(f'Nonfinite optimization: {method}, seed{seed}, step{step}')
            history.append(values)
            if values['valid']<best:
                best,selected_step,best_state=values['valid'],step,cpu_state(model)
        if step%250==0 or step==args.steps:
            sync(device)
            elapsed=time.perf_counter()-start
            progress={'status':'running','method':method,'seed':seed,'step':step,
                      'elapsed_seconds':elapsed,'interval_seconds':elapsed-previous_log_seconds,
                      'best_early_validation':best,'future_consistency':values['future_consistency']}
            save_json(out/'progress.json',progress)
            with (out/'progress_history.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(progress)+'\n')
            previous_log_seconds=elapsed
            print('TRAIN',method,seed,step,'early_valid',round(values['valid'],7),
                  'future_consistency',round(values['future_consistency'],7),'seconds',round(elapsed,1),flush=True)
    sync(device)
    seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
    model.load_state_dict(best_state)
    model.eval()
    torch.testing.assert_close(model.get_scales(),scale_before,rtol=0,atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model.gamma_values(train_ic),gamma_before,rtol=0,atol=0)
    assert state_hash(model.ssv.state_dict())==gamma_hash
    assert all(not p.requires_grad and p.grad is None for p in model.ssv.parameters())
    record={'method':method,'seed':seed,'steps':args.steps,'selected_step':selected_step,
        'initial_state_sha256':init_hash,'initial_encoder_sha256':encoder_hash,
        'trainable_parameters':sum(p.numel() for p in params),'frozen_parameters':3122,
        'energy_parameters':0,'gf_evaluations':0,'early_weight':1.,'future_weight':fw,
        'gamma_predictor_sha256':gamma_hash,'gamma_before_training':gamma_before.cpu().tolist(),
        'scales':scale_before.cpu().tolist(),'scales_unchanged':True,'gamma_unchanged':True,
        'seconds':seconds,'peak_gpu_memory_bytes':peak,'history':history}
    return model,best_state,record


def evaluate(model,ic,truth,x,t,ids,out,method,seed):
    device=x.device
    model.eval()
    with torch.no_grad():
        sync(device)
        start=time.perf_counter()
        pred,parts=model(ic,t,x,components=True)
        sync(device)
        seconds=time.perf_counter()-start
        if not torch.isfinite(pred).all() or not torch.isfinite(parts).all():
            raise RuntimeError(f'Nonfinite test prediction: {method}, seed{seed}')
        pc,yc=pred.cpu(),truth.cpu()
        result=metrics(pc,yc,x.cpu(),t.cpu())
        result['initial']=result['curve'][0]
        result['label_times_error']=float(np.asarray(result['curve'])[list(OBS)].mean())
        result['prediction_seconds']=seconds
        result['diagnostics']=diagnostics(model,pred,parts,truth,x,t)
        pp,yy=pred.double(),truth.double()
        dx=float(x[1]-x[0])
        result['diagnostics']['mean_predicted_mass_curve']=(pp.sum(-1)*dx).mean(0).cpu().tolist()
        result['diagnostics']['field_norm_ratio_curve']=(pp.square().sum(-1)/yy.square().sum(-1).clamp_min(1e-12)).sqrt().mean(0).cpu().tolist()
        result['diagnostics']['negative_mass_curve']=(pp.clamp_max(0).neg().sum(-1)*dx).mean(0).cpu().tolist()
        relative=((pc.double()-yc.double()).square().sum(-1)/yc.double().square().sum(-1).clamp_min(1e-12)).sqrt()
        result['per_ic_long']=relative[:,t.cpu()>2].mean(-1).tolist()
        result['per_ic_final']=relative[:,-1].tolist()
        np.savez_compressed(out/f'{method}_{seed}_test.npz',prediction=pc.numpy(),components=parts.cpu().numpy(),ids=ids)
    return result


def run(args):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use the configured .venv-cuda')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=ROOT/args.out
    if (out/'config.json').exists() and not args.resume:
        raise FileExistsError('Preserving existing run; use a new --out or --resume')
    out.mkdir(parents=True,exist_ok=True)
    data=make_data()
    with np.load(ROOT/'runs/two_gaussian_main_cuda/data.npz') as previous:
        for name,value in data.items():
            np.testing.assert_array_equal(value,previous[name])
    calibration_path=ROOT/args.gamma_source
    calibration=torch.load(calibration_path,map_location='cpu',weights_only=True)
    assert calibration['config']['data_fingerprint']==fingerprint(data)
    x,t=[torch.tensor(data[k],device=device,dtype=torch.float32) for k in ('x','t')]
    all_ic=torch.tensor(data['u'][:,0],device=device)
    observed=torch.tensor(data['u'][:,list(OBS)],device=device)
    ix={s:np.flatnonzero(data['split']==s) for s in ('train','valid','test')}
    assert len(t)==49 and len(future_indices(t))==44
    source_names=('future_consistency.py','gamma_gf_model.py','original_scale_model.py',
                  'original_scale.py','experiment.py','moving_charts.py','two_stage.py','two_gaussian_data.py')
    config=vars(args)|{'comparison':'same two-chart architecture; only future consistency coefficient changes 0 -> 1',
        'latent':16,'width':128,'depth':3,'condition_dim':16,'ic_samples':192,
        'fourier_coefficients':33,'alpha':.5,'beta':.5,'centers':[-4.,4.],'gate_width':4.,
        'ssv':'pretrained independent IC-to-two-gamma predictor, frozen; identical calibration reused',
        'gamma_source_sha256':hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        'gamma_calibration_selection':calibration['selection'],'trainable_parameters':138866,'frozen_parameters':3122,
        'gf_enabled':False,'energy_network_instantiated':False,'gf_evaluations':0,
        'observed_times':t[list(OBS)].cpu().tolist(),'early_consistency_weight':1.,
        'future_consistency_weights':FUTURE_WEIGHTS,'future_times':t[future_indices(t)].cpu().tolist(),
        'rollout_times':t.cpu().tolist(),'rollout':'both arms integrate and decode the same full 49-state trajectory',
        'normalization':'each term separately averaged over its times and trajectories; divide spatial squared norm by initial field energy',
        'schedule':'future weight is 0 or 1 from first update; no warmup or horizon curriculum',
        'checkpoint_selection':'minimum mean per-IC initial-L2-normalized error on three early validation frames; every10 updates',
        'initialization':'paired identical model states; fresh latent/decoder/encoder weights, no GF-trained weights loaded',
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':1e-6,'clip_norm':5.,'eta_min':1e-5,'scheduler':'cosine'},
        'data_protocol':PROTOCOL,'data_fingerprint':fingerprint(data),'split_counts':{s:len(ids) for s,ids in ix.items()},
        'boundary_check':boundary_report(data),'torch':str(torch.__version__),'numpy':np.__version__,
        'python':platform.python_version(),'cuda_runtime':torch.version.cuda,'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        'threads':1,'tf32':False,'deterministic_algorithms':True,'synchronized_gpu_timing':True,
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},
        'original_train_sha256':hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest()}
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for k in ('steps','seeds','methods','data_fingerprint','gamma_source_sha256','source_sha256','original_train_sha256','device_name','torch'):
            if old[k]!=config[k]:
                raise ValueError(f'Resume mismatch: {k}')
    else:
        save_json(out/'config.json',config)
        np.savez_compressed(out/'data.npz',**data)
        snapshot=out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
        for n in source_names:
            (snapshot/n).write_bytes((ROOT/n).read_bytes())
        (snapshot/'upstream_train.py').write_bytes(ORIGINAL_SOURCE.read_bytes())
        (out/'gamma_predictor.pt').write_bytes(calibration_path.read_bytes())
    records=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    completed={(r['method'],r['seed']) for r in records}
    paired_initializations={r['seed']:r['initial_state_sha256'] for r in records}
    for seed in range(args.seeds):
        for method in args.methods:
            if (method,seed) in completed:
                continue
            model,best,record=train_one(args,method,seed,calibration['state'],
                all_ic[ix['train']],observed[ix['train']],all_ic[ix['valid']],observed[ix['valid']],x,t,out)
            if seed in paired_initializations:
                assert record['initial_state_sha256']==paired_initializations[seed]
            paired_initializations[seed]=record['initial_state_sha256']
            # Full reference fields first enter here, after model selection.
            truth=torch.tensor(data['u'][ix['test']],device=device)
            record['test']=evaluate(model,all_ic[ix['test']],truth,x,t,ix['test'],out,method,seed)
            with torch.no_grad():
                record['gamma']={s:{'ids':ids.tolist(),'values':model.gamma_values(all_ic[ids]).cpu().tolist()} for s,ids in ix.items()}
            del truth
            torch.save({'model':best,'method':method,'seed':seed,'future_weight':FUTURE_WEIGHTS[method]},out/f'{method}_{seed}.pt')
            records.append(record)
            save_json(out/'results.json',records)
            print('DONE',method,seed,'test_long',record['test']['long'],'test_final',record['test']['final'],
                  'seconds',round(record['seconds'],1),'peak_MiB',round(record['peak_gpu_memory_bytes']/2**20,1),flush=True)
    save_json(out/'progress.json',{'status':'complete','runs':len(records)})
    print('COMPLETE',len(records),'runs',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--gamma-source',default='runs/two_gaussian_gamma_calibration/gamma_predictor.pt')
    parser.add_argument('--out',default='runs/future_consistency_cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.steps,args.seeds)<1 or len(set(args.methods))!=len(args.methods):
        parser.error('Positive steps/seeds and unique methods required')
    run(args)
