"""Unattended paired physical LDNet / fixed global SSV / global SSV+GF run.

Uses the existing two-Gaussian data, original-size backbone, three early labels,
and prior optimizer/selection protocol. No chart-consistency term is present.
SSV is fixed alpha=beta=.5, gamma=1: a representation ablation, not a rerun of
the paper's learned-exponent identification. A physical baseline also removes
the logarithmic clock. Outputs, source snapshots and summaries stay in --out.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from experiment import ROOT, loss, metrics
from future_consistency import evaluate, observed_validation
from gamma_gf import make_energy
from moving_charts import state_hash
from original_scale import OBS, cpu_state, save_json, sync, make_gf_indices, raw_gf_loss
from original_scale_model import OriginalScaleModel, ORIGINAL_SOURCE
from two_gaussian_data import make_data, fingerprint, boundary_report, PROTOCOL

METHODS=('latent','global_ssv','global_ssv_gf')
NAMES={'latent':'Latent (physical time)','global_ssv':'Latent + fixed global SSV',
       'global_ssv_gf':'Latent + fixed global SSV + original GF'}


class GlobalBaseline(OriginalScaleModel):
    def __init__(self,method):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(charts=1)
        self.method=method
        if method=='latent':
            self.experts[0].mode='physical'
            self.set_scales([[0.,0.]])
        # The inherited SSV clock is log(1+t), with gamma identically one.


def supervised_loss(model,ic,observed,x,t):
    if observed.shape!=(len(ic),3,len(x)):
        raise ValueError('Training accepts only three early labelled fields')
    return loss(model(ic,t[:5],x)[:,list(OBS)],observed)


def train_one(args,method,seed,ic,observed,valid_ic,valid_observed,x,t,out):
    device=x.device
    torch.manual_seed(seed)
    model=GlobalBaseline(method)
    initial_hash=state_hash(dict(model.named_parameters()))
    energy=make_energy(seed) if method=='global_ssv_gf' else None
    energy_hash=state_hash(energy.state_dict()) if energy is not None else None
    model=model.to(device)
    if energy is not None:
        energy=energy.to(device)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    backbone=list(model.parameters())
    assert sum(p.numel() for p in backbone)==91073
    parameters=backbone+(list(energy.parameters()) if energy is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=1e-5)
    scales=model.get_scales().clone()
    gf_indices=make_gf_indices(t,32) if energy is not None else None
    history=[]; best=float('inf'); selected=None; best_state=None; best_energy=None; gf_scale=None
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device); start=time.perf_counter()
    print('START',method,seed,'parameters',sum(p.numel() for p in parameters),flush=True)
    for step in range(1,args.steps+1):
        model.train()
        data_value=supervised_loss(model,ic,observed,x,t)
        raw=raw_gf_loss(model,energy,ic,t,gf_indices,24) if energy is not None else data_value.new_zeros(())
        if energy is not None and step==101:
            gf_scale=max(float(raw.detach()),1e-12)
        weight=1e-4*min((step-100)/100,1.) if energy is not None and step>100 else 0.
        normalized=raw/gf_scale if weight else raw.new_zeros(())
        objective=data_value+weight*normalized
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(backbone,5.)
        optimizer.step(); scheduler.step()
        if step==1 or step%10==0 or step==args.steps:
            model.eval()
            with torch.no_grad():
                val=observed_validation(model,valid_ic,valid_observed,x,t)
            row={'step':step,'train':float(objective.detach()),'data':float(data_value.detach()),
                 'raw_gf':float(raw.detach()),'normalized_gf':float(normalized.detach()),
                 'gf_weight':weight,'valid':float(val)}
            if not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError(f'Nonfinite training: {method}, seed {seed}, step {step}')
            history.append(row)
            if row['valid']<best:
                best=row['valid']; selected=step; best_state=cpu_state(model)
                best_energy=cpu_state(energy) if energy is not None else None
        if step==1 or step%250==0 or step==args.steps:
            sync(device)
            progress={'status':'training','pid':os.getpid(),'method':method,'seed':seed,'step':step,
                      'steps_per_run':args.steps,'seconds':time.perf_counter()-start,'best_early_validation':best}
            save_json(out/'progress.json',progress)
            print('TRAIN',method,seed,step,'early_valid',best,'seconds',round(progress['seconds'],1),flush=True)
    sync(device)
    seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
    model.load_state_dict(best_state); model.eval()
    torch.testing.assert_close(model.get_scales(),scales,rtol=0,atol=0)
    with torch.no_grad():
        p=model.encode(ic)
        torch.testing.assert_close(model.experts[0].gamma(p),torch.ones(len(ic),device=device),rtol=0,atol=0)
    record={'method':method,'seed':seed,'steps':args.steps,'selected_step':selected,'history':history,
            'initial_parameters_sha256':initial_hash,'initial_energy_sha256':energy_hash,
            'parameters':sum(p.numel() for p in parameters),'backbone_parameters':91073,
            'energy_parameters':4417 if energy is not None else 0,'gf_scale':gf_scale,
            'gf_evaluations':args.steps if energy is not None else 0,
            'scales':scales.cpu().tolist(),'scales_unchanged':True,'gamma':1.,
            'seconds':seconds,'peak_gpu_memory_bytes':peak}
    checkpoint={'model':best_state,'energy':best_energy,'method':method,'seed':seed}
    return model,checkpoint,record


def finish(args,out,records,data,x,t):
    """Automatic final replay/metric checks; no assistant monitoring required."""
    expected={(m,s) for m in METHODS for s in range(args.seeds)}
    assert {(r['method'],r['seed']) for r in records}==expected and len(records)==len(expected)
    ids=np.flatnonzero(data['split']=='test')
    valid_ids=np.flatnonzero(data['split']=='valid')
    ic=torch.tensor(data['u'][:,0],device=x.device)
    summary={}; predictions={m:[] for m in METHODS}
    for seed in range(args.seeds):
        hashes={r['initial_parameters_sha256'] for r in records if r['seed']==seed}
        assert len(hashes)==1
    for r in records:
        method,seed=r['method'],r['seed']
        assert r['steps']==args.steps and r['scales_unchanged']
        selected=min(r['history'],key=lambda h:h['valid'])
        assert selected['step']==r['selected_step']
        for h in r['history']:
            weight=1e-4*min((h['step']-100)/100,1.) if method=='global_ssv_gf' and h['step']>100 else 0.
            assert h['gf_weight']==weight
            np.testing.assert_allclose(h['train'],h['data']+weight*h['normalized_gf'],rtol=2e-6,atol=1e-8)
        checkpoint=torch.load(out/f'{method}_{seed}.pt',map_location='cpu',weights_only=True)
        model=GlobalBaseline(method).to(x.device)
        model.load_state_dict(checkpoint['model']); model.eval()
        assert (checkpoint['energy'] is not None)==(method=='global_ssv_gf')
        with torch.no_grad():
            pred=model(ic[ids],t,x)
            val=observed_validation(model,ic[valid_ids],torch.tensor(data['u'][valid_ids][:,list(OBS)],device=x.device),x,t)
        np.testing.assert_allclose(float(val),selected['valid'],rtol=0,atol=1e-8)
        saved=dict(np.load(out/f'{method}_{seed}_test.npz'))
        np.testing.assert_array_equal(ids,saved['ids'])
        np.testing.assert_array_equal(pred.cpu().numpy(),saved['prediction'])
        recalculated=metrics(torch.tensor(saved['prediction']),torch.tensor(data['u'][ids]),x.cpu(),t.cpu())
        for key in ('curve','short','long','final'):
            np.testing.assert_allclose(recalculated[key],r['test'][key],rtol=0,atol=1e-12)
        predictions[method].append(saved['prediction'])
    for method in METHODS:
        rows=sorted([r for r in records if r['method']==method],key=lambda r:r['seed'])
        item={}
        for key in ('initial','label_times_error','short','long','final'):
            values=np.asarray([r['test'][key] for r in rows])
            item[key]={'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)) if len(values)>1 else 0.,'values':values.tolist()}
        item['training_seconds']=[r['seconds'] for r in rows]
        summary[method]=item
    for before,after in (('latent','global_ssv'),('global_ssv','global_ssv_gf')):
        delta=np.asarray(summary[after]['long']['values'])-summary[before]['long']['values']
        summary[f'{after}_minus_{before}']={'long_error_differences':delta.tolist(),'improved_seeds':int((delta<0).sum())}
    save_json(out/'summary.json',summary)
    save_json(out/'audit.json',{'passed':True,'runs':len(records),'updates':args.steps*len(records),
        'paired_initial_parameters':True,'early_only_checkpoint_selection':True,'fixed_ssv':True,
        'same_device_checkpoint_predictions_bitwise_equal':True,'metrics_recomputed':True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    colors=('#777777','#2364AA','#D65A31')
    for method,color in zip(METHODS,colors):
        curves=100*np.asarray([r['test']['curve'] for r in records if r['method']==method])
        for curve in curves:
            ax.plot(data['t'],curve,color=color,alpha=.2,lw=.6)
        ax.plot(data['t'],curves.mean(0),color=color,label=NAMES[method],lw=1.8)
    ax.axvline(.5,color='gray',ls='--',lw=1)
    ax.axhline(100,color='gray',ls=':',lw=1)
    ax.set(xlabel='Physical time t',ylabel='Test relative L2 error (%)',yscale='log',
           title='Two-Gaussian ICs: individual seeds and mean')
    ax.legend(fontsize=8)
    fig.savefig(out/'errors.png',dpi=170); fig.savefig(out/'errors.pdf'); plt.close(fig)
    lines=['# Global baseline comparison on two-Gaussian initial conditions','',
        f'Completed {len(records)} runs, {args.steps} updates each, {args.seeds} paired seeds. CUDA: {torch.cuda.get_device_name(x.device) if x.device.type=="cuda" else "CPU"}.',
        '', 'Identical dataset and splits (36/12/24) as the preceding chart experiment. Labels only at t=0,.25,.5; evaluation to t=6. Initial backbone parameters match across all three arms.',
        '', 'Pure latent uses physical time t and unscaled x/u. Global SSV fixes alpha=beta=.5 and gamma=1, with clock log(1+t). This tests fixed global SSV, not learned-exponent identification. Both share the original latent=16, width=128, depth=3, 33-coefficient Fourier decoder and 192-to-16 IC encoder (91,073 backbone parameters).',
        '', 'GF adds the original 4,417-parameter energy network, 32 rollout times and 24 sampled xi points; weight zero through update100, detached normalization at101, ramp to 1e-4 at200. Alpha/beta/gamma remain fixed. No chart-consistency loss is used. No future truth is used for training or checkpoint selection.',
        '', 'Optimizer: full-batch AdamW, lr=.001, decay=1e-6, backbone gradient clipping=5, cosine lr to1e-5. Selection: minimum three-label validation error at update1 and every10 updates.',
        '', 'Relative L2 errors below are mean +/- sample SD over seeds, on 24 held-out ICs. SD describes initialization variation, not uncertainty across datasets.','',
        '| Method | Three labels (%) | (.5,2] (%) | (2,6] (%) | t=6 (%) |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        values=[f"{100*summary[method][k]['mean']:.2f} +/- {100*summary[method][k]['sample_sd']:.2f}" for k in ('label_times_error','short','long','final')]
        lines.append('| '+NAMES[method]+' | '+' | '.join(values)+' |')
    lines+=['','![Error curves](errors.png)','',
        f"SSV improves long-time error in {summary['global_ssv_minus_latent']['improved_seeds']}/{args.seeds} paired seeds; adding GF improves it in {summary['global_ssv_gf_minus_global_ssv']['improved_seeds']}/{args.seeds}.",
        '',f"Total synchronized training wall time: {sum(r['seconds'] for r in records)/60:.2f} minutes (includes any host scheduling/pauses).",
        '', 'Automatic checks passed: matched initialization, early-only selection, frozen scalings, same-device checkpoint replay and independently recomputed metrics. Raw predictions, checkpoints, histories, config and source snapshots are in this directory. GF uses unlabelled time coordinates through t=6; no performance beyond that horizon is asserted.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and not args.resume:
        raise FileExistsError('Preserving existing experiment; choose new --out or --resume')
    data=make_data()
    with np.load(ROOT/'runs/two_gaussian_main_cuda/data.npz') as previous:
        for key in data:
            np.testing.assert_array_equal(data[key],previous[key])
    names=('global_baselines.py','test_global_baselines.py','start_global_baselines.ps1','original_scale_model.py','original_scale.py',
           'gamma_gf.py','gamma_gf_model.py','future_consistency.py','experiment.py','moving_charts.py','two_stage.py','two_gaussian_data.py')
    config=vars(args)|{'methods':list(METHODS),'data_protocol':PROTOCOL,'data_fingerprint':fingerprint(data),
        'boundary_check':boundary_report(data),'alpha_beta_ssv':[.5,.5],'gamma_ssv':1.,
        'latent_clock':'t','ssv_clock':'log(1+t)','ssv_training':'none; fixed coefficients',
        'gf_weight':1e-4,'gf_times':32,'gf_collocation':24,'consistency_weight':0.,
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':1e-6,'eta_min':1e-5,'clip_backbone':5.},
        'labels':[0.,.25,.5],'selection':'three-label validation, step1/every10/final; no future labels',
        'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version(),
        'cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        'deterministic':True,'tf32':False,'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        'upstream_train_sha256':hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest()}
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for key in ('steps','seeds','methods','data_fingerprint','source_sha256','upstream_train_sha256','torch','device_name'):
            if config[key]!=old[key]:
                raise ValueError(f'Resume mismatch: {key}')
    else:
        save_json(out/'config.json',config)
        np.savez_compressed(out/'data.npz',**data)
        source=out/'source_snapshot'; source.mkdir(exist_ok=True)
        for name in names:
            (source/name).write_bytes((ROOT/name).read_bytes())
        (source/'upstream_train.py').write_bytes(ORIGINAL_SOURCE.read_bytes())
    save_json(out/'progress.json',{'status':'starting','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
    records=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    completed={(r['method'],r['seed']) for r in records}
    x,t=[torch.tensor(data[k],device=device,dtype=torch.float32) for k in ('x','t')]
    ic=torch.tensor(data['u'][:,0],device=device)
    observed=torch.tensor(data['u'][:,list(OBS)],device=device)
    ix={s:np.flatnonzero(data['split']==s) for s in ('train','valid','test')}
    paired={r['seed']:r['initial_parameters_sha256'] for r in records}
    for seed in range(args.seeds):
        for method in METHODS:
            if (method,seed) in completed:
                continue
            model,checkpoint,record=train_one(args,method,seed,ic[ix['train']],observed[ix['train']],
                ic[ix['valid']],observed[ix['valid']],x,t,out)
            if seed in paired:
                assert paired[seed]==record['initial_parameters_sha256']
            paired[seed]=record['initial_parameters_sha256']
            torch.save(checkpoint,out/f'{method}_{seed}.pt')
            # Test future fields first enter after early-only model selection.
            truth=torch.tensor(data['u'][ix['test']],device=device)
            record['test']=evaluate(model,ic[ix['test']],truth,x,t,ix['test'],out,method,seed)
            del truth
            records.append(record); save_json(out/'results.json',records)
            print('DONE',method,seed,'long',record['test']['long'],'final',record['test']['final'],flush=True)
    save_json(out/'progress.json',{'status':'verifying','pid':os.getpid(),'runs':len(records)})
    finish(args,out,records,data,x,t)
    save_json(out/'progress.json',{'status':'complete','runs':len(records),'optimizer_updates':args.steps*len(records),
                                'completed_utc':datetime.now(timezone.utc).isoformat()})
    print('COMPLETE',len(records),'runs',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',default='runs/global_baselines_cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.steps,args.seeds)<1:
        parser.error('Positive steps and seeds required')
    try:
        run(args)
    except Exception as error:
        destination=ROOT/args.out
        destination.mkdir(parents=True,exist_ok=True)
        # Existing completed results are never overwritten by an accidental rerun.
        if not isinstance(error,FileExistsError):
            save_json(destination/'progress.json',{'status':'failed','pid':os.getpid(),'error':str(error)})
        (destination/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
