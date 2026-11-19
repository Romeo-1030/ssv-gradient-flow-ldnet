"""Matched SSV x GF comparison on ordinary two-Gaussian initial conditions.

Reuse the already completed shared-SSV+GF arm and its validation-selected
gamma. Train latent, latent+GF, and shared-SSV without GF from matched fresh
weights on identical six-frame data. No new gamma search or test tuning.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

import ic_gamma_experiment as base
from ic_gamma_data import SIX
from ic_gamma_model import pack_inputs
from ordinary_full_comparison_fit import fit_final, evaluate_and_save
from original_scale_model import ORIGINAL_SOURCE, original
from retry_json import save_json
from two_gaussian_data import fingerprint
from shared_gamma_domain16 import sha, load_data, CORE_SOURCES
from ordinary_full_comparison_report import build_report
from ordinary_full_comparison_freeze import build_diagnostics

ROOT = Path(__file__).resolve().parent
METHODS = ('latent','latent_gf','ssv','ssv_gf')
NEW_METHODS = METHODS[:3]


def load_source(source):
    source = Path(source).resolve()
    config = base.read(source/'config.json')
    rows = base.read(source/'results.json')
    assert base.read(source/'progress.json')['status']=='complete'
    assert config['domain']==[-16.,16.] and config['prediction_horizon']==10.
    assert config['steps']==3000 and config['seeds']==4 and not config['smoke_only']
    assert config['treatment']=='ordinary_two_gaussian_shared_dictionary_gf'
    assert len(rows)==4 and {r['seed'] for r in rows}==set(range(4))
    assert all(r['method']=='ssv_gf' for r in rows)
    for name in CORE_SOURCES:
        if sha(ROOT/name) != config['source_sha256'][name]:
            raise RuntimeError(f'Core source changed since SSV+GF experiment: {name}')
    assert sha(ORIGINAL_SOURCE)==config['source_sha256']['upstream_train.py']
    data = load_data(source/'data.npz')
    selection = base.read(source/'shared_gamma_selection.json')
    assert selection['data_fingerprint']==fingerprint(data)==config['data_fingerprint']
    gamma = float(selection['selected_gamma'])
    assert gamma==4. and selection['completed_fits']==21
    assert not selection['test_used'] and not selection['future_labels_after_1_25_used']
    names = ['config.json','data.npz','results.json','audit.json',
        'shared_gamma_selection.json','gamma_selection.json','data_audit.json']
    for row in rows:
        names += [row['checkpoint'],row['prediction'],row['energy_surface'],f"ssv_gf_{row['seed']}_history.json"]
    hashes = {name:sha(source/name) for name in names}
    return data,config,rows,selection,hashes


def audit_rows(rows,seeds,smoke=False):
    expected = {(m,s) for m in METHODS for s in range(seeds)}
    assert len(rows)==len(expected) and {(r['method'],r['seed']) for r in rows}==expected
    for seed in range(seeds):
        group = {r['method']:r for r in rows if r['seed']==seed}
        assert len({r['initial_parameters_sha256'] for r in group.values()})==1
        assert len({r['initial_energy_sha256'] for r in group.values() if r['gf']})==1
        for name,row in group.items():
            physical,use_gf = name in ('latent','latent_gf'), name.endswith('_gf')
            assert row['physical']==physical and row['gf']==use_gf
            assert row['backbone_parameters']==91329 and row['energy_parameters']==(4417 if use_gf else 0)
            assert row['gamma_frozen'] and row['fourier_frequencies']==16 and row['decoder_extent']==16.
            assert row['scales']==([[0.,0.]] if physical else [[.5,.5]])
            assert row['label_times']==[0.,.25,.5,.75,1.,1.25]
            if use_gf:
                assert row['gf_times']==group['ssv_gf']['gf_times']
                assert row['gf_spatial_extent']==16. and row['gf_collocation']==24
            if not smoke:
                assert row['steps']==3000
        if not smoke:
            # The same supervised objectives and initial weights must agree
            # before the GF weight becomes nonzero, within each coordinate system.
            warmup = lambda r:[(h['step'],h['data'],h['valid']) for h in r['history'] if h['step']<=100]
            assert warmup(group['latent'])==warmup(group['latent_gf'])
            assert warmup(group['ssv'])==warmup(group['ssv_gf'])


def run(args):
    out,source = (ROOT/args.out).resolve(),(ROOT/args.source).resolve()
    if out==source:
        raise ValueError('A separate comparison directory is required')
    if not args.smoke and (args.steps!=3000 or args.seeds!=4):
        raise ValueError('Production requires four seeds and 3000 updates per new fit')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    data,source_config,source_rows,selection,hashes=load_source(source)
    gamma=float(selection['selected_gamma'])
    arguments=vars(args).copy()
    arguments.pop('resume')
    manifest=base.source_manifest()
    manifest['start_ordinary_full_comparison.ps1']=sha(ROOT/'start_ordinary_full_comparison.ps1')
    config={'arguments':arguments,'created_utc':base.utc(),'source_run':str(source),
        'source_sha256':manifest,'protected_source_files':hashes,
        'data_fingerprint':fingerprint(data),'data_protocol':source_config['data_protocol'],
        'domain':[-16.,16.],'nx':192,'fourier_frequencies':16,'prediction_horizon':10.,
        'methods':list(METHODS),'new_methods':list(NEW_METHODS),'reused_method':'ssv_gf',
        'seeds':args.seeds,'steps':args.steps,'smoke_only':args.smoke,'std_ddof':0,
        'shared_gamma':gamma,'gamma_selection':'Reuse completed training/validation dictionary choice; freeze same gamma in both SSV arms; no reselection using GF or test errors.',
        'training':'Identical six labels, architecture, paired optimizer seeds, optimizer and validation criterion; SSV and GF are crossed in a 2x2 comparison.',
        'inference_inputs':'All arms condition on the initial field only. SSV adds the one shared clock learned on train/validation trajectories.',
        'primary_metric':'Original-paper pooled full-domain current-time RelRMSE.',
        'paper_scope':'Non-Gaussian extension with six observed frames and validation-selected shared gamma; not an unchanged reproduction of the original three-frame Gaussian experiment.',
        'freeze_diagnostic':'After training, retain each SSV model state at t=1.25 and continue only its SSV scaling. Prediction counterfactual, not a separately trained baseline.',
        'torch':str(torch.__version__),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}
    if (out/'config.json').exists():
        if not args.resume:
            raise FileExistsError('Run already exists; explicit resume required after worker ended')
        saved=base.read(out/'config.json')
        for key in ('arguments','source_sha256','protected_source_files'):
            if saved[key]!=config[key]:
                raise RuntimeError(f'Resume changed frozen {key}')
        assert sha(out/'data.npz')==hashes['data.npz']
    else:
        snapshot=out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in manifest:
            shutil.copy2(ORIGINAL_SOURCE if name=='upstream_train.py' else ROOT/name,snapshot/name)
        for name in ('data.npz','shared_gamma_selection.json','gamma_selection.json','data_audit.json'):
            shutil.copy2(source/name,out/name)
        save_json(out/'config.json',config)
    rows=base.read(out/'results.json',[])
    for source_row in source_rows:
        if source_row['seed']>=args.seeds or any(r['method']=='ssv_gf' and r['seed']==source_row['seed'] for r in rows):
            continue
        names=[source_row['checkpoint'],source_row['prediction'],source_row['energy_surface'],f"ssv_gf_{source_row['seed']}_history.json"]
        for name in names:
            shutil.copy2(source/name,out/name)
            assert sha(out/name)==hashes[name]
        row=copy.deepcopy(source_row)
        row.update(reused_from=str(source),new_training_performed=False)
        rows.append(row)
    save_json(out/'results.json',rows)
    base.progress(out,'preparing_comparison',completed_runs=len(rows),expected_runs=4*args.seeds,shared_gamma=gamma)
    build_report(out)
    ids={s:np.flatnonzero(data['split']==s) for s in ('train','valid','test')}
    x=torch.tensor(data['x'],dtype=torch.float32,device=device)
    t=torch.tensor(data['t'],dtype=torch.float32,device=device)
    fields={s:torch.tensor(data['u'][idx,0],device=device) for s,idx in ids.items()}
    labels={s:torch.tensor(data['u'][ids[s]][:,list(SIX)],device=device) for s in ('train','valid')}
    for seed in range(args.seeds):
        for method in NEW_METHODS:
            if any(r['method']==method and r['seed']==seed for r in rows):
                continue
            physical=method in ('latent','latent_gf')
            inputs={s:pack_inputs(fields[s],np.full(len(idx),1. if physical else gamma)) for s,idx in ids.items()}
            model,energy,row,checkpoint=fit_final(args,method,seed,
                {s:inputs[s] for s in ('train','valid')},labels,x,t,out)
            ref=next(r for r in source_rows if r['seed']==seed)
            assert row['initial_parameters_sha256']==ref['initial_parameters_sha256']
            if energy is not None:
                assert row['initial_energy_sha256']==ref['initial_energy_sha256']
            row.update(treatment=method,shared_gamma=None if physical else gamma,
                gamma_selection_file=None if physical else 'shared_gamma_selection.json',new_training_performed=True)
            checkpoint.update(treatment=method,shared_gamma=None if physical else gamma,
                gamma_source='Physical coordinates; gamma input ignored' if physical else 'shared_gamma_selection.json; frozen shared gamma')
            row=evaluate_and_save(model,energy,row,checkpoint,data,ids['test'],inputs['test'],x,t,out)
            with np.load(out/row['prediction']) as saved:
                curve=original.full_rel_curve(torch.from_numpy(saved['truth']).double(),torch.from_numpy(saved['pred']).double())
            np.testing.assert_allclose(curve.numpy(),row['metrics']['curve'],rtol=2e-7,atol=1e-9)
            row['metric_matches_original_full_rel_curve']=True
            rows.append(row)
            save_json(out/'results.json',rows)
            build_report(out)
            print('COMPARISON DONE',method,seed,'observed',row['metrics']['observed'],
                'extrapolation',row['metrics']['extrapolation'],'final',row['metrics']['final'],flush=True)
            del model,energy,checkpoint
            if device.type=='cuda':
                torch.cuda.empty_cache()
    audit_rows(rows,args.seeds,args.smoke)
    base.progress(out,'frozen_latent_diagnostics',completed_runs=len(rows),expected_runs=4*args.seeds)
    build_diagnostics(out,methods=('ssv','ssv_gf'),device='cpu')
    for name,digest in hashes.items():
        assert sha(source/name)==digest, f'Protected source changed: {name}'
    save_json(out/'audit.json',{'complete_runs':len(rows),'expected_runs':4*args.seeds,
        'new_training_runs':3*args.seeds,'reused_runs':args.seeds,
        'same_data_bytes':sha(out/'data.npz')==hashes['data.npz'],
        'paired_initialization_passed':True,'GF_warmup_equivalence_passed':not args.smoke,
        'same_frozen_gamma_in_both_SSV_arms':True,'shared_gamma':gamma,
        'only_six_labels_used':True,'test_inputs_initial_field_only':True,
        'checkpoint_replay_passed':True,'original_metric_verified':True,
        'source_artifacts_unchanged':True,'frozen_latent_diagnostic_complete':True,
        'smoke_only':args.smoke,'completed_utc':base.utc()})
    save_json(out/'progress.json',{'status':'complete','phase':'complete','pid':os.getpid(),
        'completed_runs':len(rows),'completed_utc':base.utc(),'shared_gamma':gamma})
    build_report(out)
    (out/'stop_awake').touch()
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='runs/ordinary_shared_gamma_domain16_t10_cuda')
    parser.add_argument('--out',default='runs/ordinary_full_comparison_domain16_t10_cuda')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.steps,args.seeds)<1 or args.seeds>4:
        parser.error('Positive counts and at most four matched seeds required')
    try:
        run(args)
    except BaseException as error:
        out=(ROOT/args.out).resolve()
        out.mkdir(parents=True,exist_ok=True)
        save_json(out/'progress.json',{'status':'failed','phase':'failed','pid':os.getpid(),
            'error':repr(error),'traceback':traceback.format_exc(),'failed_utc':base.utc()})
        (out/'stop_awake').touch()
        raise
