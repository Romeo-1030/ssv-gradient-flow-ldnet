"""Ordinary independent two-Gaussian ICs with the current shared-gamma method.

Keep the old seed71 mixture parameters and splits, sample their analytic
heat solutions on the current domain16/t10 grid, and repeat the unchanged
shared dictionary selection and GF final training. No IC is filtered or
rescaled to impose a reference-clock distribution.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

import ic_gamma_experiment as base
from ic_gamma_data import SIX
from ic_gamma_model import pack_inputs
from ic_gamma_domain16_fit import fit_final, evaluate_and_save
from original_scale_model import ORIGINAL_SOURCE, original
from retry_json import save_json
from two_gaussian_data import fingerprint
from shared_gamma_domain16 import sha, load_data, check_pair, CORE_SOURCES
from shared_gamma_domain16_selector import calibrate_shared
from ordinary_shared_gamma_data import prepare_data
from ordinary_shared_gamma_report import build_report

ROOT = Path(__file__).resolve().parent
CANDIDATES = [.125, .25, .5, 1., 2., 4., 8.]
SEARCH_SEEDS = [200, 201, 202]


def load_reference(reference):
    reference = Path(reference).resolve()
    config = base.read(reference/'config.json')
    rows = base.read(reference/'results.json')
    assert base.read(reference/'progress.json')['status'] == 'complete'
    assert config['domain'] == [-16.,16.] and config['prediction_horizon'] == 10.
    assert config['steps'] == config['selector_steps'] == 3000 and config['seeds'] == 4
    assert config['candidates'] == CANDIDATES and config['selector_seeds'] == SEARCH_SEEDS
    assert config['gamma_selection_weights'] == [1.,2.,3.]
    assert len(rows) == 4 and {r['seed'] for r in rows} == set(range(4))
    assert all(r['method'] == 'ssv_gf' for r in rows)
    for name in (*CORE_SOURCES, 'shared_gamma_domain16_selector.py', 'gamma_selection.py'):
        if sha(ROOT/name) != config['source_sha256'][name]:
            raise RuntimeError(f'Algorithm source changed since controlled run: {name}')
    assert sha(ORIGINAL_SOURCE) == config['source_sha256']['upstream_train.py']
    protected = ['config.json','data.npz','results.json','shared_gamma_selection.json']
    protected += [r[key] for r in rows for key in ('checkpoint','prediction')]
    hashes = {str(reference/name):sha(reference/name) for name in protected}
    return config, rows, hashes


def run(args):
    out = (ROOT/args.out).resolve()
    source = (ROOT/args.source_dataset).resolve()
    reference = (ROOT/args.reference).resolve()
    if out in (source, reference):
        raise ValueError('A separate output directory is required')
    if not args.smoke and (args.steps != 3000 or args.seeds != 4 or args.selector_steps != 3000):
        raise ValueError('Production requires four final seeds and 3000 updates per fit')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    reference_config, reference_rows, protected_hashes = load_reference(reference)
    previous, data, data_audit = prepare_data(source)
    protected_hashes[str(source/'data.npz')] = sha(source/'data.npz')
    candidates = [1.,2.] if args.smoke else CANDIDATES.copy()
    search_seeds = [200] if args.smoke else SEARCH_SEEDS.copy()
    arguments = vars(args).copy()
    arguments.pop('resume')
    manifest = base.source_manifest()
    manifest['start_ordinary_shared_gamma_domain16.ps1'] = sha(ROOT/'start_ordinary_shared_gamma_domain16.ps1')
    config = {
        'arguments':arguments, 'created_utc':base.utc(), 'source_sha256':manifest,
        'reference_run':str(reference), 'source_dataset_run':str(source),
        'protected_file_sha256':protected_hashes, 'data_fingerprint':fingerprint(data),
        'data_protocol':data_audit['protocol'], 'data_seed':71,
        'seeds':args.seeds, 'steps':args.steps, 'methods':['ssv_gf'],
        'treatment':'ordinary_two_gaussian_shared_dictionary_gf', 'smoke_only':args.smoke,
        'domain':[-16.,16.], 'nx':192, 'fourier_frequencies':16,
        'prediction_horizon':10., 'std_ddof':0,
        'model_class':'ic_gamma_domain16_model.Domain16GammaModel',
        'backbone_parameters':91329, 'energy_parameters':4417,
        'selector_steps':args.selector_steps, 'selector_seeds':search_seeds,
        'candidates':candidates, 'gamma_selection_weights':[1.,2.,3.],
        'selector':reference_config['selector'], 'training':reference_config['training'],
        'gamma_reference_distribution_imposed':False, 'local_gamma_refinement':False,
        'old_parameters_and_splits_unchanged':True, 'shapes_filtered_or_rescaled':False,
        'test_time_information':'Initial field plus one shared gamma selected on training/validation ICs only.',
        'comparison':'Same algorithm and paired optimizer initialization as the controlled-data run, on a different IC distribution. Descriptive cross-dataset comparison, not same-trajectory paired errors.',
        'primary_metric':'Original paper full-domain current-time population RelRMSE',
        'gf_horizon':10., 'gf_time_points':32, 'gf_spatial_extent':16.,
        'torch':str(torch.__version__),
        'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
    }
    if (out/'config.json').exists():
        if not args.resume:
            raise FileExistsError('Run exists; use explicit resume after confirming worker ended')
        saved = base.read(out/'config.json')
        for key in ('arguments','source_sha256','protected_file_sha256','data_fingerprint'):
            if saved[key] != config[key]:
                raise RuntimeError(f'Resume changed frozen {key}')
        if fingerprint(load_data(out/'data.npz')) != fingerprint(data):
            raise RuntimeError('Saved ordinary data changed')
    else:
        snapshot = out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in manifest:
            shutil.copy2(ORIGINAL_SOURCE if name=='upstream_train.py' else ROOT/name, snapshot/name)
        np.savez_compressed(out/'data.npz', **data)
        save_json(out/'config.json', config)
    np.testing.assert_array_equal(data['parameters'], previous['parameters'])
    np.testing.assert_array_equal(data['split'], previous['split'])
    save_json(out/'data_audit.json', data_audit)
    base.progress(out,'ordinary_shared_gamma_calibration',candidates=candidates,search_seeds=search_seeds)
    build_report(out)
    ids = {s:np.flatnonzero(data['split']==s) for s in ('train','valid','test')}
    observed = {s:np.ascontiguousarray(data['u'][idx][:,list(SIX)]) for s,idx in ids.items() if s!='test'}
    selection_path = out/'shared_gamma_selection.json'
    if selection_path.exists():
        selection = base.read(selection_path)
        assert selection['data_fingerprint'] == fingerprint(data)
    else:
        selection = calibrate_shared(observed['train'],observed['valid'],data['x'],
            data['t'][list(SIX)],out/'calibration',device=str(device),steps=args.selector_steps,
            seeds=tuple(search_seeds),candidates=tuple(candidates),resume=True,
            progress_callback=lambda value:base.progress(out,'ordinary_shared_gamma_calibration',selector=value))
        selection.update(data_fingerprint=fingerprint(data),test_used=False,
            future_labels_after_1_25_used=False, IC_parameters_used=False)
        save_json(selection_path,selection)
    gamma = float(selection['selected_gamma'])
    assert gamma in candidates
    save_json(out/'gamma_selection.json', {'selected_gamma':[gamma]*len(data['u']),
        'shared_gamma':gamma,'source':'shared_gamma_selection.json',
        'data_fingerprint':fingerprint(data),'frozen':True})
    print('ORDINARY SHARED GAMMA SELECTED',gamma,flush=True)
    build_report(out)
    x = torch.tensor(data['x'],dtype=torch.float32,device=device)
    t = torch.tensor(data['t'],dtype=torch.float32,device=device)
    inputs = {s:pack_inputs(torch.tensor(data['u'][idx,0],device=device),
        np.full(len(idx),gamma)) for s,idx in ids.items()}
    labels = {s:torch.tensor(observed[s],device=device) for s in ('train','valid')}
    rows = base.read(out/'results.json',[])
    for seed in range(args.seeds):
        if any(r['seed']==seed for r in rows):
            continue
        model,energy,row,checkpoint = fit_final(args,'ssv_gf',seed,
            {s:inputs[s] for s in ('train','valid')},labels,x,t,out)
        reference_row = next(r for r in reference_rows if r['seed']==seed)
        check_pair(row,reference_row,args.smoke)
        row.update(treatment='ordinary_two_gaussian_shared_dictionary_gf',shared_gamma=gamma,
            gamma_selection_file='shared_gamma_selection.json',
            same_algorithm_and_initialization_as_controlled=True,
            reference_checkpoint=reference_row['checkpoint'])
        checkpoint.update(treatment=row['treatment'],shared_gamma=gamma,
            gamma_source='shared_gamma_selection.json; constant frozen gamma for every IC')
        row = evaluate_and_save(model,energy,row,checkpoint,data,ids['test'],inputs['test'],x,t,out)
        with np.load(out/row['prediction']) as saved:
            curve = original.full_rel_curve(torch.from_numpy(saved['truth']).double(),
                torch.from_numpy(saved['pred']).double())
        np.testing.assert_allclose(curve.numpy(),row['metrics']['curve'],rtol=2e-7,atol=1e-9)
        row['metric_matches_original_full_rel_curve'] = True
        rows.append(row)
        save_json(out/'results.json',rows)
        build_report(out)
        print('ORDINARY FINAL DONE',seed,'gamma',gamma,'long',row['metrics']['long'],
            'final',row['metrics']['final'],flush=True)
        del model,energy,checkpoint
        if device.type=='cuda':
            torch.cuda.empty_cache()
    assert len(rows)==args.seeds and {r['seed'] for r in rows}==set(range(args.seeds))
    for path,before in protected_hashes.items():
        if sha(path) != before:
            raise RuntimeError(f'Protected reference or source artifact changed: {path}')
    save_json(out/'audit.json', {'complete_runs':len(rows),'expected_runs':args.seeds,
        'test_used_for_gamma_selection':False,'labels_after_1_25_used':False,
        'IC_parameters_enter_training':False,'old_parameters_and_splits_preserved':True,
        'IC_filtering_or_reference_clock_rescaling':False,'same_algorithm_source_as_controlled':True,
        'same_final_initialization_as_controlled':True,'checkpoint_replay_passed':True,
        'original_metric_verified':True,'gamma_frozen':True,'selected_gamma':gamma,
        'fresh_final_weights':True,'reference_artifacts_unchanged':True,
        'smoke_only':args.smoke,'completed_utc':base.utc()})
    save_json(out/'progress.json', {'status':'complete','phase':'complete','pid':os.getpid(),
        'completed_runs':len(rows),'completed_utc':base.utc(),'shared_gamma':gamma})
    build_report(out)
    (out/'stop_awake').touch()
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dataset',default='runs/gamma_pipeline_suite_cuda/existing71')
    parser.add_argument('--reference',default='runs/shared_gamma_domain16_t10_cuda')
    parser.add_argument('--out',default='runs/ordinary_shared_gamma_domain16_t10_cuda')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--selector-steps',type=int,default=3000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.seeds,args.selector_steps)<1 or args.seeds>4:
        parser.error('Positive counts required and at most four paired optimizer seeds')
    try:
        run(args)
    except BaseException as error:
        out = (ROOT/args.out).resolve()
        out.mkdir(parents=True,exist_ok=True)
        save_json(out/'progress.json', {'status':'failed','phase':'failed','pid':os.getpid(),
            'error':repr(error),'traceback':traceback.format_exc(),'failed_utc':base.utc()})
        (out/'stop_awake').touch()
        raise
