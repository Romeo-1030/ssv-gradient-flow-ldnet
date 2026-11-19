"""One shared-gamma dictionary+GF arm, paired with the saved per-IC arm.

Reuse exact domain16 data and final fit implementation. The historical
shared selector uses temporary LDNet models on train/validation ICs only;
its weights are discarded. This is a complete-pipeline comparison, not an
isolated comparison of gamma granularity or equal calibration budgets.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
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
from shared_gamma_domain16_selector import calibrate_shared
from shared_gamma_domain16_report import build_report

ROOT = Path(__file__).resolve().parent
CORE_SOURCES = ('ic_gamma_domain16_fit.py', 'ic_gamma_domain16_model.py',
    'ic_gamma_model.py', 'original_scale_model.py', 'original_scale.py',
    'experiment.py', 'moment_baselines.py', 'forecast_gamma_model.py',
    'ic_gamma_experiment.py', 'moving_charts.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    with np.load(path) as source:
        return {k: source[k] for k in source.files}


def load_parent(parent):
    parent = Path(parent).resolve()
    config = base.read(parent/'config.json')
    rows = [r for r in base.read(parent/'results.json') if r['method'] == 'ssv_gf']
    if config['domain'] != [-16., 16.] or config['prediction_horizon'] != 10.:
        raise ValueError('Parent must be the completed domain16/t10 experiment')
    if config['steps'] != 3000 or config['seeds'] != 4 or {r['seed'] for r in rows} != set(range(4)):
        raise ValueError('Expected the completed four-seed per-IC SSV+GF arm')
    for name in CORE_SOURCES:
        if sha(ROOT/name) != config['source_sha256'][name]:
            raise RuntimeError(f'Final-training source changed since parent run: {name}')
    if sha(ORIGINAL_SOURCE) != config['source_sha256']['upstream_train.py']:
        raise RuntimeError('Upstream source differs from the parent run')
    data = load_data(parent/'data.npz')
    assert data['u'].shape == (72, 81, 192)
    np.testing.assert_array_equal(data['t'], np.arange(81)/8)
    np.testing.assert_array_equal(data['x'], np.linspace(-16., 16., 192, endpoint=False))
    np.testing.assert_array_equal(data['t'][list(SIX)], [0., .25, .5, .75, 1., 1.25])
    assert {s:int(np.sum(data['split'] == s)) for s in ('train','valid','test')} == {'train':36,'valid':12,'test':24}
    protected = ['config.json', 'data.npz', 'results.json', 'gamma_selection.json']
    protected += [r[key] for r in rows for key in ('checkpoint', 'prediction')]
    hashes = {name:sha(parent/name) for name in protected}
    return data, config, rows, hashes


def check_pair(row, parent_row, smoke=False):
    keys = ('initial_parameters_sha256', 'initial_energy_sha256',
        'backbone_parameters', 'energy_parameters', 'scales', 'gf_times',
        'gf_collocation', 'gf_spatial_extent', 'decoder_extent',
        'fourier_frequencies', 'label_times', 'gamma_frozen')
    for key in keys:
        if row[key] != parent_row[key]:
            raise AssertionError(f'Paired final protocol differs: {key}')
    if not smoke:
        assert row['steps'] == parent_row['steps'] == 3000
        assert row['gf_max_weight'] == parent_row['gf_max_weight']


def run(args):
    out, parent = (ROOT/args.out).resolve(), (ROOT/args.parent).resolve()
    if out == parent:
        raise ValueError('A separate output folder is required')
    if not args.smoke and (args.steps != 3000 or args.seeds != 4 or args.selector_steps != 3000):
        raise ValueError('Production protocol is fixed: four final seeds, 3000 steps per fit')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    data, parent_config, parent_rows, parent_hashes = load_parent(parent)
    candidates = [1., 2.] if args.smoke else [.125, .25, .5, 1., 2., 4., 8.]
    search_seeds = [200] if args.smoke else [200, 201, 202]
    arguments = vars(args).copy()
    arguments.pop('resume')
    manifest = base.source_manifest()
    manifest['start_shared_gamma_domain16.ps1'] = sha(ROOT/'start_shared_gamma_domain16.ps1')
    config = {
        'arguments': arguments, 'created_utc': base.utc(), 'parent': str(parent),
        'source_sha256': manifest, 'parent_file_sha256': parent_hashes,
        'data_fingerprint': fingerprint(data), 'data_protocol': parent_config['data_protocol'],
        'seeds': args.seeds, 'steps': args.steps, 'methods': ['ssv_gf'],
        'treatment': 'shared_gamma_dictionary_gf', 'smoke_only': args.smoke,
        'domain': [-16.,16.], 'nx':192, 'fourier_frequencies':16,
        'prediction_horizon':10., 'std_ddof':0,
        'model_class':'ic_gamma_domain16_model.Domain16GammaModel',
        'backbone_parameters':91329, 'energy_parameters':4417,
        'selector_steps':args.selector_steps, 'selector_seeds':search_seeds,
        'candidates':candidates, 'gamma_selection_weights':[1.,2.,3.],
        'selector':'Temporary LDNet without GF: train early three frames, early-validation checkpoint, later three validation frames weighted 1:2:3; average three seeds; no local gamma refinement.',
        'training':'Reuse unchanged parent fit_final; fresh shared LDNet and energy initialization matched by seed; six observed labels; selected shared gamma frozen.',
        'comparison':'Complete selector pipelines; final data, model, optimizer, GF and paired seeds fixed. Selector architecture, calibration budget and test-time information differ.',
        'test_time_information':{'shared_gamma_gf':'Initial field only; gamma selected on training/validation ICs',
            'parent_per_ic_gamma_gf':'Each test IC supplies all six early fields to its ordinary gamma selector'},
        'gf_horizon':10., 'gf_time_points':32, 'gf_spatial_extent':16.,
        'primary_metric':'Original paper full-domain current-time population RelRMSE',
        'torch':str(torch.__version__),
        'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
    }
    if (out/'config.json').exists():
        if not args.resume:
            raise FileExistsError('Run exists; resume explicitly only after worker ended')
        saved = base.read(out/'config.json')
        for key in ('arguments','source_sha256','parent_file_sha256'):
            if saved[key] != config[key]:
                raise RuntimeError(f'Resume changed frozen {key}')
        if sha(out/'data.npz') != parent_hashes['data.npz']:
            raise RuntimeError('Saved data differs from parent')
    else:
        snapshot = out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in manifest:
            shutil.copy2(ORIGINAL_SOURCE if name=='upstream_train.py' else ROOT/name, snapshot/name)
        shutil.copy2(parent/'data.npz', out/'data.npz')
        save_json(out/'config.json', config)
    assert sha(out/'data.npz') == parent_hashes['data.npz']
    save_json(out/'data_audit.json', {'copied_byte_identical':True,
        'parent_data_sha256':parent_hashes['data.npz'], 'source_IC_parameters_and_splits_unchanged':True,
        'counts':{'train':36,'valid':12,'test':24}, 'labels':[0.,.25,.5,.75,1.,1.25],
        'domain':[-16.,16.], 'horizon':10.})
    base.progress(out, 'shared_gamma_calibration', candidates=candidates, search_seeds=search_seeds)
    build_report(out)
    indices = {s:np.flatnonzero(data['split'] == s) for s in ('train','valid','test')}
    observed = {s:np.ascontiguousarray(data['u'][ids][:,list(SIX)]) for s,ids in indices.items() if s!='test'}
    selection_path = out/'shared_gamma_selection.json'
    if selection_path.exists():
        selection = base.read(selection_path)
        assert selection['data_fingerprint'] == fingerprint(data)
    else:
        # The selector receives no test ICs, latent labels after1.25, or hidden IC parameters.
        selection = calibrate_shared(observed['train'], observed['valid'], data['x'],
            data['t'][list(SIX)], out/'calibration', device=str(device), steps=args.selector_steps,
            seeds=tuple(search_seeds), candidates=tuple(candidates), resume=True,
            progress_callback=lambda value:base.progress(out,'shared_gamma_calibration',selector=value))
        selection['data_fingerprint'] = fingerprint(data)
        selection['data_sha256'] = parent_hashes['data.npz']
        selection['test_used'] = False
        selection['future_labels_after_1_25_used'] = False
        save_json(selection_path, selection)
    selected_gamma = float(selection['selected_gamma'])
    assert selected_gamma in candidates
    # A full per-IC vector is persisted for the same prediction interface; every entry is equal.
    save_json(out/'gamma_selection.json', {'selected_gamma':[selected_gamma]*len(data['u']),
        'shared_gamma':selected_gamma, 'source':'shared_gamma_selection.json',
        'data_fingerprint':fingerprint(data), 'frozen':True})
    print('SHARED GAMMA SELECTED', selected_gamma, flush=True)
    build_report(out)
    x = torch.tensor(data['x'], dtype=torch.float32, device=device)
    t = torch.tensor(data['t'], dtype=torch.float32, device=device)
    inputs = {s:pack_inputs(torch.tensor(data['u'][ids,0], device=device),
        np.full(len(ids),selected_gamma)) for s,ids in indices.items()}
    labels = {s:torch.tensor(observed[s],device=device) for s in ('train','valid')}
    rows = base.read(out/'results.json', [])
    for seed in range(args.seeds):
        if any(r['seed']==seed for r in rows):
            continue
        model, energy, row, checkpoint = fit_final(args, 'ssv_gf', seed,
            {s:inputs[s] for s in ('train','valid')}, labels, x, t, out)
        paired = next(r for r in parent_rows if r['seed']==seed)
        check_pair(row, paired, args.smoke)
        row.update(treatment='shared_gamma_dictionary_gf', shared_gamma=selected_gamma,
            gamma_selection_file='shared_gamma_selection.json',
            paired_parent_checkpoint=paired['checkpoint'], final_protocol_matches_parent=True)
        checkpoint.update(treatment='shared_gamma_dictionary_gf', shared_gamma=selected_gamma,
            gamma_source='shared_gamma_selection.json; constant frozen gamma for all ICs')
        row = evaluate_and_save(model,energy,row,checkpoint,data,indices['test'],inputs['test'],x,t,out)
        with np.load(out/row['prediction']) as stored:
            curve = original.full_rel_curve(torch.from_numpy(stored['truth']).double(),
                torch.from_numpy(stored['pred']).double())
        np.testing.assert_allclose(curve.numpy(),row['metrics']['curve'],rtol=2e-7,atol=1e-9)
        row['metric_matches_original_full_rel_curve'] = True
        rows.append(row)
        save_json(out/'results.json',rows)
        build_report(out)
        print('SHARED FINAL DONE',seed,'gamma',selected_gamma,'long',row['metrics']['long'],
            'final',row['metrics']['final'],flush=True)
        del model,energy,checkpoint
        if device.type=='cuda':
            torch.cuda.empty_cache()
    assert len(rows)==args.seeds and {r['seed'] for r in rows} == set(range(args.seeds))
    for name,before in parent_hashes.items():
        if sha(parent/name) != before:
            raise RuntimeError(f'Protected parent artifact changed: {name}')
    save_json(out/'audit.json', {'complete_runs':len(rows),'expected_runs':args.seeds,
        'test_used_for_gamma_selection':False,'labels_after_1_25_used':False,
        'same_data_bytes':True,'same_final_training_source':True,'paired_initialization_passed':True,
        'checkpoint_replay_passed':True,'original_metric_verified':True,
        'gamma_frozen':True,'selected_gamma':selected_gamma,
        'fresh_final_weights':True,'parent_unchanged':True,'smoke_only':args.smoke,
        'completed_utc':base.utc()})
    save_json(out/'progress.json', {'status':'complete','phase':'complete','pid':os.getpid(),
        'completed_runs':len(rows),'completed_utc':base.utc(),'shared_gamma':selected_gamma})
    build_report(out)
    (out/'stop_awake').touch()
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',default='runs/ic_gamma_domain16_t10_cuda')
    parser.add_argument('--out',default='runs/shared_gamma_domain16_t10_cuda')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--selector-steps',type=int,default=3000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.seeds,args.selector_steps)<1 or args.seeds>4:
        parser.error('Positive counts required and at most four paired final seeds')
    try:
        run(args)
    except BaseException as error:
        out = (ROOT/args.out).resolve()
        out.mkdir(parents=True,exist_ok=True)
        save_json(out/'progress.json', {'status':'failed','phase':'failed','pid':os.getpid(),
            'error':repr(error),'traceback':traceback.format_exc(),'failed_utc':base.utc()})
        (out/'stop_awake').touch()
        raise
