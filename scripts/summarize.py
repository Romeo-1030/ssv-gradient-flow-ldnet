"""Aggregate exactly five matched seeds for five final methods."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('outputs/ssv_gamma_frozen_ldnet_v5_final3000_cosine'))
p.add_argument('--out',type=Path,default=Path('results/summary/aggregate.json'))
a=p.parse_args()
result={'protocol':'final3000','units':'fraction','std_ddof':0,'methods':{}}
times=None
for method in ['physical','learned_ssv','learned_ssv_energy','oracle_ssv','oracle_ssv_energy']:
    runs=[]
    for seed in range(5):
        folder=a.root/method/'n32'/f'seed{seed}'
        meta=json.loads((folder/'summary.json').read_text())
        assert meta['training_steps']==3000 and meta['seed']==seed
        t=np.load(folder/'time.npy'); c=np.load(folder/'full_relrmse_vs_time.npy')
        assert c.shape==t.shape and np.isfinite(c).all()
        if times is None: times=t
        np.testing.assert_array_equal(times,t)
        obs=t<=.5+1e-7
        metrics={'observed':float(c[obs].mean()),'extrapolation':float(c[~obs].mean()),'final':float(c[-1])}
        for k,v in metrics.items(): np.testing.assert_allclose(v,meta[k+'_full_relrmse'],rtol=1e-6)
        runs.append({'seed':seed,'selected_epoch':meta['selected_checkpoint_epoch'],'alpha':meta['alpha'],'beta':meta['beta'],'metrics':metrics,'curve':c.tolist()})
    agg={k:{'mean':float(np.mean([s['metrics'][k] for s in runs])),'std':float(np.std([s['metrics'][k] for s in runs]))} for k in metrics}
    result['methods'][method]={'runs':runs,'aggregate':agg}
    print(method,agg)
result['time']=times.tolist()
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,separators=(',',':'))+'\n')
