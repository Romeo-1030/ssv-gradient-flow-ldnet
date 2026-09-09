#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path("outputs")
modes=['physical','physical_energy','learned_ssv','learned_ssv_energy','hand_ssv','hand_ssv_energy']; ns=[2,4,8,16,32]
labels={'physical':'Physical','physical_energy':'Physical+E','learned_ssv':'Learned SSV','learned_ssv_energy':'Learned SSV+E','hand_ssv':'Hand SSV','hand_ssv_energy':'Hand SSV+E'}
summary={}; fig,ax=plt.subplots(1,2,figsize=(10,3.8),dpi=200)
for mode in modes:
    means=[]; stds=[]; absmeans=[]
    for n in ns:
        vals=[json.loads(p.read_text()) for p in sorted((root/mode/f'n{n}').glob('seed*/summary.json'))]
        if not vals: means.append(np.nan); stds.append(np.nan); absmeans.append(np.nan); continue
        r=np.array([v['extrapolation_relrmse'] for v in vals]); q=np.array([v['extrapolation_rmse'] for v in vals])
        means.append(r.mean()); stds.append(r.std()); absmeans.append(q.mean())
        summary[f'{mode}/n{n}']={'count':len(vals),'relrmse_mean':float(r.mean()),'relrmse_std':float(r.std()),'rmse_mean':float(q.mean()),'alpha_mean':float(np.mean([v['alpha'] for v in vals])),'beta_mean':float(np.mean([v['beta'] for v in vals]))}
    ax[0].errorbar(ns,means,yerr=stds,marker='o',label=labels[mode]); ax[1].plot(ns,absmeans,marker='o',label=labels[mode])
for a,y in zip(ax,['extrapolation RelRMSE','extrapolation absolute RMSE']): a.set(xlabel='number of training trajectories',ylabel=y,xscale='log',yscale='log'); a.grid(alpha=.2); a.legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig(root/'data_scaling.png',bbox_inches='tight'); (root/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
