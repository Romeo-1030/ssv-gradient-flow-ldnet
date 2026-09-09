#!/usr/bin/env python3
"""Analytic large-domain heat data from a low-dimensional single-Gaussian IC family."""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--samples',type=int,default=160)
p.add_argument('--nx',type=int,default=256); p.add_argument('--nt',type=int,default=201); p.add_argument('--xmax',type=float,default=16.)
p.add_argument('--tmax',type=float,default=10.); p.add_argument('--seed',type=int,default=71); a=p.parse_args()
rng=np.random.default_rng(a.seed); x=np.linspace(-a.xmax,a.xmax,a.nx); t=np.linspace(0,a.tmax,a.nt); nu=1.
amp=rng.uniform(.8,1.2,a.samples); center=rng.uniform(-.6,.6,a.samples); width=rng.uniform(.7,1.05,a.samples)
var=width[:,None,None]**2+2*nu*t[None,:,None]
u=amp[:,None,None]*width[:,None,None]/np.sqrt(var)*np.exp(-.5*(x[None,None]-center[:,None,None])**2/var)
params=np.stack((amp,center,width),-1).astype(np.float32); a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out,u=u.astype(np.float32),params=params,x=x.astype(np.float32),t=t.astype(np.float32),diffusivity=np.float32(nu))
meta={'shape':list(u.shape),'family':'single Gaussian','parameter_ranges':{'amplitude':[.8,1.2],'center':[-.6,.6],'width':[.7,1.05]},'train_tmax':.5,'test_tmax':a.tmax,'known_alpha':.5,'known_beta':.5,'note':'models receive u0, never params/PDE/RHS/analytic energy'}
a.out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
