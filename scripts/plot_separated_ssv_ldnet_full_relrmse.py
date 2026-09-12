#!/usr/bin/env python3
"""Re-evaluate latest frozen-SSV models using standard full-domain RelRMSE."""
from pathlib import Path
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from train_fourier_longtime_gf import ShiftedSSVLDNet  # noqa: E402
D=np.load(ROOT/"data/single_gaussian_heat_fixed_domain_v1.npz")
x=torch.tensor(D["x"]); t=torch.tensor(D["t"]); params=torch.tensor(D["params"])
truth=torch.tensor(D["u"]); test=np.random.default_rng(123).permutation(len(truth))[128:160]
NEW=ROOT/"outputs/ssv_gamma_frozen_ldnet_v5_final3000_cosine"
specs={
 "LDNet":(NEW/"physical/n32","physical",False,False,"#666666","-"),
 "LDNet + learned SSV":(NEW/"learned_ssv/n32","learned_ssv",True,False,"#3478b8","--"),
 "LDNet + learned SSV + energy":(NEW/"learned_ssv_energy/n32","learned_ssv_energy",True,False,"#174a7e","-"),
 "LDNet + oracle SSV":(NEW/"oracle_ssv/n32","hand_ssv",False,True,"#e28e2c","--"),
 "LDNet + oracle SSV + energy":(NEW/"oracle_ssv_energy/n32","hand_ssv_energy",False,True,"#a65300","-"),
}
curves={}; results={}; ext=D["t"]>.5+1e-7; obs=~ext
for label,(folder,mode,use_gamma,oracle_gamma,color,ls) in specs.items():
    seed_curves=[]
    for seed in range(5):
        model=ShiftedSSVLDNet(len(x),mode,24,16,128,33,float(x.abs().max()),4.5,
                              use_gamma=use_gamma,oracle_gamma=oracle_gamma)
        state=torch.load(folder/f"seed{seed}"/"model.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"]); model.eval()
        with torch.no_grad(): pred,_,_=model.rollout(params[test],t,x[None].expand(len(test),-1))
        err=(pred-truth[test]).square().sum((0,2)); den=truth[test].square().sum((0,2)).clamp_min(1e-12)
        seed_curves.append((100*torch.sqrt(err/den)).numpy())
    a=np.stack(seed_curves); curves[label]=a
    results[label]={
      "observed_mean_percent":float(a[:,obs].mean(1).mean()),"observed_std_percent":float(a[:,obs].mean(1).std()),
      "extrapolation_mean_percent":float(a[:,ext].mean(1).mean()),"extrapolation_std_percent":float(a[:,ext].mean(1).std()),
      "final_mean_percent":float(a[:,-1].mean()),"final_std_percent":float(a[:,-1].std())}

fig,axes=plt.subplots(1,2,figsize=(10.8,3.25),dpi=220)
for label,(_,_,_,_,color,ls) in specs.items():
    a=curves[label]; mean,std=a.mean(0),a.std(0)
    for ax,mask in zip(axes,(obs,D["t"]>=.5-1e-8)):
        ax.plot(D["t"][mask],mean[mask],color=color,ls=ls,lw=2.1,label=label)
        ax.fill_between(D["t"][mask],np.maximum(mean[mask]-std[mask],1e-4),mean[mask]+std[mask],color=color,alpha=.11)
axes[0].set(title="Observed window",xlabel="time",ylabel="Full-domain RelRMSE (%)",yscale="log",xlim=(0,.5))
axes[1].set(title="Long-time extrapolation",xlabel="time",ylabel="Full-domain RelRMSE (%)",yscale="log",xlim=(.5,10))
for ax in axes: ax.grid(alpha=.22,which="both")
axes[1].legend(frameon=False,fontsize=7.5,loc="upper left")
fig.tight_layout()
out=ROOT/"figures/fixed_domain_two_panel_log"; fig.savefig(out.with_suffix(".png"),bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight")
out.with_suffix(".json").write_text(json.dumps(results,indent=2)+"\n"); print(json.dumps(results,indent=2))
