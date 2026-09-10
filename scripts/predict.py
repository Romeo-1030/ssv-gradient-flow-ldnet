#!/usr/bin/env python3
"""Run a released SSV-GF-LDNet checkpoint without retraining."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from train import EnergyDensity, ShiftedSSVLDNet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "artifacts/pretrained/learned_ssv_energy_n32_seed0.pt"

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,default=DEFAULT_CKPT)
    p.add_argument("--input",type=Path,help="Optional .npy u0 or .npz containing u0 (or u[0]) and x.")
    p.add_argument("--out_dir",type=Path,default=ROOT/"outputs/inference_demo")
    p.add_argument("--device",default="auto",help="auto, cpu, cuda, or cuda:0")
    p.add_argument("--tmax",type=float,default=10.0); p.add_argument("--nt",type=int,default=201)
    return p.parse_args()

def load_input(path):
    x=np.linspace(-16.,16.,256,dtype=np.float32)
    if path is None:
        amp,center,width=1.,.25,.85; u0=amp*np.exp(-.5*((x-center)/width)**2)
        return x,u0.astype(np.float32),(amp,center,width)
    obj=np.load(path)
    if isinstance(obj,np.ndarray): u0=obj
    else:
        x=obj["x"].astype(np.float32) if "x" in obj else x
        if "u0" in obj: u0=obj["u0"]
        elif "u" in obj: u0=obj["u"][0] if obj["u"].ndim>1 else obj["u"]
        else: raise ValueError("The .npz file must contain 'u0' or 'u'.")
    u0=np.asarray(u0,dtype=np.float32).squeeze()
    if x.shape!=(256,) or u0.shape!=(256,): raise ValueError("This checkpoint expects x and u0 with shape (256,).")
    return x,u0,None

def main():
    a=parse_args(); device=("cuda:0" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device
    x_np,u0_np,exact_params=load_input(a.input); x=torch.tensor(x_np,device=device); u0=torch.tensor(u0_np[None],device=device)
    times=torch.linspace(0.,a.tmax,a.nt,device=device)
    model=ShiftedSSVLDNet(256,"learned_ssv_energy",24,16,128,33,4.5*np.sqrt(11.),4.5).to(device); energy=EnergyDensity().to(device)
    try: state=torch.load(a.checkpoint,map_location=device,weights_only=True)
    except TypeError: state=torch.load(a.checkpoint,map_location=device)
    model.load_state_dict(state["model"]); energy.load_state_dict(state["energy"]); model.eval()
    with torch.no_grad(): pred,_,_=model.rollout(u0,times,x[None]); alpha,beta=model.scales()
    pred_np=pred[0].cpu().numpy(); t_np=times.cpu().numpy(); metrics={"device":device,"alpha":float(alpha),"beta":float(beta)}; truth=None
    if exact_params is not None:
        amp,center,width=exact_params; var=width**2+2.*t_np[:,None]; truth=amp*width/np.sqrt(var)*np.exp(-.5*(x_np[None]-center)**2/var)
        mask=np.abs(x_np[None])<=4.5*np.sqrt(1.+t_np[:,None]); err2=((pred_np-truth)**2*mask).sum(1)
        rel=np.sqrt(err2/np.maximum((truth**2*mask).sum(1),1e-12)); fixed=np.sqrt(err2/mask.sum(1))/np.sqrt(np.mean(u0_np**2))
        metrics.update({"observed_window_mean_relrmse":float(rel[t_np<=.5+1e-7].mean()),"extrapolation_mean_fixed_scale_error":float(fixed[t_np>.5+1e-7].mean()),"final_relrmse":float(rel[-1])})
    a.out_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out_dir/"prediction.npz",x=x_np,t=t_np,u0=u0_np,prediction=pred_np); (a.out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    ids=np.unique(np.round(np.linspace(0,a.nt-1,5)).astype(int)); fig,axes=plt.subplots(1,len(ids),figsize=(3.1*len(ids),2.8),sharey=True); axes=np.atleast_1d(axes)
    for ax,j in zip(axes,ids):
        if truth is not None: ax.plot(x_np,truth[j],"k--",lw=1.5,label="exact")
        ax.plot(x_np,pred_np[j],color="#b54a2a",lw=1.8,label="prediction"); ax.set_title(f"t={t_np[j]:.2f}"); ax.set_xlabel("x")
    axes[0].set_ylabel("u"); axes[-1].legend(frameon=False); fig.tight_layout(); fig.savefig(a.out_dir/"rollout.png",dpi=180); plt.close(fig)
    print(json.dumps(metrics,indent=2)); print(f"Saved inference outputs to {a.out_dir}")
if __name__=="__main__": main()
