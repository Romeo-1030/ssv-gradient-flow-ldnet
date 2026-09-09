#!/usr/bin/env python3
"""Shifted LDNet with learned initial-condition representation, SSV, and local energy."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, ni, no, width=128, depth=3):
        super().__init__(); dims=[ni]+[width]*(depth-1)+[no]; layers=[]
        for a,b in zip(dims[:-2],dims[1:-1]): layers += [nn.Linear(a,b),nn.SiLU()]
        layers += [nn.Linear(dims[-2],dims[-1])]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class InitialFieldEncoder(nn.Module):
    """Learn a representation from the observed initial field, not generator parameters."""
    def __init__(self,nx,rep_dim,width):
        super().__init__(); self.net=MLP(nx,rep_dim,width,3)
    def forward(self,u0): return self.net(u0)

class RBFDecoder(nn.Module):
    def __init__(self,latent,modes,extent,width):
        super().__init__(); self.coeff=MLP(latent,modes,width,3)
        c=torch.linspace(-extent,extent,modes); h=float(c[1]-c[0])
        self.register_buffer("centers",c); self.rbf_width=1.25*h
    def forward(self,z,q):
        phi=torch.exp(-.5*((q[...,None]-self.centers)/self.rbf_width)**2)
        return (self.coeff(z)[:,None,:]*phi).sum(-1)

class EnergyDensity(nn.Module):
    def __init__(self): super().__init__(); self.net=MLP(2,1,64,3)
    def forward(self,u,ux): return self.net(torch.stack((u,ux),-1)).squeeze(-1)

class ShiftedSSVLDNet(nn.Module):
    def __init__(self,nx,mode,rep_dim,latent,width,modes,physical_extent,ssv_extent):
        super().__init__(); self.mode=mode; self.latent=latent
        self.encoder=InitialFieldEncoder(nx,rep_dim,width)
        self.dynamics=MLP(latent+rep_dim,latent,width,3)
        extent=physical_extent if mode.startswith("physical") else ssv_extent
        self.decoder=RBFDecoder(latent,modes,extent,width)
        if mode.startswith("learned"):
            self.alpha_raw=nn.Parameter(torch.tensor(-.45)); self.beta_raw=nn.Parameter(torch.tensor(-.45))
    def scales(self):
        zero=self.decoder.centers.new_tensor(0.)
        if self.mode.startswith("physical"): return zero,zero
        if self.mode.startswith("hand_ssv"): return zero+.5,zero+.5
        return .05+.9*torch.sigmoid(self.alpha_raw),torch.sigmoid(self.beta_raw)
    def clock(self,t): return t if self.mode.startswith("physical") else torch.log1p(t)
    def field(self,z,p): return self.dynamics(torch.cat((z,p),-1))
    def advance(self,z,p,h):
        f=lambda q:self.field(q,p); k1=f(z); k2=f(z+.5*h*k1); k3=f(z+.5*h*k2); k4=f(z+h*k3)
        return z+h*(k1+2*k2+2*k3+k4)/6
    def latent_rollout(self,u0,times):
        p=self.encoder(u0); z=torch.zeros(len(u0),self.latent,device=u0.device,dtype=u0.dtype)
        z=self.advance(z,p,1.0); zs=[z]  # z_-1=0; learned first step creates z_0
        clock=self.clock(times)
        for j in range(1,len(times)): z=self.advance(z,p,clock[j]-clock[j-1]); zs.append(z)
        return torch.stack(zs,1),p
    def decode_physical(self,z,t,x):
        alpha,beta=self.scales(); s=1+t; return self.decoder(z,x/s.pow(alpha))/s.pow(beta)
    def rollout(self,u0,times,x):
        zs,p=self.latent_rollout(u0,times)
        pred=torch.stack([self.decode_physical(zs[:,j],times[j],x) for j in range(len(times))],1)
        return pred,zs,p

def gradient_rhs(model,energy,z,p,xi):
    xi=xi.detach().requires_grad_(True); dz=model.field(z,p)
    w,wt=torch.autograd.functional.jvp(lambda zz:model.decoder(zz,xi),z,dz,create_graph=True)
    wx=torch.autograd.grad(w,xi,torch.ones_like(w),create_graph=True)[0]
    e=energy(w,wx); eu,eux=torch.autograd.grad(e.sum(),(w,wx),create_graph=True,retain_graph=True)
    dx_eux=torch.autograd.grad(eux,xi,torch.ones_like(eux),create_graph=True,retain_graph=True)[0]
    alpha,beta=model.scales(); return wt-alpha*xi*wx-beta*w,-eu+dx_eux

def rel_curve(y,yhat,mask):
    return torch.sqrt((((yhat-y)**2*mask).sum((0,2)))/(y.square().mul(mask).sum((0,2)).clamp_min(1e-12)))

def args():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out_dir',required=True)
    p.add_argument('--mode',choices=('physical','physical_energy','hand_ssv','hand_ssv_energy','learned_ssv','learned_ssv_energy'),required=True)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=1000); p.add_argument('--num_train',type=int,default=32)
    p.add_argument('--train_pool',type=int,default=96); p.add_argument('--num_valid',type=int,default=32); p.add_argument('--num_test',type=int,default=32); p.add_argument('--train_tmax',type=float,default=.5)
    p.add_argument('--rep_dim',type=int,default=24); p.add_argument('--latent',type=int,default=16); p.add_argument('--width',type=int,default=128)
    p.add_argument('--decoder_modes',type=int,default=65); p.add_argument('--window',type=float,default=4.5); p.add_argument('--collocation',type=int,default=24)
    p.add_argument('--lambda_gf',type=float,default=1e-3); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--device',default='cuda:0')
    return p.parse_args()

def main():
    a=args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    d=np.load(a.data); u=torch.tensor(d['u'],device=a.device); x=torch.tensor(d['x'],device=a.device); t=torch.tensor(d['t'],device=a.device)
    perm=np.random.default_rng(123).permutation(len(u)); train_pool=perm[:a.train_pool]; it=train_pool[:a.num_train]
    iv=perm[a.train_pool:a.train_pool+a.num_valid]; ie=perm[a.train_pool+a.num_valid:a.train_pool+a.num_valid+a.num_test]
    model=ShiftedSSVLDNet(u.shape[-1],a.mode,a.rep_dim,a.latent,a.width,a.decoder_modes,a.window*np.sqrt(float(t[-1])+1),a.window).to(a.device)
    energy=EnergyDensity().to(a.device); use_energy=a.mode.endswith('energy')
    opt=torch.optim.AdamW(list(model.parameters())+(list(energy.parameters()) if use_energy else []),lr=a.lr,weight_decay=1e-6)
    ids=torch.where(t<=a.train_tmax+1e-7)[0]; obs=torch.tensor([0,len(ids)//2,len(ids)-1],device=a.device)
    s=1+t; fair=(torch.abs(x[None])<=a.window*torch.sqrt(s[:,None])).float()[None]; xb=x[None].expand(a.num_train,-1)
    best=1e99; state=None; hist=[]; gf_scale=None
    for ep in range(1,a.epochs+1):
        model.train(); pred,z,p=model.rollout(u[it,0],t[ids],xb); truth=u[it][:,ids]; m=fair[:,ids]
        sq=((pred[:,obs]-truth[:,obs])**2*m[:,obs]).sum(-1); den=(truth[:,obs].square()*m[:,obs]).sum(-1).clamp_min(1e-6)
        data_loss=(sq/den).mean(); raw=pred.new_tensor(0.)
        if use_energy:
            zz=z.reshape(-1,z.shape[-1]); pp=p[:,None].expand(-1,len(ids),-1).reshape(-1,p.shape[-1]); xi=(2*torch.rand(len(zz),a.collocation,device=a.device)-1)*a.window
            intrinsic,rhs=gradient_rhs(model,energy,zz,pp,xi); raw=(intrinsic-rhs).square().mean()
            if ep==101: gf_scale=max(float(raw.detach()),1e-12)
        lam=0. if (not use_energy or ep<=100) else a.lambda_gf*min((ep-100)/100,1.)
        loss=data_loss+(0 if lam==0 else lam*raw/gf_scale); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        if ep==1 or ep%10==0:
            model.eval()
            with torch.no_grad(): pv,_,_=model.rollout(u[iv,0],t[ids],x[None].expand(len(iv),-1)); val=float(rel_curve(u[iv][:,ids],pv,fair[:,ids]).mean())
            rec={'epoch':ep,'valid':val,'relative_data_loss':float(data_loss.detach()),'raw_gf':float(raw.detach()),'alpha':float(model.scales()[0].detach()),'beta':float(model.scales()[1].detach())}; hist.append(rec)
            if val<best: best=val; state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'energy':{k:v.detach().cpu().clone() for k,v in energy.state_dict().items()}}
            print(json.dumps(rec),flush=True)
    model.load_state_dict(state['model']); energy.load_state_dict(state['energy']); model.eval()
    with torch.no_grad(): pe,_,_=model.rollout(u[ie,0],t,x[None].expand(len(ie),-1))
    curve=rel_curve(u[ie],pe,fair).cpu().numpy(); abs_curve=torch.sqrt((((pe-u[ie])**2*fair).sum((0,2)))/(fair.sum((0,2))*len(ie))).cpu().numpy(); ext=t.cpu().numpy()>a.train_tmax+1e-7; alpha,beta=model.scales()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); np.save(out/'relrmse_vs_time.npy',curve); np.save(out/'rmse_vs_time.npy',abs_curve)
    summary={'mode':a.mode,'seed':a.seed,'num_train':a.num_train,'supervised_frames':3,'test_seen_time_relrmse':float(curve[~ext].mean()),'extrapolation_relrmse':float(curve[ext].mean()),'final_relrmse':float(curve[-1]),'extrapolation_rmse':float(abs_curve[ext].mean()),'final_rmse':float(abs_curve[-1]),'alpha':float(alpha.detach()),'beta':float(beta.detach()),'best_valid':best,'args':vars(a)}
    torch.save(state,out/'model.pt'); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n'); (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print('FINAL',json.dumps(summary),flush=True)
if __name__=='__main__': main()
