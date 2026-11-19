#!/usr/bin/env python3
"""Parameter-conditioned Euler LDNet with continuous Fourier decoder, SSV, and local energy."""
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

class FourierDecoder(nn.Module):
    """Continuous real Fourier series with a fixed total coefficient budget."""
    def __init__(self,latent,modes,extent,width):
        super().__init__()
        if modes < 3 or modes % 2 != 1:
            raise ValueError("decoder_modes must be odd: 1 + 2*K coefficients")
        self.num_frequencies=(modes-1)//2
        self.extent=float(extent)
        self.coeff=MLP(latent+3,modes,width,3)
        self.register_buffer("frequencies",torch.arange(1,self.num_frequencies+1,dtype=torch.float32))
    def forward(self,z,q,p):
        phase=torch.pi*q[...,None]*self.frequencies/self.extent
        basis=torch.cat((torch.ones_like(q[...,None]),torch.cos(phase),torch.sin(phase)),-1)
        return (self.coeff(torch.cat((z,p),-1))[:,None,:]*basis).sum(-1)

class EnergyDensity(nn.Module):
    def __init__(self): super().__init__(); self.net=MLP(2,1,64,3)
    def forward(self,u,ux): return self.net(torch.stack((u,ux),-1)).squeeze(-1)

class ShiftedSSVLDNet(nn.Module):
    def __init__(self,nx,mode,rep_dim,latent,width,modes,physical_extent,ssv_extent,alpha_init=None,beta_init=None,use_gamma=False,oracle_gamma=False):
        super().__init__(); self.mode=mode; self.latent=latent; self.oracle_gamma=oracle_gamma
        self.dynamics=MLP(latent+3,latent,width,3)
        extent=physical_extent if mode.startswith("physical") else ssv_extent
        self.decoder=FourierDecoder(latent,modes,physical_extent,width)
        if mode.startswith("learned"):
            alpha_init = 0.400467 if alpha_init is None else float(alpha_init)
            beta_init = 0.389361 if beta_init is None else float(beta_init)
            self.alpha_raw=nn.Parameter(torch.logit(torch.tensor((alpha_init-.05)/.9)))
            self.beta_raw=nn.Parameter(torch.logit(torch.tensor(beta_init)))
        if use_gamma:
            self.gamma_net=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1))
    def scales(self):
        zero=self.decoder.frequencies.new_tensor(0.)
        if self.mode.startswith("physical"): return zero,zero
        if self.mode.startswith("hand_ssv"): return zero+.5,zero+.5
        return .05+.9*torch.sigmoid(self.alpha_raw),torch.sigmoid(self.beta_raw)
    def gamma(self,p):
        if self.oracle_gamma: return 2.0/p[:,2].square()
        if hasattr(self,"gamma_net"): return torch.nn.functional.softplus(self.gamma_net(p).squeeze(-1))
        return torch.ones(len(p),device=p.device,dtype=p.dtype)
    def clock(self,t,p): return t[None].expand(len(p),-1) if self.mode.startswith("physical") else torch.log1p(self.gamma(p)[:,None]*t[None])
    def field(self,z,p): return self.dynamics(torch.cat((z,p),-1))
    def advance(self,z,p,h):
        return z+h*self.field(z,p)
    def latent_rollout(self,params,times):
        p=params; z=torch.zeros(len(params),self.latent,device=params.device,dtype=params.dtype)
        z=self.advance(z,p,1.0); zs=[z]
        clock=self.clock(times,p)
        for j in range(1,len(times)): z=self.advance(z,p,(clock[:,j]-clock[:,j-1])[:,None]); zs.append(z)
        return torch.stack(zs,1),p
    def decode_physical(self,z,p,t,x):
        alpha,beta=self.scales(); s=1+self.gamma(p)*t; return self.decoder(z,x/s[:,None].pow(alpha),p)/s[:,None].pow(beta)
    def rollout(self,params,times,x):
        zs,p=self.latent_rollout(params,times)
        pred=torch.stack([self.decode_physical(zs[:,j],p,times[j],x) for j in range(len(times))],1)
        return pred,zs,p

def gradient_rhs(model,energy,z,p,xi):
    xi=xi.detach().requires_grad_(True); dz=model.field(z,p)
    w,wt=torch.autograd.functional.jvp(lambda zz:model.decoder(zz,xi,p),z,dz,create_graph=True)
    wx=torch.autograd.grad(w,xi,torch.ones_like(w),create_graph=True)[0]
    e=energy(w,wx); eu,eux=torch.autograd.grad(e.sum(),(w,wx),create_graph=True,retain_graph=True)
    dx_eux=torch.autograd.grad(eux,xi,torch.ones_like(eux),create_graph=True,retain_graph=True)[0]
    alpha,beta=model.scales(); intrinsic=model.gamma(p)[:,None]*(wt-alpha*xi*wx-beta*w); return intrinsic,-eu+dx_eux

def rel_curve(y,yhat,mask):
    return torch.sqrt((((yhat-y)**2*mask).sum((0,2)))/(y.square().mul(mask).sum((0,2)).clamp_min(1e-12)))

def full_rel_curve(y,yhat):
    """Full-domain RelRMSE, aggregating squared norms over trajectories."""
    return torch.sqrt((yhat-y).square().sum((0,2))/y.square().sum((0,2)).clamp_min(1e-12))

def fixed_initial_l2_curve(y,yhat):
    """Full-domain L2 error normalized per trajectory by its initial L2 norm."""
    numerator=(yhat-y).square().sum(-1)
    denominator=y[:,0].square().sum(-1,keepdim=True).clamp_min(1e-12)
    return torch.sqrt(numerator/denominator).mean(0)

def oracle_support_curve(y,yhat,x,t,window):
    """Legacy heat-support diagnostic; never used for optimization or selection."""
    mask=(torch.abs(x[None])<=window*torch.sqrt(1+t[:,None])).to(y.dtype)
    numerator=((yhat-y).square()*mask[None]).sum(-1)
    denominator=(y[:,0].square()*mask[0]).sum(-1,keepdim=True).clamp_min(1e-12)
    return torch.sqrt(numerator/denominator).mean(0)

def args():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out_dir',required=True)
    p.add_argument('--mode',choices=('physical','physical_energy','hand_ssv','hand_ssv_energy','learned_ssv','learned_ssv_energy'),required=True)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=3000); p.add_argument('--num_train',type=int,default=32)
    p.add_argument('--train_pool',type=int,default=96); p.add_argument('--num_valid',type=int,default=32); p.add_argument('--num_test',type=int,default=32); p.add_argument('--train_tmax',type=float,default=.5)
    p.add_argument('--rep_dim',type=int,default=24); p.add_argument('--latent',type=int,default=16); p.add_argument('--width',type=int,default=128)
    p.add_argument('--decoder_modes',type=int,default=33); p.add_argument('--window',type=float,default=4.5); p.add_argument('--collocation',type=int,default=24)
    p.add_argument('--lambda_gf',type=float,default=1e-4); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--eta_min',type=float,default=1e-5); p.add_argument('--device',default='cuda:0')
    p.add_argument('--benchmark',choices=('legacy_window','fixed_domain'),default='fixed_domain')
    p.add_argument('--gf_tmax',type=float,default=10.0); p.add_argument('--gf_time_points',type=int,default=32)
    p.add_argument('--scale_init',choices=('fixed','random'),default='fixed'); p.add_argument('--scale_checkpoint'); p.add_argument('--oracle_gamma',action='store_true'); p.add_argument('--scale_init_low',type=float,default=.2); p.add_argument('--scale_init_high',type=float,default=.8)
    return p.parse_args()

def main():
    a=args()
    if a.mode.startswith('learned') and not a.scale_checkpoint:
        raise ValueError('Learned SSV requires --scale_checkpoint.')
    if a.mode.startswith('hand_ssv'):
        a.oracle_gamma = True
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    d=np.load(a.data); u=torch.tensor(d['u'],device=a.device); params=torch.tensor(d['params'],device=a.device); x=torch.tensor(d['x'],device=a.device); t=torch.tensor(d['t'],device=a.device)
    perm=np.random.default_rng(123).permutation(len(u)); train_pool=perm[:a.train_pool]; it=train_pool[:a.num_train]
    iv=perm[a.train_pool:a.train_pool+a.num_valid]; ie=perm[a.train_pool+a.num_valid:a.train_pool+a.num_valid+a.num_test]
    physical_extent=float(torch.abs(x).max()) if a.benchmark=='fixed_domain' else a.window*np.sqrt(float(t[-1])+1)
    alpha_init=beta_init=None
    if a.mode.startswith('learned') and a.scale_init=='random':
        scale_rng=np.random.default_rng(10000+a.seed)
        alpha_init=float(scale_rng.uniform(a.scale_init_low,a.scale_init_high)); beta_init=float(scale_rng.uniform(a.scale_init_low,a.scale_init_high))
    model=ShiftedSSVLDNet(u.shape[-1],a.mode,a.rep_dim,a.latent,a.width,a.decoder_modes,physical_extent,a.window,alpha_init,beta_init,bool(a.scale_checkpoint),a.oracle_gamma).to(a.device)
    if a.scale_checkpoint:
        scale_state=torch.load(a.scale_checkpoint,map_location=a.device,weights_only=True)
        model.alpha_raw.data.copy_(scale_state['ar']); model.beta_raw.data.copy_(scale_state['br'])
        model.gamma_net.load_state_dict({k[2:]:v for k,v in scale_state.items() if k.startswith('g.')})
        for parameter in (model.alpha_raw,model.beta_raw): parameter.requires_grad_(False)
        for parameter in model.gamma_net.parameters(): parameter.requires_grad_(False)
    energy=EnergyDensity().to(a.device); use_energy=a.mode.endswith('energy')
    opt=torch.optim.AdamW(list(model.parameters())+(list(energy.parameters()) if use_energy else []),lr=a.lr,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.eta_min)
    ids=torch.where(t<=a.train_tmax+1e-7)[0]; obs=torch.tensor([0,len(ids)//2,len(ids)-1],device=a.device)
    if a.gf_time_points < 3:
        raise ValueError('gf_time_points must be at least 3')
    # Always include the three supervised times, then sample the remaining
    # unlabeled rollout states uniformly in physical time over (train_tmax, gf_tmax].
    gf_future=torch.linspace(a.train_tmax,a.gf_tmax,a.gf_time_points-2,device=a.device)[1:]
    gf_targets=torch.cat((t[obs],gf_future))
    gf_pick=torch.searchsorted(t,gf_targets).clamp_max(len(t)-1).unique()
    if len(gf_pick) != a.gf_time_points:
        raise ValueError(f'GF time selection produced {len(gf_pick)} unique points, expected {a.gf_time_points}')
    s=1+t; fair=(torch.abs(x[None])<=a.window*torch.sqrt(s[:,None])).float()[None]; xb=x[None].expand(a.num_train,-1)
    best=1e99; best_epoch=None; state=None; hist=[]; gf_scale=None
    for ep in range(1,a.epochs+1):
        model.train(); pred,z,p=model.rollout(params[it],t[ids],xb); truth=u[it][:,ids]; m=fair[:,ids]
        if a.benchmark=='fixed_domain':
            sq=(pred[:,obs]-truth[:,obs]).square().sum(-1)
            den=truth[:,0].square().sum(-1,keepdim=True).clamp_min(1e-6)
        else:
            sq=((pred[:,obs]-truth[:,obs])**2*m[:,obs]).sum(-1)
            den=(truth[:,obs].square()*m[:,obs]).sum(-1).clamp_min(1e-6)
        data_loss=(sq/den).mean(); raw=pred.new_tensor(0.)
        if use_energy:
            zg,pg=model.latent_rollout(params[it],t)
            zg=zg[:,gf_pick]
            zz=zg.reshape(-1,zg.shape[-1])
            pp=pg[:,None].expand(-1,len(gf_pick),-1).reshape(-1,pg.shape[-1])
            xi=(2*torch.rand(len(zz),a.collocation,device=a.device)-1)*physical_extent
            intrinsic,rhs=gradient_rhs(model,energy,zz,pp,xi); raw=(intrinsic-rhs).square().mean()
            if ep==101: gf_scale=max(float(raw.detach()),1e-12)
        lam=0. if (not use_energy or ep<=100) else a.lambda_gf*min((ep-100)/100,1.)
        normalized_gf=raw.new_tensor(0.) if lam==0 else raw/gf_scale
        loss=data_loss+lam*normalized_gf; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); scheduler.step()
        if ep==1 or ep%10==0:
            model.eval()
            with torch.no_grad():
                pv,_,_=model.rollout(params[iv],t[ids],x[None].expand(len(iv),-1))
                val=float(fixed_initial_l2_curve(u[iv][:,ids][:,obs],pv[:,obs]).mean() if a.benchmark=='fixed_domain' else rel_curve(u[iv][:,ids][:,obs],pv[:,obs],fair[:,ids][:,obs]).mean())
            rec={'epoch':ep,'valid':val,'training_loss':float(loss.detach()),'data_loss':float(data_loss.detach()),'relative_data_loss':float(data_loss.detach()),'raw_gf':float(raw.detach()),'normalized_gf':float(normalized_gf.detach()),'gf_weight':float(lam),'lr':float(opt.param_groups[0]['lr']),'alpha':float(model.scales()[0].detach()),'beta':float(model.scales()[1].detach())}; hist.append(rec)
            if val<best: best=val; best_epoch=ep; state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'energy':{k:v.detach().cpu().clone() for k,v in energy.state_dict().items()}}
            print(json.dumps(rec),flush=True)
    model.load_state_dict(state['model']); energy.load_state_dict(state['energy']); model.eval()
    with torch.no_grad(): pe,_,_=model.rollout(params[ie],t,x[None].expand(len(ie),-1))
    curve=rel_curve(u[ie],pe,fair).cpu().numpy()
    full_curve=fixed_initial_l2_curve(u[ie],pe).cpu().numpy()
    full_rel=full_rel_curve(u[ie],pe).cpu().numpy()
    oracle_curve=oracle_support_curve(u[ie],pe,x,t,a.window).cpu().numpy()
    ext=t.cpu().numpy()>a.train_tmax+1e-7; alpha,beta=model.scales()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    final_window=[r for r in hist if r['epoch']>=a.epochs-300]
    plateau_start=min(hist,key=lambda r:abs(r['epoch']-(a.epochs-300)))['valid']
    plateau_best=min(r['valid'] for r in final_window)
    final300_improvement=max(0.0,(plateau_start-plateau_best)/max(plateau_start,1e-12))
    np.save(out/'relrmse_vs_time.npy',curve); np.save(out/'full_relrmse_vs_time.npy',full_rel); np.save(out/'nerror_full_vs_time.npy',full_curve)
    np.save(out/'oracle_support_nerror_vs_time.npy',oracle_curve); np.save(out/'time.npy',t.cpu().numpy())
    summary={'mode':a.mode,'seed':a.seed,'num_train':a.num_train,'benchmark':a.benchmark,'conditioning':'gaussian_parameters','integrator':'forward_euler','decoder':'continuous_fourier','fourier_frequencies':(a.decoder_modes-1)//2,'alpha_init':alpha_init,'beta_init':beta_init,'gf_tmax':a.gf_tmax,'gf_time_points':a.gf_time_points,'domain':[float(x.min()),float(x.max())],
             'optimizer':'AdamW','scheduler':'CosineAnnealingLR','initial_lr':a.lr,'eta_min':a.eta_min,'training_steps':a.epochs,'selected_checkpoint_epoch':best_epoch,'final300_validation_improvement_fraction':final300_improvement,'final300_validation_improves_less_than_1pct':bool(final300_improvement<.01),
             'supervised_frames':3,'observed_times':[float(t[j]) for j in obs],'validation_times':[float(t[j]) for j in obs],'gf_times':[float(t[j]) for j in gf_pick],
             'observed_full_relrmse':float(full_rel[~ext].mean()),'extrapolation_full_relrmse':float(full_rel[ext].mean()),'final_full_relrmse':float(full_rel[-1]),
             'observed_nerror_full':float(full_curve[~ext].mean()),'extrapolation_nerror_full':float(full_curve[ext].mean()),'final_nerror_full':float(full_curve[-1]),
             'oracle_support_observed_nerror':float(oracle_curve[~ext].mean()),'oracle_support_extrapolation_nerror':float(oracle_curve[ext].mean()),'oracle_support_final_nerror':float(oracle_curve[-1]),
             'legacy_extrapolation_relrmse':float(curve[ext].mean()),'alpha':float(alpha.detach()),'beta':float(beta.detach()),'best_valid':best,'args':vars(a)}
    torch.save(state,out/'model.pt'); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n'); (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print('FINAL',json.dumps(summary),flush=True)
if __name__=='__main__': main()
