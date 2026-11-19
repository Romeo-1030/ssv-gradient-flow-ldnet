"""Reproducible CPU pilot: periodic heat reference and fixed-dictionary chart LDNet."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

def initial(x, family, rng):
    c = rng.uniform(-1, 1); a = rng.uniform(.8, 1.2)
    if family in ('asymmetric', 'bimodal'):
        gap = rng.uniform(1, 2) if family == 'asymmetric' else rng.uniform(2.5, 4)
        w = rng.uniform(.5, 1); ratio = rng.uniform(.2, .5) if family == 'asymmetric' else rng.uniform(.8, 1.2)
        return a*(np.exp(-.5*((x-c+gap/2)/w)**2)+ratio*np.exp(-.5*((x-c-gap/2)/(.7*w))**2))
    r = rng.uniform(1.5, 3); q = (x-c)/r
    if family == 'flat': return a*np.exp(-q**8)
    out = np.zeros_like(x); mask = abs(q)<1
    out[mask] = a*np.exp(1-1/(1-q[mask]**2))
    return out

def heat(u0, t, length=48.):
    k = 2*np.pi*np.fft.rfftfreq(len(u0), d=length/len(u0))
    return np.fft.irfft(np.fft.rfft(u0)[None]*np.exp(-t[:,None]*k[None]**2), n=len(u0), axis=-1)

def dataset():
    rng = np.random.default_rng(71); x = np.linspace(-24,24,192,endpoint=False); t = np.linspace(0,6,49)
    ys=[]; labels=[]; splits=[]
    for family in ('asymmetric','bimodal','flat','bump'):
        for i in range(24):
            # All bump ICs are held out; other families have disjoint IC draws.
            split = 'ood' if family=='bump' else ('train' if i<12 else 'valid' if i<16 else 'test')
            ys.append(heat(initial(x,family,rng),t)); labels.append(family); splits.append(split)
    return x,t,np.array(ys,dtype=np.float32),np.array(labels),np.array(splits)

def mlp(ni,no,w): return nn.Sequential(nn.Linear(ni,w),nn.SiLU(),nn.Linear(w,no))

class Model(nn.Module):
    def __init__(self, mode, width):
        super().__init__(); self.mode=mode; self.charts=1 if mode=='global' else 3
        self.encoder=mlp(48,8,width)
        self.dynamics=nn.ModuleList([mlp(8,8,width) for _ in range(self.charts)])
        self.decoders=nn.ModuleList([mlp(8,17,width) for _ in range(self.charts)])
        self.register_buffer('dictionary',torch.tensor([[0.,0.],[.25,.25],[.5,.5],[.75,.75]]))
        self.register_buffer('assignment',torch.tensor([2] if self.charts==1 else [1,2,3]))
        self.register_buffer('centers',torch.tensor([0.] if self.charts==1 else [-4.,0.,4.]))
        self.register_buffer('enabled',torch.ones(self.charts))
    def weights(self,x):
        w=torch.exp(-.5*((x[None]-self.centers[:,None])/4.)**2)*self.enabled[:,None]
        return w/w.sum(0,keepdim=True).clamp_min(1e-30)
    def forward(self,ic,t,x,components=False):
        z0=self.encoder(ic[:,::4]); outputs=[]
        for j in range(self.charts):
            alpha,beta=self.dictionary[self.assignment[j]] if self.mode!='pou' else self.dictionary[0]
            s=1+t; clock=t if self.mode=='pou' else torch.log1p(t)
            z=z0; zs=[z]
            for n in range(1,len(t)):
                z=z+(clock[n]-clock[n-1])*self.dynamics[j](z); zs.append(z)
            coeff=self.decoders[j](torch.stack(zs,1))
            q=(x[None]-self.centers[j])/s[:,None]**alpha
            phase=np.pi*q[:,:,None]*torch.arange(1,9,device=x.device)/24
            basis=torch.cat([torch.ones_like(q[:,:,None]),phase.cos(),phase.sin()],-1)
            outputs.append(torch.einsum('btm,txm->btx',coeff,basis)/s[None,:,None]**beta)
        parts=torch.stack(outputs,1); pred=(parts*self.weights(x)[None,:,None]).sum(1)
        return (pred,parts) if components else pred

def loss(pred,y,indices=None):
    if indices is not None: pred=pred[:,indices]; y=y[:,indices]
    return ((pred-y).square().sum(-1)/y[:,0].square().sum(-1)[:,None].clamp_min(1e-12)).mean()

def adapt(model,ic,t,x,y,merge,indices=None):
    """Discrete validation-only coordinate search; no optimizer or continuous SSV fit."""
    events=[]
    with torch.no_grad():
        base=float(loss(model(ic,t,x),y,indices))
        for j in range(model.charts):
            old=int(model.assignment[j]); best=old; score=base
            for k in range(len(model.dictionary)):
                model.assignment[j]=k; value=float(loss(model(ic,t,x),y,indices))
                if value<score*.995: score=value; best=k
            model.assignment[j]=best
            if best!=old: events.append({'op':'swap','chart':j,'old':old,'new':best}); base=score
        if merge:
            _,p=model(ic,t,x,True)
            for j in range(1,model.charts):
                # Merge only adjacent experts with matching SSV and similar overlap profiles.
                overlap=(model.weights(x)[j]>.1)&(model.weights(x)[j-1]>.1)
                disagreement=float((p[:,j,:,overlap]-p[:,j-1,:,overlap]).square().mean().sqrt())
                if model.enabled[j] and model.assignment[j]==model.assignment[j-1] and disagreement<.05:
                    model.enabled[j]=0; value=float(loss(model(ic,t,x),y,indices))
                    if value<=base*1.01+1e-10: events.append({'op':'merge','chart':j,'disagreement':disagreement}); base=value
                    else: model.enabled[j]=1
    return events

def metrics(pred,y,x,t):
    e=(pred-y).double(); y=y.double(); curve=(e.square().sum((0,2))/y.square().sum((0,2))).sqrt()
    return {'observed':float(curve[t<=.5].mean()),'short':float(curve[(t>.5)&(t<=2)].mean()),'long':float(curve[t>2].mean()),'final':float(curve[-1]),'curve':curve.tolist(),
            'local_rmse':[float(e[:,:,m].square().mean().sqrt()) for m in (x<-2,abs(x)<=2,x>2)]}

def run(args):
    torch.set_num_threads(1); x,t,y,labels,splits=dataset(); out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'data.npz',x=x,t=t,u=y,family=labels,split=splits)
    xt,tt,yt=map(lambda a:torch.tensor(a,dtype=torch.float32),(x,t,y)); obs=tt<=.5
    ix={s:np.flatnonzero(splits==s) for s in ('train','valid','test','ood')}; reports=[]
    for seed in range(args.seeds):
      for mode in ('global','pou','dictionary','swap','merge_swap'):
        torch.manual_seed(seed)
        # Search width to match global trainable parameter budget (within 3%).
        target=sum(p.numel() for p in Model('global',64).parameters())
        width=min(range(4,80),key=lambda w:abs(sum(p.numel() for p in Model(mode,w).parameters())-target))
        torch.manual_seed(seed); m=Model(mode,width); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=1e-6)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
        best=float('inf'); selected=None; events=[]; history=[]; start=time.perf_counter()
        for step in range(1,args.steps+1):
            m.train(); pred=m(yt[ix['train'],0],tt[obs],xt); l=loss(pred,yt[ix['train']][:,obs],[0,2,4]); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); scheduler.step()
            if step in (args.steps//3,2*args.steps//3) and mode in ('swap','merge_swap'):
                events.extend([dict(step=step,**e) for e in adapt(m,yt[ix['valid'],0],tt[obs],xt,yt[ix['valid']][:,obs],mode=='merge_swap',[0,2,4])])
            if step==1 or step%10==0 or step==args.steps:
                with torch.no_grad(): v=float(loss(m(yt[ix['valid'],0],tt[obs],xt),yt[ix['valid']][:,obs],[0,2,4]))
                history.append({'step':step,'train':float(l.detach()),'valid':v})
                if v<best: best=v; selected=copy.deepcopy(m.state_dict()); selected_step=step
        m.load_state_dict(selected); record={'mode':mode,'seed':seed,'parameters':sum(p.numel() for p in m.parameters()),'width':width,'steps':args.steps,'selected_step':selected_step,'seconds':time.perf_counter()-start,'events':events,'assignment':m.assignment.tolist(),'active_charts':int(m.enabled.sum()),'history':history}
        with torch.no_grad():
            for split in ('test','ood'):
                pred,parts=m(yt[ix[split],0],tt,xt,True); record[split]=metrics(pred,yt[ix[split]],xt,tt)
                # Profile drift in each expert's own transformed coordinate grid.
                drifts=[]
                q=np.linspace(-4,4,64)
                for j in range(m.charts):
                    a,b=m.dictionary[m.assignment[j]].tolist() if mode!='pou' else (0.,0.)
                    profiles=np.array([[np.interp(m.centers[j].item()+q*(1+time_)**a,x,row)*(1+time_)**b for time_,row in zip(t,sample)] for sample in parts[:,j].numpy()])
                    drifts.append(float(np.sqrt(np.mean((profiles[:,1:]-profiles[:,:-1])**2))/np.diff(t).mean()))
                record[split]['transformed_profile_temporal_rms']=drifts
                np.savez_compressed(out/f'{mode}_{seed}_{split}.npz',prediction=pred.numpy(),ids=ix[split])
        torch.save(selected,out/f'{mode}_{seed}.pt'); reports.append(record)
        (out/'results.json').write_text(json.dumps(reports,indent=2)); print(mode,seed,record['test']['long'],record['ood']['long'],flush=True)
    config={'steps':args.steps,'seeds':args.seeds,'nu':1,'domain':[-24,24],'boundary':'periodic','nx':192,'nt':49,'observations':[0,.25,.5],'train':36,'valid':12,'test':24,'ood':24,'dictionary':[[0,0],[.25,.25],[.5,.5],[.75,.75]]}
    (out/'config.json').write_text(json.dumps(config,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=300); p.add_argument('--seeds',type=int,default=2); p.add_argument('--out',default='runs/pilot'); run(p.parse_args())
