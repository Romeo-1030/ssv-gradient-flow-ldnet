"""Two methods only: pretrain SSV by profile alignment, freeze, then train LDNet."""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import ROOT, Model, dataset, loss, metrics, heat

class StagedModel(Model):
    def __init__(self, charts, width):
        super().__init__('global' if charts==1 else 'dictionary',width)
        # One separately fitted pair per chart; no dictionary search or adaptation.
        self.dictionary=torch.full((charts,2),.5)
        self.assignment=torch.arange(charts)

class Scale(nn.Module):
    def __init__(self, charts):
        super().__init__()
        self.raw=nn.Parameter(torch.randn(charts,2)*.5)
    def values(self):
        return torch.stack((.05+.9*self.raw[:,0].sigmoid(),self.raw[:,1].sigmoid()),-1)

def aligned_initial(ic,x,t,scales,centers):
    """Predict observations with a stationary initial template in each SSV chart.

    Interpolate u0(c+(x-c)/(1+t)^alpha), then multiply by (1+t)^(-beta).
    No neural template, latent network, future frame, or IC label is fitted here.
    """
    parts=[]
    for pair,c in zip(scales,centers):
        a,b=pair
        location=c+(x[None]-c)/(1+t[:,None])**a
        normalized=2*(location-x[0])/(x[-1]-x[0])-1
        grid=torch.stack((normalized,torch.zeros_like(normalized)),-1)
        sampled=F.grid_sample(ic[:,None,None,:],grid[None].expand(len(ic),-1,-1,-1),mode='bilinear',padding_mode='zeros',align_corners=True)[:,0]
        parts.append(sampled/(1+t[None,:,None])**b)
    return torch.stack(parts,1)

def scale_loss(ic,truth,x,t,values,centers,weights):
    p=aligned_initial(ic,x,t,values,centers)
    # Each pair is identified from its own fixed physical chart; global uses w=1.
    numerator=((p-truth[:,None]).square()*weights[None,:,None]).sum(-1)
    denominator=(ic[:,None].square()*weights[None]).sum(-1).clamp_min(1e-8)
    return (numerator/denominator[:,:,None]).mean()

def scale_observations(y,t,frames):
    if frames==3:
        obs=torch.tensor([0,2,4]); return t[obs],y[:,obs]
    if frames!=17: raise ValueError('Supported scale frame counts: 3 or 17')
    times=np.linspace(0,.5,17)
    # Evaluate the same Fourier heat semigroup, not interpolation between saved times.
    truth=np.stack([heat(ic.astype(np.float64),times) for ic in y[:,0].numpy()])
    truth=torch.tensor(truth,dtype=y.dtype)
    # Preserve exact shared saved-frame values (including their float32 rounding).
    truth[:,[0,8,16]]=y[:,[0,2,4]]
    return torch.tensor(times,dtype=t.dtype),truth

def pretrain(charts,ix,y,x,t,steps,restarts,frames=3):
    structure=StagedModel(charts,16); weights=structure.weights(x); centers=structure.centers
    tt,target=scale_observations(y,t,frames)
    best=float('inf'); selected=None; records=[]
    for restart in range(restarts):
        torch.manual_seed(9100+restart); scale=Scale(charts); opt=torch.optim.Adam(scale.parameters(),lr=.02)
        for step in range(1,steps+1):
            v=scale_loss(y[ix['train'],0],target[ix['train']],x,tt,scale.values(),centers,weights)
            opt.zero_grad(); v.backward(); opt.step()
            if step==1 or step%25==0 or step==steps:
                with torch.no_grad(): valid=float(scale_loss(y[ix['valid'],0],target[ix['valid']],x,tt,scale.values(),centers,weights))
                records.append({'restart':restart,'step':step,'train':float(v.detach()),'valid':valid,'scales':scale.values().detach().tolist()})
                if valid<best:
                    best=valid; selected=scale.values().detach().clone(); selection={'restart':restart,'step':step,'valid':valid}
    return selected,{'selection':selection,'history':records,'values':selected.tolist(),'observation_times':tt.tolist()}

def train(args):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    x,t,y,labels,splits=dataset(); out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'data.npz',x=x,t=t,u=y,family=labels,split=splits)
    x,t,y=map(lambda a:torch.tensor(a,dtype=torch.float32),(x,t,y))
    ix={s:np.flatnonzero(splits==s) for s in ('train','valid','test','ood')}
    target=sum(p.numel() for p in StagedModel(1,64).parameters()); reports=[]
    config=vars(args)|{'nu':1,'gamma':1,'scale_bounds':{'alpha':[.05,.95],'beta':[0,1]},'supervised_times':[0,.25,.5],'scale_observation_times':scale_observations(y,t,args.scale_frames)[0].tolist(),'scale_objective':'fixed-initial-template, chart-weighted alignment','split_counts':{k:len(v) for k,v in ix.items()},'latent':8,'fourier_frequencies':8,'ic_samples':48,'pou_centers':[-4,0,4],'pou_width':4,'scale_fitted_once_per_method_shared_across_latent_seeds':True}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for charts,name in ((1,'global'),(3,'charts3')):
        start=time.perf_counter(); scales,scale_record=pretrain(charts,ix,y,x,t,args.scale_steps,args.restarts,args.scale_frames)
        scale_record['seconds']=time.perf_counter()-start
        (out/(name+'_pretrain.json')).write_text(json.dumps(scale_record,indent=2)); torch.save(scales,out/(name+'_scales.pt'))
        print('PRETRAIN',name,scales.tolist(),flush=True)
        width=min(range(4,100),key=lambda w:abs(sum(p.numel() for p in StagedModel(charts,w).parameters())-target))
        for seed in range(args.seeds):
            torch.manual_seed(seed); m=StagedModel(charts,width); m.dictionary.copy_(scales)
            assert not m.dictionary.requires_grad
            frozen=m.dictionary.clone(); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=1e-6)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
            best=float('inf'); state=None; history=[]; start=time.perf_counter()
            for step in range(1,args.steps+1):
                pred=m(y[ix['train'],0],t[:5],x); v=loss(pred,y[ix['train'],:5],[0,2,4])
                opt.zero_grad(); v.backward(); nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); scheduler.step()
                if step==1 or step%10==0 or step==args.steps:
                    with torch.no_grad(): valid=float(loss(m(y[ix['valid'],0],t[:5],x),y[ix['valid'],:5],[0,2,4]))
                    history.append({'step':step,'train':float(v.detach()),'valid':valid})
                    if valid<best: best=valid; state=copy.deepcopy(m.state_dict()); selected_step=step
                if step%500==0: print('TRAIN',name,seed,step,valid,flush=True)
            m.load_state_dict(state); torch.testing.assert_close(frozen,m.dictionary,rtol=0,atol=0)
            record={'method':name,'seed':seed,'width':width,'parameters':sum(p.numel() for p in m.parameters()),'steps':args.steps,'selected_step':selected_step,'seconds':time.perf_counter()-start,'scales':scales.tolist(),'scales_unchanged':True,'history':history}
            with torch.no_grad():
                for split in ('test','ood'):
                    pred=m(y[ix[split],0],t,x); assert torch.isfinite(pred).all()
                    record[split]=metrics(pred,y[ix[split]],x,t)
                    record[split]['initial']=record[split]['curve'][0]
                    np.savez_compressed(out/(name+'_'+str(seed)+'_'+split+'.npz'),prediction=pred.numpy(),ids=ix[split])
            torch.save(state,out/(name+'_'+str(seed)+'.pt')); reports.append(record)
            (out/'results.json').write_text(json.dumps(reports,indent=2))
            print('DONE',name,seed,record['test']['long'],record['ood']['long'],flush=True)
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=3000); p.add_argument('--scale-steps',type=int,default=1500); p.add_argument('--scale-frames',type=int,choices=(3,17),default=3); p.add_argument('--restarts',type=int,default=3); p.add_argument('--seeds',type=int,default=3); p.add_argument('--out',default='runs/two_stage'); train(p.parse_args())
