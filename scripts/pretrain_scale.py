"""Refit scale from three frames (reconstructed; original checkpoint is supplied)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = np.random.default_rng(10000 + seed)
        a, b = rng.uniform(.2, .8, 2)
        self.ar = nn.Parameter(torch.logit(torch.tensor((a-.05)/.9, dtype=torch.float32)))
        self.br = nn.Parameter(torch.logit(torch.tensor(b, dtype=torch.float32)))
        self.g = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, params, times):
        clock = torch.log1p(nn.functional.softplus(self.g(params)) * times)
        return (.05 + .9*self.ar.sigmoid())*clock, -self.br.sigmoid()*clock


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/single_gaussian_heat_fixed_domain_v1.npz')
    p.add_argument('--out_dir', type=Path, default=Path('outputs/scale_refit'))
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--restarts', type=int, default=10)
    a = p.parse_args()
    d = np.load(a.data)
    ids = [int(np.argmin(abs(d['t']-t))) for t in (0, .25, .5)]
    assert np.allclose(d['t'][ids], [0, .25, .5])
    u = torch.tensor(d['u'][:, ids]); x = torch.tensor(d['x'])
    weights = u.square(); mass = weights.sum(-1)
    center = (weights*x).sum(-1)/mass
    width = (2*(weights*(x-center[..., None]).square()).sum(-1)/mass).sqrt()
    peak = u.abs().amax(-1)
    target_w = (width/width[:, :1]).log(); target_a = (peak/peak[:, :1]).log()
    params = torch.tensor(d['params']); times = torch.tensor(d['t'][ids])
    perm = np.random.default_rng(123).permutation(len(u))
    train, valid = perm[:32], perm[96:128]
    def loss(model, indices):
        w, h = model(params[indices], times)
        return ((w-target_w[indices]).square()+(h-target_a[indices]).square()).mean()
    best = float('inf'); records = []; state = None
    for seed in range(a.restarts):
        torch.manual_seed(seed); model = Scale(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(1, a.epochs+1):
            opt.zero_grad(); loss(model, train).backward(); opt.step()
            if step == 1 or step % 25 == 0 or step == a.epochs:
                with torch.no_grad(): value = float(loss(model, valid))
                if value < best:
                    best = value; state = {k:v.detach().clone() for k,v in model.state_dict().items()}
                    selected = {'seed':seed, 'step':step, 'valid_loss':value}
        records.append({'seed':seed, 'valid_loss':value})
    a.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, a.out_dir/'selected_model.pt')
    (a.out_dir/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    (a.out_dir/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(selected)


if __name__ == '__main__':
    main()
