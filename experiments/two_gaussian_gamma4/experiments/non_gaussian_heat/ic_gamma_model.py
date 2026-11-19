"""One shared original-size LDNet, with an external frozen gamma per IC.

The final input column is a supplied coefficient, never a learned parameter.
The encoder still sees only the initial field. All arms allocate identical
parameters; the physical controls' extra conditioning feature is zero.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from original_scale_model import OriginalScaleModel, ConditionalExpert, MLP


class ExternalGammaExpert(ConditionalExpert):
    def __init__(self, physical=False):
        super().__init__(width=128, condition_dim=17, center=0., extent=24.)
        if physical:
            self.mode = 'physical'
            self.scale_pair.zero_()

    def gamma(self, p):
        if self.mode.startswith('physical'):
            return torch.ones(len(p), device=p.device, dtype=p.dtype)
        return p[:, -1].exp()


class ExternalGammaModel(OriginalScaleModel):
    def __init__(self, physical=False):
        nn.Module.__init__(self)
        self.charts, self.width, self.condition_dim = 1, 128, 17
        self.input_dim, self.field_dim, self.latent = 193, 192, 16
        self.physical = bool(physical)
        self.encoder = MLP(self.field_dim, 16, 128, 3)
        self.register_buffer('centers', torch.tensor([0.]))
        self.experts = nn.ModuleList([ExternalGammaExpert(physical)])

    def encode(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError('Expected initial field plus frozen gamma, shape B,193')
        gamma = inputs[:, -1:].detach()
        if not torch.isfinite(gamma).all() or (gamma <= 0).any():
            raise ValueError('Every supplied gamma must be finite and positive')
        field_code = self.encoder(inputs[:, :self.field_dim])
        gamma_code = torch.zeros_like(gamma) if self.physical else gamma.log()
        return torch.cat((field_code, gamma_code), dim=-1)


def pack_inputs(initial, gamma):
    if initial.ndim != 2 or initial.shape[1] != 192:
        raise ValueError('Expected 192 initial-field samples')
    gamma = torch.as_tensor(gamma, device=initial.device, dtype=initial.dtype).reshape(-1)
    if len(gamma) != len(initial) or not torch.isfinite(gamma).all() or (gamma <= 0).any():
        raise ValueError('One positive gamma per IC is required')
    return torch.cat((initial, gamma.detach()[:, None]), dim=-1).detach()


def per_ic_metrics(pred, truth, t):
    error = (pred-truth).double().square().sum(-1)
    denom = truth.double().square().sum(-1).clamp_min(1e-12)
    curve = (error/denom).sqrt()
    return {'per_ic_long': curve[:, t > 2].mean(1).cpu().tolist(),
            'per_ic_final': curve[:, -1].cpu().tolist(),
            'per_ic_observed': curve[:, t <= 1.25].mean(1).cpu().tolist()}
