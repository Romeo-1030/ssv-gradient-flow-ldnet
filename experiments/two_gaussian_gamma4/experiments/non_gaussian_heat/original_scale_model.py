"""Original-size SSV-LDNet backbone adapted to non-Gaussian initial fields.

The preserved release supplies the MLP, Fourier decoder, shifted-Euler latent
rollout and local-energy implementation. The necessary input adaptation is a
shared 192-value IC encoder producing a 16-dimensional conditioning vector,
in place of the original three Gaussian parameters. Every expert keeps the
original latent dimension, two hidden layers and 33 Fourier coefficients.

Scales and chart centers are frozen. Training grids must begin at t=0; the
inherited shifted initialization is z0=f(0,p), not z0=encoder(ic).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
ORIGINAL_SOURCE = ROOT.parents[1] / 'source' / 'ssv-gradient-flow-ldnet-main' / 'scripts' / 'train.py'


def _load_original():
    spec = importlib.util.spec_from_file_location('preserved_ssv_ldnet_original', ORIGINAL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load preserved model: {ORIGINAL_SOURCE}')
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        # Reading the preserved implementation must not create source artifacts.
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


original = _load_original()
MLP = original.MLP
FourierDecoder = original.FourierDecoder
EnergyDensity = original.EnergyDensity
gradient_rhs = original.gradient_rhs


class ConditionalExpert(original.ShiftedSSVLDNet):
    """Original expert with a learned IC condition and a fixed local origin."""

    def __init__(self, width=128, condition_dim=16, center=0., extent=24.):
        # hand_ssv avoids creating unused trainable exponent parameters. The
        # overrides below return the externally fitted, frozen exponent pair.
        super().__init__(
            nx=192, mode='hand_ssv', rep_dim=24, latent=16, width=width,
            modes=33, physical_extent=extent, ssv_extent=extent,
            use_gamma=False, oracle_gamma=False,
        )
        self.condition_dim = condition_dim
        self.dynamics = MLP(self.latent + condition_dim, self.latent, width, 3)
        self.decoder.coeff = MLP(self.latent + condition_dim, 33, width, 3)
        self.register_buffer('scale_pair', torch.tensor([.5, .5]))
        self.register_buffer('center', torch.tensor(float(center)))

    def scales(self):
        return self.scale_pair[0], self.scale_pair[1]

    def gamma(self, p):
        return torch.ones(len(p), device=p.device, dtype=p.dtype)

    def decode_physical(self, z, p, t, x):
        alpha, beta = self.scales()
        s = 1 + self.gamma(p) * t
        q = (x - self.center) / s[:, None].pow(alpha)
        return self.decoder(z, q, p) / s[:, None].pow(beta)

    def rollout_batched(self, params, times, x):
        """Original Euler dynamics, with time-batched Fourier decoding.

        The inherited ``rollout`` remains available as an independent loop
        reference. Both evaluate exactly the same architecture and time grid.
        ``x`` is a shared 1D grid or a per-trajectory B-by-X grid.
        """
        zs, p = self.latent_rollout(params, times)
        batch, count, _ = zs.shape
        if x.ndim == 1:
            x = x[None].expand(batch, -1)
        if x.ndim != 2 or x.shape[0] != batch:
            raise ValueError('x must have shape X or B,X')
        alpha, beta = self.scales()
        s = 1 + self.gamma(p)[:, None] * times[None]
        q = (x[:, None] - self.center) / s[:, :, None].pow(alpha)
        conditions = p[:, None].expand(-1, count, -1)
        values = self.decoder(
            zs.reshape(batch * count, self.latent),
            q.reshape(batch * count, x.shape[-1]),
            conditions.reshape(batch * count, self.condition_dim),
        ).reshape(batch, count, x.shape[-1])
        return values / s[:, :, None].pow(beta), zs, p


class OriginalScaleModel(nn.Module):
    """One global expert or three full-size experts with fixed physical POU."""

    def __init__(self, charts=1, width=128, condition_dim=16, input_dim=192):
        super().__init__()
        if charts not in (1, 3):
            raise ValueError('Supported chart counts are one and three')
        if min(width, condition_dim, input_dim) < 1:
            raise ValueError('Network dimensions must be positive')
        self.charts = charts
        self.width = width
        self.condition_dim = condition_dim
        self.input_dim = input_dim
        self.latent = 16
        self.encoder = MLP(input_dim, condition_dim, 128, 3)
        centers = [0.] if charts == 1 else [-4., 0., 4.]
        self.register_buffer('centers', torch.tensor(centers))
        self.experts = nn.ModuleList([
            ConditionalExpert(width, condition_dim, center, 24.) for center in centers
        ])

    def encode(self, ic):
        if ic.ndim != 2 or ic.shape[-1] != self.input_dim:
            raise ValueError(f'Expected an initial field of shape B,{self.input_dim}')
        return self.encoder(ic)

    def set_scales(self, scales):
        reference = self.experts[0].scale_pair
        values = torch.as_tensor(scales, device=reference.device, dtype=reference.dtype)
        if values.shape != (self.charts, 2) or not torch.isfinite(values).all():
            raise ValueError(f'Expected finite scale pairs of shape {self.charts},2')
        with torch.no_grad():
            for expert, pair in zip(self.experts, values):
                expert.scale_pair.copy_(pair)

    def get_scales(self):
        return torch.stack([expert.scale_pair for expert in self.experts])

    def weights(self, x):
        if x.ndim != 1:
            raise ValueError('The chart blend expects a shared one-dimensional x grid')
        if self.charts == 1:
            return x.new_ones((1, len(x)))
        logits = -.5 * ((x[None] - self.centers[:, None]) / 4.) ** 2
        return logits.softmax(dim=0)

    def forward(self, ic, t, x, components=False):
        if t.ndim != 1 or len(t) == 0:
            raise ValueError('t must be a nonempty one-dimensional grid beginning at zero')
        p = self.encode(ic)
        fields = [expert.rollout_batched(p, t, x)[0] for expert in self.experts]
        parts = torch.stack(fields, dim=1)
        pred = (parts * self.weights(x)[None, :, None]).sum(dim=1)
        return (pred, parts) if components else pred


__all__ = [
    'OriginalScaleModel', 'ConditionalExpert', 'MLP', 'FourierDecoder',
    'EnergyDensity', 'gradient_rhs', 'original', 'ORIGINAL_SOURCE',
]
