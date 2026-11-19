"""Frozen per-IC gamma LDNet on the restored [-16,16) spatial grid."""
from __future__ import annotations

import torch

from ic_gamma_model import ExternalGammaModel, pack_inputs, per_ic_metrics
from original_scale_model import gradient_rhs


EXTENT = 16.
SPATIAL_POINTS = 192
FOURIER_FREQUENCIES = 16


class Domain16GammaModel(ExternalGammaModel):
    """Same learned parameters/RNG order as the old model, new Fourier extent.

    The persistent extent buffer makes an old Domain24 loader fail loudly
    instead of silently interpreting these weights with the wrong basis.
    """
    def __init__(self, physical=False):
        super().__init__(physical=physical)
        self.experts[0].decoder.extent = EXTENT
        self.register_buffer('spatial_extent', torch.tensor(EXTENT))
        self.assert_geometry()

    def assert_geometry(self):
        if float(self.spatial_extent) != EXTENT:
            raise ValueError('This checkpoint does not use extent16')
        if self.experts[0].decoder.extent != EXTENT:
            raise ValueError('Fourier decoder extent does not match GF/input domain')
        if self.experts[0].decoder.num_frequencies != FOURIER_FREQUENCIES:
            raise ValueError('This experiment requires16 Fourier frequencies')
        if self.field_dim != SPATIAL_POINTS or self.input_dim != SPATIAL_POINTS+1:
            raise ValueError('Expected192 initial-field values plus fixed gamma')
        if sum(p.numel() for p in self.parameters()) != 91329:
            raise ValueError('Backbone parameter budget changed')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'spatial_extent' not in state_dict:
            raise ValueError('Domain16 checkpoints must record spatial_extent')
        if float(state_dict['spatial_extent']) != EXTENT:
            raise ValueError('Checkpoint spatial extent is not16')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.assert_geometry()
        return result


def validate_grid(x, t):
    """Require the fixed grid chosen for this controlled domain comparison."""
    expected_x = torch.arange(SPATIAL_POINTS, device=x.device, dtype=x.dtype)*(2*EXTENT/SPATIAL_POINTS)-EXTENT
    expected_t = torch.arange(81, device=t.device, dtype=t.dtype)/8
    if x.shape != expected_x.shape or t.shape != expected_t.shape:
        raise ValueError('Expected192 spatial points and81 times through10')
    torch.testing.assert_close(x, expected_x, rtol=0, atol=2e-6)
    torch.testing.assert_close(t, expected_t, rtol=0, atol=0)


def gf_points(model, count, collocation, device):
    """Use the decoder extent for the GF coordinate domain as well."""
    extent = model.experts[0].decoder.extent
    if extent != EXTENT:
        raise ValueError('GF and Fourier domains must both use extent16')
    return (2*torch.rand(count, collocation, device=device)-1)*extent


def raw_gf_loss(model, energy, ic, t, gf_indices, collocation=24):
    """Original GF residual; only the coordinate sampling extent changes."""
    if model.charts != 1:
        raise ValueError('The domain16 experiment uses one global chart')
    expert = model.experts[0]
    p = model.encode(ic)
    z, p = expert.latent_rollout(p, t)
    selected = z[:, gf_indices]
    zz = selected.reshape(-1, model.latent)
    pp = p[:, None].expand(-1, len(gf_indices), -1).reshape(-1, model.condition_dim)
    xi = gf_points(model, len(zz), collocation, ic.device)
    intrinsic, rhs = gradient_rhs(expert, energy, zz, pp, xi)
    return (intrinsic-rhs).square().mean()


__all__ = ['Domain16GammaModel', 'pack_inputs', 'per_ic_metrics', 'raw_gf_loss',
           'validate_grid', 'gf_points', 'EXTENT', 'SPATIAL_POINTS', 'FOURIER_FREQUENCIES']
