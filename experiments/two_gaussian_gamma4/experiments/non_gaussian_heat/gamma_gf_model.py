"""Fixed heat exponents with constant or IC-conditioned chart clocks.

Each expert retains the original-size conditional shifted-Euler LDNet. An
independent IC-to-gamma predictor is pretrained, then frozen in its entirety.
Gamma never depends on the subsequently trainable LDNet encoder. GF is evaluated per
expert with ONE shared energy density, then averaged across experts. It is
not a GF constraint on the partition-of-unity blend.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from original_scale_model import OriginalScaleModel, ConditionalExpert, MLP, gradient_rhs


METHODS = ('global_fixed_gf', 'charts2_gamma_gf')


class GammaPredictor(nn.Module):
    """A separate SSV module, initialized to gamma=1 for both chart outputs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192, 16), nn.Tanh(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(1.)))

    def forward(self, ic):
        return F.softplus(self.net(ic))


class ContextDecoder(nn.Module):
    """Only the 16-dimensional code conditions the original Fourier decoder."""
    def __init__(self, decoder):
        super().__init__()
        self.base = decoder

    def forward(self, z, q, p):
        return self.base(z, q, p[:, :16])


class GammaExpert(ConditionalExpert):
    def __init__(self, center=0.):
        super().__init__(width=128, condition_dim=16, center=center, extent=24.)
        self.decoder = ContextDecoder(self.decoder)

    def gamma(self, p):
        # The 17th context entry carries an independently computed frozen rate.
        return p[:, 16]

    def field(self, z, p):
        return self.dynamics(torch.cat((z, p[:, :16]), -1))

    def rollout_batched(self, params, times, x):
        zs, p = self.latent_rollout(params, times)
        batch, count, _ = zs.shape
        if x.ndim == 1:
            x = x[None].expand(batch, -1)
        alpha, beta = self.scales()
        s = 1 + self.gamma(p)[:, None] * times[None]
        q = (x[:, None] - self.center) / s[:, :, None].pow(alpha)
        conditions = p[:, None].expand(-1, count, -1)
        values = self.decoder(zs.reshape(batch * count, 16),
                              q.reshape(batch * count, x.shape[-1]),
                              conditions.reshape(batch * count, 17)).reshape(batch, count, x.shape[-1])
        return values / s[:, :, None].pow(beta), zs, p


class GammaGFModel(OriginalScaleModel):
    def __init__(self, method, gamma_state=None):
        if method not in METHODS:
            raise ValueError(method)
        nn.Module.__init__(self)
        self.method = method
        self.charts = 1 if method == 'global_fixed_gf' else 2
        self.width, self.condition_dim, self.input_dim, self.latent = 128, 16, 192, 16
        self.encoder = MLP(192, 16, 128, 3)
        centers = [0.] if self.charts == 1 else [-4., 4.]
        self.register_buffer('centers', torch.tensor(centers))
        self.experts = nn.ModuleList([GammaExpert(center) for center in centers])
        self.ssv = None
        if self.charts == 2:
            self.ssv = GammaPredictor()
            if gamma_state is not None:
                self.ssv.load_state_dict(gamma_state)
            self.ssv.requires_grad_(False)
            self.ssv.eval()

    def gamma_values(self, ic):
        if self.ssv is None:
            return ic.new_ones((len(ic), 1))
        return self.ssv(ic)

    def contexts(self, ic):
        p = self.encode(ic)
        gamma = self.gamma_values(ic)
        return [torch.cat((p, gamma[:, j:j+1]), -1) for j in range(self.charts)]

    def forward(self, ic, t, x, components=False):
        fields = [expert.rollout_batched(p, t, x)[0]
                  for expert, p in zip(self.experts, self.contexts(ic))]
        parts = torch.stack(fields, dim=1)
        pred = (parts * self.weights(x)[None, :, None]).sum(1)
        return (pred, parts) if components else pred

    def train(self, mode=True):
        super().train(mode)
        if self.ssv is not None:
            self.ssv.eval()
        return self


def raw_gf_terms(model, energy, ic, times, selected, collocation=24):
    """One original GF residual per expert; no reference future fields.

The inherited operator includes that expert's gamma in both its latent clock
and the intrinsic derivative gamma*(U_tau - xi*U_xi/2 - U/2).
Sampling remains uniform in each transformed xi window [-24,24], as in the
previous original-size run. Charts share energy parameters, not coordinates.
"""
    terms = []
    for expert, p in zip(model.experts, model.contexts(ic)):
        z, _ = expert.latent_rollout(p, times)
        zz = z[:, selected].reshape(-1, model.latent)
        pp = p[:, None].expand(-1, len(selected), -1).reshape(-1, 17)
        xi = (2 * torch.rand(len(zz), collocation, device=ic.device, dtype=ic.dtype) - 1) * 24.
        lhs, rhs = gradient_rhs(expert, energy, zz, pp, xi)
        terms.append((lhs - rhs).square().mean())
    return torch.stack(terms)
