"""Fixed shared gamma and a six-frame temporal-holdout protocol."""
import numpy as np
import torch
from torch import nn

from experiment import loss
from original_scale_model import OriginalScaleModel, ConditionalExpert, MLP

EARLY = (0, 2, 4)
HOLDOUT = (6, 8, 10)
SIX = EARLY+HOLDOUT
FINAL_METHODS = ('latent6', 'fixed1_gf6', 'selected_gf6')


class SharedGammaExpert(ConditionalExpert):
    def __init__(self, gamma):
        super().__init__(width=128, condition_dim=16, center=0., extent=24.)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('Gamma must be positive and finite')
        self.register_buffer('shared_gamma', torch.tensor(float(gamma)))

    def gamma(self, p):
        return self.shared_gamma.expand(len(p))


class SharedGammaModel(OriginalScaleModel):
    def __init__(self, gamma=1., physical=False):
        # Preserve the original constructor's RNG order and parameter names.
        nn.Module.__init__(self)
        self.charts, self.width, self.condition_dim, self.input_dim, self.latent = 1, 128, 16, 192, 16
        self.encoder = MLP(192, 16, 128, 3)
        self.register_buffer('centers', torch.tensor([0.]))
        self.experts = nn.ModuleList([SharedGammaExpert(gamma)])
        if physical:
            self.experts[0].mode = 'physical'
            self.set_scales([[0., 0.]])


def labelled_prediction(model, ic, labelled, x, t, indices):
    if tuple(indices) not in (EARLY, HOLDOUT, SIX):
        raise ValueError('Unknown label schedule')
    if labelled.shape != (len(ic), len(indices), len(x)):
        raise ValueError('Only explicitly selected labelled fields are accepted')
    return model(ic, t[:max(indices)+1], x)[:, list(indices)]


def data_loss(model, ic, labelled, x, t, indices):
    if tuple(indices) not in (EARLY, SIX):
        raise ValueError('Holdout labels cannot be used as a standalone training target')
    return loss(labelled_prediction(model, ic, labelled, x, t, indices), labelled)


def checkpoint_score(model, ic, labelled, x, t, indices):
    # Same per-IC initial-norm validation criterion as previous experiments.
    pred = labelled_prediction(model, ic, labelled, x, t, indices)
    return ((pred-labelled).square().sum(-1)/labelled[:, :1].square().sum(-1).clamp_min(1e-6)).sqrt().mean()


@torch.no_grad()
def holdout_score(model, ic, targets, x, t):
    """Mean physical-space population relative L2 at .75, 1, 1.25 only."""
    pred = labelled_prediction(model, ic, targets, x, t, HOLDOUT)
    error = (pred-targets).double()
    curve = (error.square().sum((0, 2))/targets.double().square().sum((0, 2)).clamp_min(1e-12)).sqrt()
    return float(curve.mean()), curve.cpu().tolist()


def six_frame_metrics(pred, truth, t):
    error = (pred-truth).double()
    curve = (error.square().sum((0, 2))/truth.double().square().sum((0, 2)).clamp_min(1e-12)).sqrt()
    return {'initial': float(curve[0]), 'label_times_error': float(curve[list(SIX)].mean()),
        'observed': float(curve[t <= 1.25].mean()),
        'short': float(curve[(t > 1.25) & (t <= 2)].mean()),
        'long': float(curve[t > 2].mean()), 'extrapolation': float(curve[t > 1.25].mean()),
        'final': float(curve[-1]), 'curve': curve.tolist()}
