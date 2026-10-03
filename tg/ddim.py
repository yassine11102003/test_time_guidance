"""DDIM sampler (primer eq. 4-5, eta = 0) with a single entry point for guidance.

A guidance method is any object with
    setup(model, x_T)    called once before the loop (e.g. LiDAR lookahead samples)
    eps(model, x, st)    returns eps_tilde = eps - sigma_t * grad log h_t(x_t)
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class Step:
    t: int  # current timestep
    s: int  # next (lower) timestep, negative on the last step
    alpha_t: float
    sigma_t: float
    alpha_s: float
    sigma_s: float


def ddim_schedule(alphas_cumprod, final_alpha_cumprod, num_steps=50, num_train_steps=1000, steps_offset=1):
    """'leading' spacing as in diffusers for SD1.5: t = 981, 961, ..., 1."""
    step = num_train_steps // num_steps
    ts = [i * step + steps_offset for i in reversed(range(num_steps))]
    sched = []
    for t in ts:
        s = t - step
        ab_t = float(alphas_cumprod[t])
        ab_s = float(alphas_cumprod[s]) if s >= 0 else float(final_alpha_cumprod)
        sched.append(Step(t, s, math.sqrt(ab_t), math.sqrt(1 - ab_t), math.sqrt(ab_s), math.sqrt(1 - ab_s)))
    return sched


class NoGuidance:
    """Pretrained sampler: grad log h = 0, so eps_tilde = eps."""

    def setup(self, model, x_T):
        pass

    @torch.no_grad()
    def eps(self, model, x, st):
        return model.eps(x, st.t)


def sample(model, x_T, guidance=None, num_steps=50):
    """Deterministic DDIM from x_T; returns the final latent x_0."""
    guidance = guidance or NoGuidance()
    x = x_T.float()
    guidance.setup(model, x)
    for st in ddim_schedule(model.alphas_cumprod, model.final_alpha_cumprod, num_steps):
        eps = guidance.eps(model, x, st)                  # eps_tilde: the only method-specific line
        x0 = (x - st.sigma_t * eps) / st.alpha_t          # eq. (4), Tweedie estimate
        x = (st.alpha_s * x0 + st.sigma_s * eps).detach()  # eq. (5) with eta = 0
    return x
