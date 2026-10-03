"""Guidance methods. Each one returns eps_tilde = eps - sigma_t * grad log h_t(x_t) (see tg/ddim.py)."""
import torch


class DPS:
    """Tweedie-gradient guidance: log h_t(x_t) ~ beta * R(x0bar(x_t))  (primer eq. 14), differentiated."""

    def __init__(self, beta):
        self.beta = beta
        self.log = []  # per-step diagnostics

    def setup(self, model, x_T):
        self.log = []

    def eps(self, model, x, st):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            eps = model.eps(x, st.t)                               # 1. noise prediction (UNet)
            x0 = (x - st.sigma_t * eps) / st.alpha_t               # 2. Tweedie estimate
            r = model.reward(x0)                                   # 3. R(D(x0bar))
            grad = torch.autograd.grad(self.beta * r.sum(), x)[0]  # 4. beta * dR/dx_t, through UNet + VAE

        correction = st.sigma_t * grad
        self.log.append({
            "t": st.t,
            "R_x0bar": r.detach().mean().item(),
            "correction_norm": correction.flatten(1).norm(dim=1).mean().item(),
            "eps_norm": eps.detach().flatten(1).norm(dim=1).mean().item(),
        })
        return eps.detach() - correction                           # 5. eps_tilde
