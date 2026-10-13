"""Guidance methods. Each one returns eps_tilde = eps - sigma_t * grad log h_t(x_t) (see tg/ddim.py)."""
import torch

from tg.ddim import NoGuidance, sample


class DPS:
    """Tweedie-gradient guidance: log h_t(x_t) ~ beta * R(x0bar(x_t)), differentiated (DPS-style, paper Eq. 5)."""

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


class LiDAR:
    """Lookahead-sample guidance (Kim et al. 2026; paper Eq. 6-7). Uses reward values only, never grad R."""

    def __init__(self, beta, n_lookahead=16, lookahead_steps=10, seed=1234):
        self.beta, self.n, self.steps, self.seed = beta, n_lookahead, lookahead_steps, seed
        self.log = []

    @torch.no_grad()
    def setup(self, model, x_T):
        """Once per prompt: n cheap samples of p and their rewards, shared by all seeds."""
        self.log = []
        g = torch.Generator(x_T.device).manual_seed(self.seed)
        z = torch.randn((self.n,) + x_T.shape[1:], generator=g, device=x_T.device)
        xhat = sample(model, z, NoGuidance(), num_steps=self.steps)
        r = model.reward(xhat)
        self.xhat = xhat.flatten(1)  # [n, D]
        self.logr = self.beta * r    # [n]
        self.lookahead_R = r.tolist()

    @torch.no_grad()
    def eps(self, model, x, st):
        eps = model.eps(x, st.t)
        d2 = ((x.flatten(1)[:, None, :] - st.alpha_t * self.xhat[None]) ** 2).sum(-1)  # [B, n]
        logw = -d2 / (2 * st.sigma_t ** 2)
        w = torch.softmax(logw, dim=1)                       # paper Eq. 6: plausibility of each x_hat as origin of x_t
        w_r = torch.softmax(logw + self.logr[None], dim=1)  # same weights, tilted by exp(beta R_i)
        grad = st.alpha_t / st.sigma_t ** 2 * ((w_r - w) @ self.xhat)  # paper Eq. 7
        correction = st.sigma_t * grad.view_as(x)

        self.log.append({
            "t": st.t,
            "ess": (1.0 / (w ** 2).sum(1)).mean().item(),  # effective number of lookahead samples
            "correction_norm": correction.flatten(1).norm(dim=1).mean().item(),
            "eps_norm": eps.flatten(1).norm(dim=1).mean().item(),
        })
        return eps - correction
