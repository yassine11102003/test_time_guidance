# Test-time guidance of Stable Diffusion 1.5 with reward gradients

Goal: sample from π(x₀) ∝ p(x₀)·exp(β R(x₀)), where p is SD1.5 (DDIM 50 steps, CFG 7.5), without retraining,
using ∇R. We compare a gradient method (DPS, Tweedie-gradient guidance) with the reference LiDAR, which
only uses reward values.

Every method plugs into the sampler at one place: the noise prediction is replaced by

    ε̃ = ε − σₜ · ∇log hₜ(xₜ)        (score of πₜ = score of pₜ + ∇log hₜ, primer eq. 12)

| Method | Estimate of ∇log hₜ | Uses ∇R |
|---|---|---|
| `none` | 0 (pretrained sampler) | – |
| `dps` | β ∇ₓₜ R(D(x̄₀(xₜ))), x̄₀ the Tweedie estimate (eq. 14, differentiated through UNet + VAE + reward) | yes |
| `lidar` | closed form of eq. 17, from n = 16 lookahead samples (10-step DDIM) per prompt | no |

Rewards: luminance −(ℓ̄ − 0.2)² and ImageReward (re-implemented in torch to be differentiable).
Held-out metric: CLIP score (ViT-L/14) between prompt and image. HPSv2 was left out: its package is
unmaintained and we kept the evaluation to one well-understood held-out metric.

## Code

```
tg/sd.py          SD1.5 wrapper: CFG noise prediction, differentiable VAE decoding
tg/ddim.py        DDIM grid + sampling loop (eq. 4-5, η = 0); the single guidance entry point
tg/rewards.py     luminance, differentiable ImageReward
tg/guidance.py    DPS, LiDAR
scripts/check_setup.py   sanity checks (our DDIM vs diffusers, our ImageReward vs official, gradients)
scripts/generate.py      generates one configuration (method × reward × β)
scripts/evaluate.py      all metrics on all runs -> results/summary.csv
scripts/plots.py         figures -> results/plots/
```

## Setup (one H100 80 GB)

```bash
pip install -r requirements.txt
```
ImageReward is unmaintained and breaks with transformers 5.x, hence the pinned versions in `requirements.txt`.

## Commands that produce each result

| Result | Command |
|---|---|
| Sanity checks | `python scripts/check_setup.py` |
| Base images (10 prompts × seeds 0–3) | `python scripts/generate.py --method none --n_prompts 10` |
| DPS sweep (luminance β ∈ {1e3, 3e3, 1e4}, ImageReward β ∈ {3, 10, 30}) | `sh scripts/sweep_dps.sh` (includes the base run) |
| LiDAR sweep (same β) | `sh scripts/sweep_lidar.sh` |
| One configuration | `python scripts/generate.py --method dps --reward luminance --beta 1e4 --n_prompts 10` |
| Table `results/summary.csv` | `python scripts/evaluate.py` |
| Figures `results/plots/` | `python scripts/plots.py` |

Each run writes `results/<method>_<reward>_b<beta>/` with the images, `results.jsonl` (one line per image)
and `diagnostics.jsonl` (per-step guidance logs: correction norm, R(x̄₀) for DPS, effective sample size for LiDAR).
All methods start from the same x_T for a given seed, so images are paired with the base image.
