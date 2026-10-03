"""Sanity checks on GPU (block 2):
  1. our DDIM loop reproduces the diffusers pipeline (same prompt, same x_T)
  2. our differentiable ImageReward matches the official (PIL-based) score
  3. reward gradients reach the latent through the VAE (finite, non-zero)
Saves the generated image to outputs/check/.

    python scripts/check_setup.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from tg.ddim import sample
from tg.rewards import ImageReward, LuminanceReward, luminance_map
from tg.sd import StableDiffusion

PROMPT = "a photo of a cat"
SEED = 0
OUT = "outputs/check"


def to_pil(y):
    return Image.fromarray((y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


def main():
    os.makedirs(OUT, exist_ok=True)
    sd = StableDiffusion()
    lum, ir = LuminanceReward(0.2), ImageReward()
    model = sd.bind(PROMPT, reward=lum)
    x_T = torch.randn(1, 4, 64, 64, generator=torch.Generator("cuda").manual_seed(SEED), device="cuda")

    # 1. our loop vs diffusers
    x_ours = sample(model, x_T)
    sd.pipe.scheduler = sd.scheduler
    x_ref = sd.pipe(PROMPT, latents=x_T.to(sd.unet_dtype), num_inference_steps=50, guidance_scale=7.5,
                    eta=0.0, output_type="latent").images.float()
    diff = (x_ours - x_ref).abs()
    print(f"[1] DDIM vs diffusers: max |diff| = {diff.max():.4f}, mean |diff| = {diff.mean():.5f}, "
          f"latent std = {x_ref.std():.3f}  (bf16 -> expect mean |diff| << std)")

    with torch.no_grad():
        y_ours, y_ref = sd.decode(x_ours), sd.decode(x_ref)
    print(f"    image mean |diff| = {(y_ours - y_ref).abs().mean():.4f} (pixel scale [0, 1])")
    to_pil(y_ours).save(f"{OUT}/cat_seed0_ours.png")
    to_pil(y_ref).save(f"{OUT}/cat_seed0_diffusers.png")

    # 2. differentiable ImageReward vs official
    with torch.no_grad():
        r_ours = ir(y_ours, PROMPT).item()
    r_official = ir.model.score(PROMPT, to_pil(y_ours))
    print(f"[2] ImageReward: ours = {r_ours:.4f}, official = {float(r_official):.4f}")

    # 3. gradients through the VAE
    for name, fn in [("luminance", lum), ("imagereward", ir)]:
        x0 = x_ours.detach().requires_grad_(True)
        r = fn(sd.decode(x0), PROMPT)
        (g,) = torch.autograd.grad(r.sum(), x0)
        print(f"[3] {name}: R = {r.item():.4f}, |grad R wrt x0| = {g.norm():.3e}, finite = {torch.isfinite(g).all().item()}")

    print(f"    mean luminance of the base image = {luminance_map(y_ours).mean().item():.3f}")
    print(f"    images saved to {OUT}/")


if __name__ == "__main__":
    main()
