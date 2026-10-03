"""Generate the images of one configuration (method x reward x beta) and save them with their rewards.

    python scripts/generate.py --method none --n_prompts 10
    python scripts/generate.py --method dps --reward luminance --beta 1e4 --n_prompts 10

Outputs in results/<name>/: images/<prompt id>_s<seed>.png, results.jsonl (one line per image),
diagnostics.jsonl (per-step guidance logs), config.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from tg.ddim import NoGuidance, sample
from tg.guidance import DPS, LiDAR
from tg.rewards import luminance_map, make_reward
from tg.sd import StableDiffusion


def to_pil(y):
    """One image [3, H, W] in [0, 1] -> PIL."""
    return Image.fromarray((y.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["none", "dps", "lidar"], default="none")
    p.add_argument("--reward", choices=["luminance", "imagereward"], default="luminance")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--lum_target", type=float, default=0.2)
    p.add_argument("--n_lookahead", type=int, default=16, help="LiDAR: lookahead samples per prompt")
    p.add_argument("--lookahead_steps", type=int, default=10, help="LiDAR: DDIM steps of the lookahead sampler")
    p.add_argument("--n_prompts", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    name = "none" if args.method == "none" else f"{args.method}_{args.reward}_b{args.beta:g}"
    if args.method == "lidar":
        name += f"_n{args.n_lookahead}"
    out = args.out or os.path.join("results", name)
    os.makedirs(f"{out}/images", exist_ok=True)
    with open(f"{out}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    prompts = [line.strip() for line in open("prompts.txt") if line.strip()][: args.n_prompts]
    sd = StableDiffusion()
    reward = make_reward(args.reward, lum_target=args.lum_target)

    f_res = open(f"{out}/results.jsonl", "w")
    f_diag = open(f"{out}/diagnostics.jsonl", "w")
    for pid, prompt in enumerate(prompts):
        model = sd.bind(prompt, reward)
        if args.method == "dps":
            guidance = DPS(args.beta)
        elif args.method == "lidar":
            guidance = LiDAR(args.beta, args.n_lookahead, args.lookahead_steps)
        else:
            guidance = NoGuidance()
        # Same x_T per seed for every method -> paired comparisons (base vs guided).
        x_T = torch.cat([
            torch.randn(1, 4, 64, 64, generator=torch.Generator("cuda").manual_seed(s), device="cuda")
            for s in args.seeds
        ])

        torch.cuda.synchronize()
        t0 = time.time()
        x0 = sample(model, x_T, guidance)  # all seeds of a prompt in one batch
        torch.cuda.synchronize()
        dt = time.time() - t0

        with torch.no_grad():
            y = sd.decode(x0)
            r = reward(y, prompt)
            lum = luminance_map(y).mean(dim=(1, 2))

        for i, s in enumerate(args.seeds):
            fname = f"{pid:02d}_s{s}.png"
            to_pil(y[i]).save(f"{out}/images/{fname}")
            f_res.write(json.dumps({
                "pid": pid, "prompt": prompt, "seed": s, "image": fname,
                "reward": r[i].item(), "luminance": lum[i].item(), "time_per_image": dt / len(args.seeds),
            }) + "\n")
        if hasattr(guidance, "log"):
            f_diag.write(json.dumps({"pid": pid, "steps": guidance.log,
                                     "lookahead_R": getattr(guidance, "lookahead_R", None)}) + "\n")
        f_res.flush()
        f_diag.flush()
        print(f"[{pid + 1}/{len(prompts)}] {prompt!r}: {args.reward} = {r.mean():.4f}, "
              f"luminance = {lum.mean():.3f}, {dt / len(args.seeds):.1f} s/image", flush=True)

    f_res.close()
    f_diag.close()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
