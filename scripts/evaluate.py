"""Compute every metric on every run in results/ and write the comparison table.

    python scripts/evaluate.py

For each image: luminance and R_lum, ImageReward, held-out CLIP score, mean absolute pixel distance
to the base image with the same prompt and seed. Both rewards are computed on every run (cross-effects).
Writes results/metrics_per_image.csv and results/summary.csv (one row per run: means over 10 prompts x 4 seeds).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from tg.rewards import ImageReward, luminance_map

BASE = "results/none"


def to_tensor(pil):
    return torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div(255)[None].cuda()


class CLIPScore:
    """100 * cos(image embedding, prompt embedding), CLIP ViT-L/14."""

    def __init__(self):
        name = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(name).cuda().eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def __call__(self, pil, prompt):
        inp = self.proc(text=[prompt], images=pil, return_tensors="pt", padding=True).to("cuda")
        img = self.model.get_image_features(pixel_values=inp.pixel_values)
        txt = self.model.get_text_features(input_ids=inp.input_ids, attention_mask=inp.attention_mask)
        return 100 * max(F.cosine_similarity(img, txt).item(), 0.0)


def main():
    runs = sorted(d for d in glob.glob("results/*")
                  if os.path.exists(f"{d}/results.jsonl") and not os.path.basename(d).startswith("test_"))
    ir, clip = ImageReward(), CLIPScore()

    rows = []
    for run in runs:
        cfg = json.load(open(f"{run}/config.json"))
        for line in open(f"{run}/results.jsonl"):
            r = json.loads(line)
            path = f"{run}/images/{r['image']}"
            pil = Image.open(path).convert("RGB")
            y = to_tensor(pil)
            base_path = f"{BASE}/images/{r['image']}"
            with torch.no_grad():
                lum = luminance_map(y).mean().item()
                ir_score = ir(y, r["prompt"]).item()
            rows.append({
                "run": os.path.basename(run),
                "method": cfg["method"],
                "reward": cfg["reward"] if cfg["method"] != "none" else "-",
                "beta": cfg["beta"],
                "pid": r["pid"], "seed": r["seed"],
                "luminance": lum,
                "R_lum": -(lum - cfg["lum_target"]) ** 2,
                "imagereward": ir_score,
                "clip": clip(pil, r["prompt"]),
                "dist_to_base": (y - to_tensor(Image.open(base_path).convert("RGB"))).abs().mean().item()
                if os.path.exists(base_path) else np.nan,
                "time_per_image": r["time_per_image"],
            })
        print(f"evaluated {run}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/metrics_per_image.csv", index=False)
    metrics = ["luminance", "R_lum", "imagereward", "clip", "dist_to_base", "time_per_image"]
    summary = (df.groupby(["run", "method", "reward", "beta"])[metrics].mean().reset_index()
               .sort_values(["reward", "method", "beta"]))
    summary.to_csv("results/summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
