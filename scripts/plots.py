"""Figures for the report, from results/summary.csv (run scripts/evaluate.py first).

    python scripts/plots.py

results/plots/<reward>_curves.png : (a) guided quantity vs beta, (b) reward vs CLIP score (trade-off)
results/plots/<reward>_grid.png   : base vs DPS vs LiDAR at each beta, same seed
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

OUT = "results/plots"
COLORS = {"dps": "#2a78d6", "lidar": "#eb6834"}  # categorical slots 1-2
NAMES = {"dps": "DPS", "lidar": "LiDAR"}
BASE_COLOR, INK, MUTED = "#898781", "#0b0b0b", "#52514e"
# What each reward's panel (a) shows: the quantity the reader can interpret.
QUANTITY = {"luminance": ("luminance", "mean luminance (target 0.2)"), "imagereward": ("imagereward", "ImageReward")}
REWARD_COL = {"luminance": "R_lum", "imagereward": "imagereward"}
GRID_PROMPTS, GRID_SEED = [0, 4, 9], 0

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "font.size": 10})


def curves(summary, reward):
    base = summary[summary.method == "none"].iloc[0]
    sub = summary[summary.reward == reward]
    col, label = QUANTITY[reward]
    rcol = REWARD_COL[reward]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method, g in sub.groupby("method"):
        g = g.sort_values("beta")
        kw = dict(color=COLORS[method], lw=2, marker="o", ms=7, label=NAMES[method])
        ax1.plot(g.beta, g[col], **kw)
        ax2.plot(g[rcol], g["clip"], **kw)
        for _, row in g.iterrows():
            ax2.annotate(f"β={row.beta:g}", (row[rcol], row["clip"]), textcoords="offset points", xytext=(5, 4),
                         fontsize=8, color=MUTED)
    ax1.axhline(base[col], color=BASE_COLOR, ls="--", lw=1.5, label="base (no guidance)")
    if reward == "luminance":
        ax1.axhline(0.2, color=INK, ls=":", lw=1, label="target ℓ* = 0.2")
    ax1.set(xscale="log", xlabel="β", ylabel=label, title="(a) effect of the guidance strength")
    betas = sorted(sub.beta.unique())
    ax1.set_xticks(betas, [f"{b:g}" for b in betas])
    ax1.minorticks_off()
    ax2.scatter([base[rcol]], [base["clip"]], color=BASE_COLOR, s=60, zorder=3, label="base")
    ax2.set(xlabel=f"reward ({rcol})", ylabel="CLIP score (held-out)", title="(b) reward vs prompt fidelity")
    for ax in (ax1, ax2):
        ax.legend(frameon=False)
    fig.suptitle(f"Guidance with the {reward} reward (10 prompts × 4 seeds)", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{reward}_curves.png", dpi=150)
    plt.close(fig)


def grid(summary, reward):
    sub = summary[summary.reward == reward].sort_values(["method", "beta"])
    cols = [("base", "none")] + [(f"{NAMES[r.method]} β={r.beta:g}", r.run) for r in sub.itertuples()]
    fig, axes = plt.subplots(len(GRID_PROMPTS), len(cols), figsize=(1.8 * len(cols), 1.9 * len(GRID_PROMPTS)))
    for i, pid in enumerate(GRID_PROMPTS):
        for j, (title, run) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(Image.open(f"results/{run}/images/{pid:02d}_s{GRID_SEED}.png"))
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=8, color=INK)
    fig.suptitle(f"{reward} guidance: same prompt and seed in each row", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{reward}_grid.png", dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = pd.read_csv("results/summary.csv")
    for reward in ("luminance", "imagereward"):
        if (summary.reward == reward).any():
            curves(summary, reward)
            grid(summary, reward)
    print(f"figures saved to {OUT}/")


if __name__ == "__main__":
    main()
