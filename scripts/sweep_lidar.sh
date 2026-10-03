#!/bin/sh
# Block 4: LiDAR with the same betas as DPS (same target pi), 10 prompts x 4 seeds.
#   nohup sh scripts/sweep_lidar.sh > sweep_lidar.log 2>&1 &
set -e
N=10
for b in 1e3 3e3 1e4; do
  python scripts/generate.py --method lidar --reward luminance --beta $b --n_prompts $N
done
for b in 3 10 30; do
  python scripts/generate.py --method lidar --reward imagereward --beta $b --n_prompts $N
done
