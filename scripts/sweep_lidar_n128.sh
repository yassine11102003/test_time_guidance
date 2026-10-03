#!/bin/sh
# Compute-matched LiDAR: DPS costs ~2.5 s/image more than LiDAR(n=16), which pays for ~100 extra lookahead
# samples per prompt. Same best DPS betas, n = 128.
#   nohup sh scripts/sweep_lidar_n128.sh > sweep_lidar_n128.log 2>&1 &
set -e
python scripts/generate.py --method lidar --reward luminance --beta 1e3 --n_lookahead 128 --n_prompts 10
python scripts/generate.py --method lidar --reward imagereward --beta 10 --n_lookahead 128 --n_prompts 10
