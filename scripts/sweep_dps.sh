#!/bin/sh
# Block 3: base + DPS beta sweep on 10 prompts x 4 seeds.
#   nohup sh scripts/sweep_dps.sh > sweep.log 2>&1 &
set -e
N=10
python scripts/generate.py --method none --n_prompts $N
for b in 1e3 3e3 1e4; do
  python scripts/generate.py --method dps --reward luminance --beta $b --n_prompts $N
done
for b in 3 10 30; do
  python scripts/generate.py --method dps --reward imagereward --beta $b --n_prompts $N
done
