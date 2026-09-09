#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?mode}; N=${2:?number of training trajectories}; SEED=${3:?seed}; DEVICE=${4:-cuda:0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=$ROOT/data/single_gaussian_heat.npz
OUT=$ROOT/outputs/$MODE/n$N/seed$SEED
mkdir -p "$OUT"
python "$ROOT/scripts/train.py" --data "$DATA" --mode "$MODE" --num_train "$N" \
  --seed "$SEED" --epochs 1000 --decoder_modes 33 --device "$DEVICE" --out_dir "$OUT"
