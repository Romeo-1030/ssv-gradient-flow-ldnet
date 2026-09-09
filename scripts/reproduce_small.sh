#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/data" "$ROOT/outputs"
python "$ROOT/scripts/generate_data.py" --samples 160 --tmax 10 --out "$ROOT/data/single_gaussian_heat.npz"
for mode in physical physical_energy learned_ssv learned_ssv_energy; do
  python "$ROOT/scripts/train.py" --data "$ROOT/data/single_gaussian_heat.npz" --mode "$mode" \
    --num_train 4 --seed 0 --epochs 200 --decoder_modes 33 --device "${DEVICE:-cpu}" \
    --out_dir "$ROOT/outputs/$mode/n4/seed0"
done
