#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/data" "$ROOT/outputs"
"$PYTHON" "$ROOT/scripts/generate_data.py" --samples 48 --nx 128 --nt 41 --tmax 2 --out "$ROOT/data/single_gaussian_heat.npz"
for mode in physical physical_energy learned_ssv learned_ssv_energy; do
  "$PYTHON" "$ROOT/scripts/train.py" --data "$ROOT/data/single_gaussian_heat.npz" --mode "$mode" \
    --num_train 4 --train_pool 32 --num_valid 8 --num_test 8 --seed 0 --epochs 120 --decoder_modes 17 --collocation 8 --device "${DEVICE:-cpu}" \
    --out_dir "$ROOT/outputs/$mode/n4/seed0"
done
