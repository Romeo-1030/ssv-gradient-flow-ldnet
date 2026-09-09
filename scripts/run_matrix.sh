#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEVICE=${DEVICE:-cuda:0}
mkdir -p "$ROOT/data" "$ROOT/outputs"
if [[ ! -f "$ROOT/data/single_gaussian_heat.npz" ]]; then
  "$PYTHON" "$ROOT/scripts/generate_data.py" --out "$ROOT/data/single_gaussian_heat.npz"
fi
modes=(physical physical_energy learned_ssv learned_ssv_energy hand_ssv hand_ssv_energy)
sizes=(2 4 8 16 32)
for mode in "${modes[@]}"; do
  for n in "${sizes[@]}"; do
    for seed in 0 1 2 3 4; do
      bash "$ROOT/scripts/run_one.sh" "$mode" "$n" "$seed" "$DEVICE"
    done
  done
done
cd "$ROOT"
"$PYTHON" scripts/summarize.py
"$PYTHON" scripts/plot_final.py
