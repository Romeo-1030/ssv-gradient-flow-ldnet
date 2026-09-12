#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
group="${2:?supply method name}"
root="outputs/ssv_gamma_frozen_ldnet_v5_final3000_cosine"
python_bin="${PYTHON:-python}"
trainer="scripts/train.py"
data="data/single_gaussian_heat_fixed_domain_v1.npz"
scale="artifacts/pretrained/scale.pt"

case "$group" in
  physical) mode=physical; extra=() ;;
  learned_ssv) mode=learned_ssv; extra=(--scale_checkpoint "$scale") ;;
  learned_ssv_energy) mode=learned_ssv_energy; extra=(--scale_checkpoint "$scale") ;;
  oracle_ssv) mode=hand_ssv; extra=(--oracle_gamma) ;;
  oracle_ssv_energy) mode=hand_ssv_energy; extra=(--oracle_gamma) ;;
  *) echo "unknown group: $group" >&2; exit 2 ;;
esac

mkdir -p "$root/logs"
for seed in 0 1 2 3 4; do
  out="$root/$group/n32/seed$seed"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$trainer" \
    --data "$data" --out_dir "$out" --mode "$mode" --seed "$seed" \
    --num_train 32 --epochs 3000 --decoder_modes 33 \
    --lambda_gf 1e-4 --gf_tmax 10 --gf_time_points 32 \
    --benchmark fixed_domain --lr 1e-3 --eta_min 1e-5 --device cuda:0 \
    "${extra[@]}" >"$root/logs/${group}_seed${seed}.log" 2>&1
done
