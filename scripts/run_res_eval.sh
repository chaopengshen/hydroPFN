#!/bin/bash
# One-at-a-time ablation of the two changes that took the ORIGINAL single-scale
# sampler to psd 0.810: residual parameterisation, then best-of-K reranking.
#
# The record's own lesson is why they are separated: "a single stacked
# 'all improvements' run would have shown a modest gain over baseline and all
# three changes would have shipped -- when two of them are actively harmful on
# top of the third."
#
# Baseline for comparison (no residual, best-of-1), same protocol:
#   1.28 km  psd 0.278  vario10 0.445  vario80 0.665
#   12.8 km  psd 0.032  vario10 0.186  vario80 0.329
#   51.2 km  psd 0.020  vario10 0.181  vario80 0.277
cd /nfs/data/cxs1024/hydroPFN || exit 1

until [ -f logs/dem_ms_ms_res.pt ]; do sleep 60; done
sleep 15

source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src

L="/nfs/data/cxs1024/dem_foundation/logs/dem_patches.npz,10,1.28"
L="$L;logs/pretrain_13km.npz,100,12.8"
L="$L;logs/pretrain_51km.npz,400,51.2"

{
  echo "=== +RESIDUAL only (best-of-1) ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_ms_res.pt --residual --n 32
  echo
  echo "=== +RESIDUAL +BEST-OF-8 rerank ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_ms_res.pt --residual --best-of 8 --n 32
} > logs/res_eval.log 2>&1
