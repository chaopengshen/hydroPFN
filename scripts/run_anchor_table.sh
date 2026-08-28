#!/bin/bash
# One table, one protocol: everything scored on the ORIGINAL hole
# distribution, the one the 0.810 anchor was measured on.
#
# Rows:
#   1. allfix.pt        -- the ORIGINAL successful single-scale sampler, via
#                          the new harness (EMA weights, residual sampling,
#                          valid-region norm, best-of-8). Fine level only:
#                          it never saw coarse patches. If this row does not
#                          land near psd 0.8, the HARNESS is wrong, not the
#                          models -- it re-anchors the measuring stick before
#                          any verdict on the new runs.
#   2. ms1 baseline     -- multi-scale, no recipe.
#   3. parity           -- multi-scale, full recipe, best-of-1 and best-of-8.
#
# Waits for the earlier run_parity.sh chain (training + its mixed-holes
# evals) to finish so the GPU is free; those mixed-holes evals stay useful as
# the parity-vs-baseline comparison on the HARD protocol.
cd /nfs/data/cxs1024/hydroPFN || exit 1

until [ -f logs/dem_ms_parity.pt ]; do sleep 120; done
while pgrep -f run_parity.sh > /dev/null; do sleep 60; done

source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src

LF="/nfs/data/cxs1024/dem_foundation/logs/dem_patches.npz,10,1.28"
L="$LF;logs/pretrain_13km.npz,100,12.8"
L="$L;logs/pretrain_51km.npz,400,51.2"

{
  echo "=== ANCHOR: allfix.pt (original single-scale), orig holes, bo8 ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$LF" --ckpt /nfs/data/cxs1024/dem_foundation/logs/allfix.pt \
    --residual --force-valid-norm --best-of 8 --holes orig --n 32
  echo
  echo "=== ms1 baseline (no recipe), orig holes ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_ms1.pt --holes orig --n 32
  echo
  echo "=== PARITY, orig holes, best-of-1 ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_parity.pt --residual --holes orig --n 32
  echo
  echo "=== PARITY, orig holes, best-of-8 ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_parity.pt --residual --best-of 8 \
    --holes orig --n 32
} > logs/anchor_table.log 2>&1
