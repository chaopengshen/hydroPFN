#!/bin/bash
# Restore the successful sampler's full recipe in the multi-scale setting.
#
# Step 1 (immediate): re-score the EXISTING ms_res checkpoint with the fixed
# zero-residual forcing. The first residual eval forced raw elevations into a
# residual-valued state at every DDIM step; its numbers (vario_80m 2.62,
# elev_rmse worse than harmonic) measured that bug, not the recipe.
#
# Step 2: train at parity -- residual + EMA(0.999, evaluated) + the original
# hole distribution + valid-region normalisation with a 0.5 m floor + 60k
# steps (~110 passes; the original had ~300 over a single scale). Then score
# best-of-1 and best-of-8.
#
# This is deliberately NOT a one-at-a-time ablation: we are restoring a
# known-good recipe to parity, not attributing credit among its parts. If
# parity is reached, the ablation "parity minus each element" is the follow-up.
cd /nfs/data/cxs1024/hydroPFN || exit 1
source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src

L="/nfs/data/cxs1024/dem_foundation/logs/dem_patches.npz,10,1.28"
L="$L;logs/pretrain_13km.npz,100,12.8"
L="$L;logs/pretrain_51km.npz,400,51.2"

{
  echo "=== ms_res RE-SCORED with zero-residual forcing (bug fix only) ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_ms_res.pt --residual --n 32
} > logs/res_eval_fixed.log 2>&1

{
  CUDA_VISIBLE_DEVICES=4 $PY -m hydropfn.train.train_dem_multiscale \
    --levels "$L" --steps 60000 --residual --orig-masks --valid-norm \
    --tag parity
  echo "=== PARITY, best-of-1 ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_parity.pt --residual --n 32
  echo "=== PARITY, best-of-8 rerank ==="
  CUDA_VISIBLE_DEVICES=4 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_parity.pt --residual --best-of 8 --n 32
} > logs/parity.log 2>&1
