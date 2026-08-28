#!/bin/bash
# The FULL recipe, all six recovered elements: residual + EMA(evaluated) +
# original hole distribution + valid-region norm (0.5 m floor) + 60k steps +
# V-PREDICTION. The concurrently-training eps run is the same thing minus v,
# so the pair is a clean one-element ablation of the parameterisation.
#
# Scored on the ORIGINAL hole distribution -- the 0.810 anchor's protocol.
cd /nfs/data/cxs1024/hydroPFN || exit 1
source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src

L="/nfs/data/cxs1024/dem_foundation/logs/dem_patches.npz,10,1.28"
L="$L;logs/pretrain_13km.npz,100,12.8"
L="$L;logs/pretrain_51km.npz,400,51.2"

{
  CUDA_VISIBLE_DEVICES=3 $PY -m hydropfn.train.train_dem_multiscale \
    --levels "$L" --steps 60000 --residual --orig-masks --valid-norm \
    --param v --tag parity_v
  echo "=== PARITY-V, orig holes, best-of-1 ==="
  CUDA_VISIBLE_DEVICES=3 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_parity_v.pt --residual --holes orig --n 32
  echo "=== PARITY-V, orig holes, best-of-8 ==="
  CUDA_VISIBLE_DEVICES=3 $PY experiments/eval_dem_texture.py \
    --levels "$L" --ckpt logs/dem_ms_parity_v.pt --residual --best-of 8 \
    --holes orig --n 32
} > logs/parity_v.log 2>&1
