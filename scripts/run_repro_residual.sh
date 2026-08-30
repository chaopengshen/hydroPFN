#!/bin/bash
# Re-run residual.pt (the probable 0.810 champion) in ITS OWN harness.
# The first attempt lost $PY to nested-heredoc escaping and ran nothing.
cd /nfs/data/cxs1024/dem_foundation || exit 1
source gpuenv.sh
CUDA_VISIBLE_DEVICES=3 $PY src/tests/test_diffusion_sampler.py \
  --ckpt logs/residual.pt --skip-figure --n-eval 40 \
  > logs/anchor_repro_residual.log 2>&1
