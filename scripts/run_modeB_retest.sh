#!/bin/bash
# Re-test of mode B (historical/training-period context) under the CURRENT
# model, prompted by Nicholas reporting that training-data-as-context HELPS.
#
# Our recorded "+0.005, dead" predates the displacement geo-encoding, causal
# masking, and the K-allocation fix -- and used a context offset ~88 days
# OFF-season, so the time-aligned path compared different seasons. Two arms:
#
#   B_mis    modern config, misaligned offset (replicates the old setting)
#   B_align  modern config, same-DOY offsets (whole-year multiples), in
#            training AND eval -- "same season, earlier years"
#
# Both without self-context, to isolate the neighbour-historical effect.
# Same protocol B as every modern table. K=0 column doubles as the control.
cd /nfs/data/cxs1024/hydroPFN || exit 1
until [ -f logs/feature_grid.done ]; do sleep 60; done
source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src

C="--nc data/CAMELS_Frederik.nc --holdout 01,11,17 --time-aligned
   --retrieval geo --context-pool all --train-end 9000 --eval-start 600
   --seed 0 --causal --geo --k-train 0,0,0,1,2,4,8,16 --k-eval 0,1,2,4,8
   --score-tail 1 --roll 40 --epochs 160 --steps 300
   --context-period train"
C=$(echo $C)

CUDA_VISIBLE_DEVICES=0 $PY -m hydropfn.train.train_pub $C \
  --context-train-start 400 --tag B_mis > logs/pub_B_mis.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m hydropfn.train.train_pub $C \
  --ctx-align --tag B_align > logs/pub_B_align.log 2>&1 &
wait
echo DONE > logs/modeB_retest.done
