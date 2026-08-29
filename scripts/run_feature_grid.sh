#!/bin/bash
# Tier-2 of the feature-source comparison: the actual cross connection.
# {control, allfix-OOD features, parity features} x 3 seeds, protocol B.
# 3 seeds are mandatory: the effect size at stake (+0.016) sits at the seed
# noise floor (~0.015), so a single-seed verdict would be noise-reading.
#
# Tier-1 prediction, fixed in advance: parity does NOT beat allfix in-model
# (geology utility is flat across a 7x generation-quality range), and may
# hurt at K=0 via its stronger location fingerprint (HUC2 0.454 vs 0.403).
cd /nfs/data/cxs1024/hydroPFN || exit 1
source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src

C="--nc data/CAMELS_Frederik.nc --holdout 01,11,17 --time-aligned
   --retrieval geo --context-pool all --train-end 9000 --eval-start 600
   --causal --geo --k-train 0,0,0,1,2,4,8,16 --k-eval 0,4
   --self-ctx-p 0.4 --score-tail 1 --roll 40 --epochs 160 --steps 300"
C=$(echo $C)

JOBS=()
for S in 0 1 2; do
  JOBS+=("--seed $S --tag FG_ctrl_s$S")
  JOBS+=("--seed $S --dem-feats logs/camels_demfeat13_8.npz --dem-p 0.5 --tag FG_allfix_s$S")
  JOBS+=("--seed $S --dem-feats logs/camels_demfeat13_parity.npz --dem-p 0.5 --tag FG_parity_s$S")
done

# 9 jobs over 5 GPUs, each GPU draining its share sequentially
for G in 0 1 2 3 4; do
  (
    i=$G
    while [ $i -lt ${#JOBS[@]} ]; do
      TAG=$(echo "${JOBS[$i]}" | grep -oE "FG_[a-z]+_s[0-9]")
      CUDA_VISIBLE_DEVICES=$G $PY -m hydropfn.train.train_pub $C ${JOBS[$i]} \
        > "logs/pub_$TAG.log" 2>&1
      i=$((i + 5))
    done
  ) &
done
wait
echo ALL-DONE > logs/feature_grid.done
