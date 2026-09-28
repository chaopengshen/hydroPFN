#!/bin/bash
#SBATCH --job-name=hydropfn_windows
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=v100s
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-9%4
#SBATCH --output=../slurm/windows_%A_%a.out
#SBATCH --error=../slurm/windows_%A_%a.err
set -euo pipefail
ROOT="$(pwd)"
PYTHON=/storage/group/cxs1024/default/cxs1024/conda_envs/tdmpc2/bin/python
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
seed=$((SLURM_ARRAY_TASK_ID / 5))
model=$((SLURM_ARRAY_TASK_ID % 5))
case "$model" in
    0) decoder=lstm; inputs=raw ;;
    1) decoder=tcn; inputs=raw ;;
    2) decoder=lstm; inputs=hybrid ;;
    3) decoder=tcn; inputs=hybrid ;;
    4) decoder=patch; inputs=latent ;;
esac
"$PYTHON" -c 'import sys; sys.path.insert(0,"scripts"); from run_recovery_stage1 import verify_sources; print("source_manifest_sha256",verify_sources()); import torch; assert torch.cuda.is_available(); print("GPU",torch.cuda.get_device_name())'
# Same allocated GPU and runtime within every fixed/daily pair.
if [ "$seed" -eq 0 ]; then modes="fixed daily"; else modes="daily fixed"; fi
for sampling in $modes; do
    tag="window_${decoder}_${inputs}_${sampling}_s${seed}"
    "$PYTHON" -u experiments/camels531_decoders.py train \
        --cache "$ROOT/../cache" --source "$ROOT/../encoder" \
        --tag "$tag" --decoder "$decoder" --inputs "$inputs" --sampling "$sampling" \
        --feature-mode online \
        --seed "$seed" --epochs 100 --checkpoint-epochs 100 --evaluate-checkpoints
done
