#!/bin/bash
#SBATCH --job-name=hydropfn_daily300
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=v100s
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-5%4
#SBATCH --output=../slurm/daily300_%A_%a.out
#SBATCH --error=../slurm/daily300_%A_%a.err
set -euo pipefail
PYTHON=/storage/group/cxs1024/default/cxs1024/conda_envs/tdmpc2/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
seed=$((SLURM_ARRAY_TASK_ID / 2))
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then decoder=lstm; else decoder=tcn; fi
"$PYTHON" -c 'import sys; sys.path.insert(0,"scripts"); from run_recovery_stage1 import verify_sources; print(verify_sources()); import torch; assert torch.cuda.is_available()'
"$PYTHON" -u experiments/camels531_decoders.py train \
    --cache ../cache --source ../encoder --tag "benchmark_${decoder}_daily_s${seed}" \
    --decoder "$decoder" --inputs hybrid --sampling daily --feature-mode online \
    --seed "$seed" --epochs 300 --checkpoint-epochs 100,200,300 --evaluate-checkpoints
