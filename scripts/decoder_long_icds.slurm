#!/bin/bash
#SBATCH --job-name=hydropfn_decoder300
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=v100s
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=../slurm/decoder300_%A_%a.out
#SBATCH --error=../slurm/decoder300_%A_%a.err

# Submit from the immutable source directory with sbatch --chdir="$PWD".
set -euo pipefail
ROOT="$(pwd)"
PYTHON=/storage/group/cxs1024/default/cxs1024/conda_envs/tdmpc2/bin/python
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
# CUDA_VISIBLE_DEVICES is assigned by Slurm, never overridden here.
case "$SLURM_ARRAY_TASK_ID" in
    0) decoder=lstm; inputs=hybrid ;;
    1) decoder=tcn; inputs=hybrid ;;
    2) decoder=patch; inputs=latent ;;
    *) exit 2 ;;
esac
tag="decoder_${decoder}_${inputs}_s0_e300_icds"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
"$PYTHON" -c 'import sys; sys.path.insert(0,"scripts"); from run_recovery_stage1 import verify_sources; print("source_manifest_sha256",verify_sources()); import torch; assert torch.cuda.is_available(), "GPU required"; print("torch",torch.__version__,"GPU",torch.cuda.get_device_name())'
"$PYTHON" tests/test_temporal_decoders.py
"$PYTHON" -u experiments/camels531_decoders.py train \
    --cache "$ROOT/../cache" --tag "$tag" --decoder "$decoder" --inputs "$inputs" \
    --seed 0 --epochs 300 --checkpoint-epochs 100,200,300 --evaluate-checkpoints --sampling fixed
