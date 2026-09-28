#!/bin/bash
#SBATCH --job-name=hydropfn_window_check
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=v100s
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=../slurm/preflight_%j.out
#SBATCH --error=../slurm/preflight_%j.err
set -euo pipefail
PYTHON=/storage/group/cxs1024/default/cxs1024/conda_envs/tdmpc2/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
"$PYTHON" -c 'import sys; sys.path.insert(0,"scripts"); from run_recovery_stage1 import verify_sources; print(verify_sources())'
"$PYTHON" tests/test_temporal_decoders.py
"$PYTHON" scripts/check_window_sampling_gpu.py --cache ../cache --source ../encoder > ../gpu_preflight.json
