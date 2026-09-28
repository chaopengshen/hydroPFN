#!/bin/bash
#SBATCH --job-name=hydropfn_residual
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=v100s
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=../slurm/residual_%A_%a.out
#SBATCH --error=../slurm/residual_%A_%a.err
set -euo pipefail
PYTHON=/storage/group/cxs1024/default/cxs1024/conda_envs/tdmpc2/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
"$PYTHON" -c 'import sys; sys.path.insert(0,"scripts"); from run_recovery_stage1 import verify_sources; print(verify_sources())'
"$PYTHON" tests/test_residual_correction.py
if [ "${PREFLIGHT:-0}" = "1" ]; then
    "$PYTHON" -u experiments/specialist_residual.py --cache ../cache --source ../encoder \
        --out ../smoke --raw-epochs 1 --correction-epochs 1 --steps 2 --batch 4
else
    "$PYTHON" -u experiments/specialist_residual.py --cache ../cache --source ../encoder \
        --out "../runs/seed${SLURM_ARRAY_TASK_ID}" --seed "$SLURM_ARRAY_TASK_ID"
fi
