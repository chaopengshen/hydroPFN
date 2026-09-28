#!/bin/bash
#SBATCH --job-name=hydropfn_window_report
#SBATCH --partition=basic
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=../slurm/report_%j.out
#SBATCH --error=../slurm/report_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/storage/group/cxs1024/default/cxs1024/conda_envs/tdmpc2/bin/python \
    experiments/window_sampling_diagnostics.py --runs logs/camels531 --out ../diagnostics
