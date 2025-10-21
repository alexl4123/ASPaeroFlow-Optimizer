#!/usr/bin/env bash
#SBATCH -J ASPaeroFlow-Optimizer
#SBATCH -p any                 # or: broadwell / sunnycove / gpu
#SBATCH -t 00:02:00            # 2 minutes
#SBATCH -N 1 -n 1              # 1 node, 1 task
#SBATCH --cpus-per-task=1
#SBATCH --mem=22G
#SBATCH -o slurm-%j.out

# Optional: pick a specific node if you must
# #SBATCH -w coppernode21

# If you use modules or conda, activate here
# module load python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv

# Activate potassco env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco

# 1) use personal gurobi license
export GRB_LICENSE_FILE="$HOME/gurobi_wls.lic"

./start_script.sh

