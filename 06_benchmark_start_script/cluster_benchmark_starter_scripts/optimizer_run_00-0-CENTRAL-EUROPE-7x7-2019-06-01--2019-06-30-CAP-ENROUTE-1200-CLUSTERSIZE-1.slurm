#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -t 720:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out     # %x = job name, %j = job id

set -euo pipefail

TIME_GRANULARITY=60
FOLDER=20260109
INSTANCE=00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1
PROBLEM=${INSTANCE}

echo "Job: $SLURM_JOB_NAME  |  ID: $SLURM_JOB_ID"
echo "Instance: $INSTANCE   |  Folder: $FOLDER  |  TG: $TIME_GRANULARITY"

# If you use modules or conda, activate here
# module load python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv

# Activate potassco env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco

# 1) use personal gurobi license
export GRB_LICENSE_FILE="$HOME/gurobi_wls.lic"

mkdir -p output/${FOLDER}
mkdir -p logs/${FOLDER}

./start_benchmark_caller.py ../05_instances/${PROBLEM} --output-root=output --output-dir=${FOLDER}/output_${INSTANCE}/ --timestep-granularity=${TIME_GRANULARITY}  --memory-limit=35 --time-limit=600 --experiment-name=${INSTANCE} --scaling-experiments=0 --hot-start  --experiment-asp-r-d-s=0 --experiment-asp-r-d-ns=0 --experiment-asp-r-nd-s=0 --experiment-asp-nr-d-s=0 --experiment-asp-r-nd-ns=0 --experiment-asp-nr-nd-s=0 --experiment-asp-nr-d-ns=0 --experiment-asp-nr-nd-ns=0 --experiment-mip=0 --experiment-asp-r-d-sp=0 --experiment-asp-nr-d-sp=0 --experiment-asp-r-nd-sp=0 --experiment-asp-nr-nd-sp=0


