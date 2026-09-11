#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -t 720:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -J v2bench
#SBATCH -o logs/%x-%A_%a.out
#
# Run the whole V2 benchmark as ONE job array -- one array task per problem directory
# (region x time granularity x capacity level).
#
#   sbatch --array=1-$(( $(wc -l < ../05_instances/problems.tsv) - 1 ))%40 run_all_benchmarks.slurm
#
# or simply, once you have read the array range the expansion script printed:
#
#   sbatch --array=1-325%40 run_all_benchmarks.slurm
#
# %40 caps concurrency at 40 tasks; raise or lower to suit the queue.
#
# Prerequisite: ../05_instances has been populated by
#   ASPaeroFlow-DataGenerator/expand_instances_for_benchmark.sh
# which also writes ../05_instances/problems.tsv, the index this array reads.

set -euo pipefail

FOLDER="${FOLDER:-$(date +%Y%m%d)_V2}"       # output namespace; override with FOLDER=... sbatch
INSTANCE_ROOT="../05_instances"
MANIFEST="${INSTANCE_ROOT}/problems.tsv"
TIME_LIMIT="${TIME_LIMIT:-600}"              # seconds per solver run
MEMORY_LIMIT="${MEMORY_LIMIT:-35}"           # GiB, must stay below --mem above
WANDB_ENABLED="${WANDB_ENABLED:-False}"      # see the note below before turning this on
RESULTS_FORMAT="${RESULTS_FORMAT:-csv.gz}"   # uncompressed would be ~2.9 TB across the full grid

[ -f "$MANIFEST" ] || { echo "[ERROR] $MANIFEST not found -- run the expansion script first" >&2; exit 1; }

# ---- pick this task's problem out of the manifest (line 1 is the header) -------------------
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
IFS=$'\t' read -r PROBLEM TIME_GRANULARITY REGION CAP_LEVEL N_INSTANCES < <(sed -n "${LINE}p" "$MANIFEST")
[ -n "${PROBLEM:-}" ] || { echo "[ERROR] no manifest row at line $LINE" >&2; exit 1; }

echo "Job: ${SLURM_JOB_NAME}  |  array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Problem:   $PROBLEM"
echo "Region:    $REGION"
echo "Capacity:  $CAP_LEVEL"
echo "TG:        $TIME_GRANULARITY   ($N_INSTANCES instances)"
echo "Output:    output/${FOLDER}/output_${PROBLEM}/"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco
export GRB_LICENSE_FILE="$HOME/gurobi_wls.lic"

# Weights & Biases is OFF for this run, deliberately. With ~76,000 solver runs across the full
# grid it would mean 76,000 logged W&B runs, network traffic from every compute node, and an
# unusable workspace. It is also a hard failure mode rather than a soft one:
# 01_ASPaeroFlow/main.py raises FileNotFoundError when wandb.key is absent, before wandb is
# ever consulted, so WANDB_MODE=disabled does NOT protect you -- the flag below does.
# The CSVs under output/ are the authoritative results either way.
# To turn it on:  WANDB_ENABLED=True sbatch ... (and make sure wandb.key exists on the nodes)

mkdir -p "output/${FOLDER}" "logs"

# ---- provenance: pin exactly what produced these numbers -----------------------------------
# The optimizer's capacity semantics and arrival-delay metric changed in 2026-09. Results from
# before that change are NOT comparable with these, so record the commit in the output tree.
{
  echo "problem=${PROBLEM}"
  echo "region=${REGION}"
  echo "capacity_level=${CAP_LEVEL}"
  echo "time_granularity=${TIME_GRANULARITY}"
  echo "time_limit_s=${TIME_LIMIT}"
  echo "memory_limit_gib=${MEMORY_LIMIT}"
  echo "optimizer_commit=$(git -C .. rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "optimizer_dirty=$(test -n "$(git -C .. status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "slurm_job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "host=$(hostname)"
  echo "started=$(date -Is)"
} > "output/${FOLDER}/run_provenance_${PROBLEM}.txt"

# ---- the experiment selection ------------------------------------------------------------
# Every flag defaults to 1 in start_benchmark_caller.py, so what is listed here is what gets
# switched OFF.
#
# --experiment-all-asp-variants expands to 27 exact-ASP configurations (ground-delay x rerouting
# x dynamic-sectorisation, each in {none, partial, full}), all running 02_ASP/main.py -- the
# EXACT solver. The full 27 belong on the small-scaling family, which exists precisely so exact
# methods can close instances; on a 31,622-flight TG=60 instance they would time out essentially
# every time, at 600 seconds per run of predictable non-information.
#
# TWO of them are kept on the large family as well, because they are the best-performing exact
# configurations and are worth having a large-scale profile for:
#     05_ASP_rp_dp_sp  =  ASP(r_p, d_p, s_p)   partial rerouting, partial delay, partial DAC
#     05_ASP_rp_d_sp   =  ASP(r_p, d,   s_p)   partial rerouting, full delay,    partial DAC
#
# The manifest distinguishes the two families: small-scaling rows carry capacity_level=NONE
# because that family has no PCAP sweep (cap-enroute is already 1).
if [ "$CAP_LEVEL" = "NONE" ]; then
  # SMALL family: everything -- heuristics, baselines, MIP and all 27 exact-ASP variants
  # -> 39 systems per instance
  FAMILY="small (everything: heuristics + MIP + 27 exact-ASP variants)"
  EXPERIMENTS=()
else
  # LARGE family: heuristics, baselines, MIP and the two best exact-ASP
  # configurations -> 12 systems per instance
  FAMILY="large (heuristics + MIP + ASP rp_dp_sp and rp_d_sp)"
  EXPERIMENTS=(
    --experiment-all-asp-variants=0
    # ENABLED here on purpose: the ASPaeroFlow heuristics and baselines, the MIP, and
    # 05_ASP_rp_dp_sp / 05_ASP_rp_d_sp -- the two best-performing exact configurations.
    --experiment-asp-r-d-s=0    --experiment-asp-r-d-ns=0
    --experiment-asp-r-nd-s=0   --experiment-asp-nr-d-s=0
    --experiment-asp-r-nd-ns=0  --experiment-asp-nr-nd-s=0
    --experiment-asp-nr-d-ns=0  --experiment-asp-nr-nd-ns=0
    --experiment-asp-r-d-sp=0   --experiment-asp-nr-d-sp=0
    --experiment-asp-r-nd-sp=0  --experiment-asp-nr-nd-sp=0
  )
fi
echo "Family:    $FAMILY"
echo "experiment_family=${FAMILY}" >> "output/${FOLDER}/run_provenance_${PROBLEM}.txt"

./start_benchmark_caller.py "${INSTANCE_ROOT}/${PROBLEM}" \
  --output-root=output \
  --output-dir="${FOLDER}/output_${PROBLEM}/" \
  --timestep-granularity="${TIME_GRANULARITY}" \
  --memory-limit="${MEMORY_LIMIT}" \
  --time-limit="${TIME_LIMIT}" \
  --experiment-name="${PROBLEM}" \
  --scaling-experiments=0 \
  --hot-start \
  --wandb-enabled="${WANDB_ENABLED}" \
  --results-format="${RESULTS_FORMAT}" \
  ${EXPERIMENTS[@]+"${EXPERIMENTS[@]}"}

echo "finished=$(date -Is)" >> "output/${FOLDER}/run_provenance_${PROBLEM}.txt"
echo "results_format=${RESULTS_FORMAT}" >> "output/${FOLDER}/run_provenance_${PROBLEM}.txt"
echo "wandb_enabled=${WANDB_ENABLED}" >> "output/${FOLDER}/run_provenance_${PROBLEM}.txt"
echo "[DONE] ${PROBLEM}"
