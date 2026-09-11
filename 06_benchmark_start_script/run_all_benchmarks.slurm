#!/usr/bin/env bash
#SBATCH -p broadwell
#SBATCH -t 720:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
# --mem is the binding constraint on how many tasks share a node, since the solver's own cap is
# 35 GiB and 40G leaves headroom for the interpreter. Tasks per node is therefore
#     min(node_memory / 40G, node_cores / cpus-per-task)
# and the array's %N should be 13 x that, not 13. Check the nodes with:
#     sinfo -N -p broadwell -o "%n %c %m"
# At 1800 s the campaign is ~40,800 core-hours worst case: ~131 days at 13 concurrent,
# ~44 at 39, ~33 at 52. Packing matters more than anything else here.
#SBATCH -J v2bench
#SBATCH -o logs/%x-%A_%a.out
#
# PARTITION: broadwell ONLY. Long-running jobs elsewhere get cancelled by the admin, and a task
# here can run for tens of hours. If `broadwell` is a node FEATURE on this cluster rather than a
# partition, keep your usual partition above and add `#SBATCH -C broadwell` instead. Check with:
#     sinfo -o "%P %N %f" | head
# Either can be overridden per submission without editing this file:
#     sbatch -p <partition> -C broadwell --array=... run_all_benchmarks.slurm
#
# Two consequences of pinning to broadwell, both worth checking before the full launch:
#   * these cores are older than sunnycove, so wall-clock estimates made elsewhere scale up;
#   * concurrency is bounded by the number of broadwell nodes -- `%40` is pointless if there
#     are fewer, and the MIP pass additionally needs licensed nodes WITHIN broadwell.
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
TIME_LIMIT="${TIME_LIMIT:-1800}"             # seconds per solver run -- the value used in the
                                             # LPNMR/ATMOS papers, and start_benchmark_caller.py's
                                             # own default. The ATMOS starter scripts overrode it
                                             # to 600; matching the papers keeps these numbers
                                             # comparable with the published ones.
MEMORY_LIMIT="${MEMORY_LIMIT:-35}"           # GiB, as in the papers; must stay below --mem above
WANDB_ENABLED="${WANDB_ENABLED:-False}"      # see the note below before turning this on
RESULTS_FORMAT="${RESULTS_FORMAT:-csv.gz}"   # uncompressed would be ~2.9 TB across the full grid
RUN_MIP="${RUN_MIP:-auto}"                   # auto | yes | no | only -- see the Gurobi note below

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
# ---- Gurobi licence selection: node-local beats WLS ---------------------------------------
# GRB_LICENSE_FILE takes precedence over Gurobi's own search path, so exporting the WLS file
# unconditionally would force WLS even on nodes carrying a manually installed licence -- and
# with it the WLS limit of 2 concurrent sessions, on precisely the nodes that have no such
# limit. So: prefer a node-local licence, fall back to WLS only when there is none.
#
# Override the search with GRB_LOCAL_LIC=/path/to/gurobi.lic if yours lives elsewhere.
GRB_LOCAL=""
for f in ${GRB_LOCAL_LIC:-} /opt/gurobi/gurobi.lic /opt/gurobi*/gurobi.lic \
         /usr/local/gurobi/gurobi.lic /usr/local/gurobi*/gurobi.lic \
         /opt/gurobi.lic /etc/gurobi.lic; do
  [ -n "$f" ] && [ -f "$f" ] && { GRB_LOCAL="$f"; break; }
done
if [ -n "$GRB_LOCAL" ]; then
  export GRB_LICENSE_FILE="$GRB_LOCAL"
  GRB_WHY="node-local licence at $GRB_LOCAL -- no session cap"
elif [ -f "$HOME/gurobi_wls.lic" ]; then
  export GRB_LICENSE_FILE="$HOME/gurobi_wls.lic"
  GRB_WHY="WLS licence (shared \$HOME) -- capped at 2 concurrent sessions"
else
  GRB_WHY="no licence found; MIP will not run here"
fi
echo "Gurobi:    ${GRB_WHY}"

# Weights & Biases is OFF for this run, deliberately. With tens of thousands of solver runs
# across the full grid it would mean as many logged W&B runs, network traffic from every node, and an
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
  echo "gurobi_licence=${GRB_WHY}"
} > "output/${FOLDER}/run_provenance_${PROBLEM}.txt"

# ---- Gurobi: only some nodes carry a licence --------------------------------------------
# 04_MIP needs Gurobi, and the licence is installed on a SUBSET of the nodes. A 325-task array
# scatters across the partition, so most tasks would land on a node without one. If MIP simply
# failed there, the result set would carry 04_MIP ERROR rows that look like solver failures but
# are really scheduling accidents -- and MIP coverage would depend on where SLURM happened to
# put each task, which is not something to publish.
#
# So MIP is decided explicitly, never accidentally:
#   RUN_MIP=auto  probe this node for a usable licence; run MIP only if one is found (default)
#   RUN_MIP=yes   force MIP on, alongside every other solver
#   RUN_MIP=no    force MIP off -- the sweep across the whole partition
#   RUN_MIP=only  run NOTHING BUT MIP
#
# RUN_MIP=only exists because the licensed nodes (coppernode25-28) are in sunnycove, while the
# campaign has to stay on broadwell or the admin cancels it. A MIP-only task is 21 instances x
# 1 solver, so at the published 1800 s limit it caps at ~10.5 h. It also removes any ordering
# dependency on the main sweep, which RUN_MIP=yes would have (that relies on hot-start to skip
# work already done, so it is only short if the main pass finished that problem first).
#
# MIP RUNS ON EVERY INSTANCE. The pass covers the full --array=1-325, i.e. all 6,920 instances
# of both families, at the same 1800 s / 35 GiB as the papers -- the limits are NOT shortened
# for MIP. --scaling-experiments=0 is passed unconditionally below, which matters here: with
# scaling experiments enabled the caller stops attempting a solver after its first failure, so
# one early timeout would silently truncate MIP's coverage. It is off, so every instance is
# attempted regardless of what happened on the previous one.
#
# Worst case that is ~71 days at %2, but it will not come to that: on the larger instances MIP
# reaches the 35 GiB cap within minutes and returns MEMOUT, so the 1800 s only ever binds where
# the model actually fits.
#
#   NOTE on auto: this script exports GRB_LICENSE_FILE=$HOME/gurobi_wls.lic before probing, and
#   $HOME is shared, so a WLS (network) licence validates on ANY node. If WLS works for you,
#   that is fine and auto is right. If it does not, auto will claim a licence where MIP then
#   fails -- verify with a real solve before trusting it, and use no/only instead.
#
# The recommended pattern is TWO submissions, into SEPARATE output folders:
#   sbatch --export=ALL,RUN_MIP=no,FOLDER=20260912_V2 \
#          --array=1-325%13 run_all_benchmarks.slurm
#   sbatch -p sunnycove -w coppernode[25-28] \
#          --export=ALL,RUN_MIP=only,FOLDER=20260912_V2_MIP \
#          --array=1-325%2 run_all_benchmarks.slurm      # all 325, full limits, %2 = WLS cap
#
# SEPARATE folders, and this matters: hot-start state lives at
# output/<FOLDER>/output_<PROBLEM>/hotstart_state.json, one file per problem. Two passes
# running CONCURRENTLY against the same FOLDER would both write it, and the last writer wins
# -- one pass's entries would vanish from the other's state. Sequential passes can share a
# folder safely (that is what makes RUN_MIP=yes resumable); concurrent ones must not. Merge
# the two result sets afterwards.
#
# Concurrency on the MIP pass depends on WHICH licence the node uses. A WLS licence permits
# only 2 concurrent Gurobi sessions, so %2 there. Nodes with a manually installed local
# licence have no such cap, and concurrency is bounded by memory and cores exactly as on the
# broadwell sweep: min(node_memory / 40G, cores / cpus-per-task) per node. Check which licence
# a node actually picked in the job log ("Gurobi: ...") or in the provenance file.
mip_licence_found() {
  # A FUNCTIONAL check, not a file check. $HOME is shared across the whole cluster, so testing
  # for $HOME/gurobi_wls.lic would succeed on every node and defeat the entire point -- the
  # licences that matter here are the node-local ones. gurobi_cl actually validates, so it is
  # authoritative when present; otherwise fall back to node-local paths only, never $HOME.
  if command -v gurobi_cl >/dev/null 2>&1; then
    gurobi_cl --license >/dev/null 2>&1 && return 0
    return 1
  fi
  for f in /opt/gurobi/gurobi.lic /opt/gurobi*/gurobi.lic \
           /usr/local/gurobi/gurobi.lic /usr/local/gurobi*/gurobi.lic; do
    [ -f "$f" ] && return 0
  done
  return 1
}

case "$RUN_MIP" in
  yes)  MIP_FLAG="--experiment-mip=1"; MIP_WHY="forced on (RUN_MIP=yes)" ;;
  no)   MIP_FLAG="--experiment-mip=0"; MIP_WHY="forced off (RUN_MIP=no)" ;;
  only) MIP_FLAG="--experiment-mip=1"; MIP_WHY="MIP ONLY (RUN_MIP=only)" ;;
  auto)
    if mip_licence_found; then
      MIP_FLAG="--experiment-mip=1"; MIP_WHY="licence found on $(hostname)"
    else
      MIP_FLAG="--experiment-mip=0"; MIP_WHY="no licence on $(hostname); MIP skipped, not failed"
    fi ;;
  *) echo "[ERROR] RUN_MIP must be auto, yes, no or only (got: $RUN_MIP)" >&2; exit 1 ;;
esac
echo "MIP:       ${MIP_FLAG#--experiment-mip=} -- ${MIP_WHY}"

# ---- the experiment selection ------------------------------------------------------------
# Every flag defaults to 1 in start_benchmark_caller.py, so what is listed here is what gets
# switched OFF.
#
# --experiment-all-asp-variants expands to 27 exact-ASP configurations (ground-delay x rerouting
# x dynamic-sectorisation, each in {none, partial, full}), all running 02_ASP/main.py -- the
# EXACT solver. The full 27 belong on the small-scaling family, which exists precisely so exact
# methods can close instances; on a 31,622-flight TG=60 instance they would time out essentially
# every time, at 1800 seconds per run of predictable non-information.
#
# TWO of them are kept on the large family as well, because they are the best-performing exact
# configurations and are worth having a large-scale profile for:
#     05_ASP_rp_dp_sp  =  ASP(r_p, d_p, s_p)   partial rerouting, partial delay, partial DAC
#     05_ASP_rp_d_sp   =  ASP(r_p, d,   s_p)   partial rerouting, full delay,    partial DAC
#
# The manifest distinguishes the two families: small-scaling rows carry capacity_level=NONE
# because that family has no PCAP sweep (cap-enroute is already 1).
if [ "$RUN_MIP" = "only" ]; then
  # Everything except 04_MIP switched off.
  FAMILY="MIP only"
  EXPERIMENTS=(
    "$MIP_FLAG"
    --experiment-asp-aero-flow=0 --experiment-asp-aero-flow-no-convex=0
    --experiment-asp-aero-flow-nr-nd=0 --experiment-asp-aero-flow-nr-d=0
    --experiment-asp-aero-flow-r-nd=0
    --experiment-casa=0 --experiment-route-delay=0 --experiment-route=0 --experiment-delay=0
    --experiment-all-asp-variants=0
    --experiment-asp-rp-dp-sp=0 --experiment-asp-rp-d-sp=0
  )
elif [ "$CAP_LEVEL" = "NONE" ]; then
  # SMALL family: everything -- heuristics, baselines, MIP and all 27 exact-ASP variants
  # -> 39 systems per instance
  FAMILY="small (everything: heuristics + MIP + 27 exact-ASP variants)"
  EXPERIMENTS=( "$MIP_FLAG" )
else
  # LARGE family: heuristics, baselines, MIP and the two best exact-ASP
  # configurations -> 12 systems per instance
  FAMILY="large (heuristics + MIP + ASP rp_dp_sp and rp_d_sp)"
  EXPERIMENTS=(
    "$MIP_FLAG"
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
echo "mip=${MIP_FLAG#--experiment-mip=} (${MIP_WHY})" >> "output/${FOLDER}/run_provenance_${PROBLEM}.txt"
echo "wandb_enabled=${WANDB_ENABLED}" >> "output/${FOLDER}/run_provenance_${PROBLEM}.txt"
echo "[DONE] ${PROBLEM}"
