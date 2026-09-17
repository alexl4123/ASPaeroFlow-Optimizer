#!/usr/bin/env bash
#SBATCH -p broadwell
#SBATCH -t 00:40:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -J v2unit
#SBATCH -o logs/%x-%A_%a.out
#
# ONE SMALL JOB PER TIMEOUT-BOUNDED SOLVER RUN.
#
# run_all_benchmarks.slurm submits one array task per PROBLEM -- 21 instances x 12 systems (large
# family) or x 39 (small family), each run bounded by --time-limit. That is tens of hours per
# task, which is what the admins object to and what backfills worst into a queue saturated by
# other people's large arrays. This script schedules the smallest naturally bounded piece of work
# instead: ONE (problem, instance, system) unit, at most TIME_LIMIT seconds.
#
# run_all_benchmarks.slurm is NOT replaced and still works. This is the second path, not the new
# one. Everything below that concerns the cluster rather than the scheduling shape -- the Gurobi
# licence selection, the RUN_MIP modes, the partition pinning, the W&B decision, the memory
# budget -- is the same as there, and the reasoning for it is spelled out there.
#
# HOW A CAMPAIGN IS RUN
#
#   1. ./build_worklist.py --folder 20260916_V2 --run-mip no --chunk 1
#        enumerates every unit into output/20260916_V2/units/worklist.tsv and prints the exact
#        sbatch line, the task count and the walltime for the CHUNK you asked for.
#   2. sbatch -t ... --export=ALL,FOLDER=...,CHUNK=...,RUN_MIP=... --array=1-N%40 <this file>
#   3. ./benchmark_progress.py --folder 20260916_V2       (works while the array runs)
#   4. ./merge_benchmark_shards.py --folder 20260916_V2
#        concatenates the shards into output/20260916_V2/output_<PROBLEM>/*.csv -- the same files,
#        at the same paths, that the monolithic form writes -- and names any unit that died.
#
# CHUNK AND WAVES
#
# Task N processes worklist units [UNIT_OFFSET + (N-1)*CHUNK + 1 .. UNIT_OFFSET + N*CHUNK].
# CHUNK=1 gives ~30-minute jobs, CHUNK=4 gives ~2-hour jobs.
#
# MaxArraySize is the constraint, and it bites: tasks = ceil(units / CHUNK), and at SLURM's
# default MaxArraySize of 1001 a 59,000-unit campaign would need CHUNK >= 59, i.e. ~30-hour
# tasks -- longer than the monolithic tasks this is meant to replace. Check yours with
#     scontrol show config | grep MaxArraySize
#
# UNIT_OFFSET is the way out, and it is why CHUNK does not have to grow with the campaign. Submit
# the SAME worklist as several WAVES of at most MaxArraySize - 1 tasks each (MaxArraySize bounds
# the task INDEX: at MaxArraySize=50000, --array=1-50000 is rejected and 1-49999 is accepted), each
# wave starting where the last one ended -- build_worklist.py prints these lines:
#     sbatch --export=ALL,FOLDER=F,CHUNK=1,UNIT_OFFSET=0    --array=1-1000%40 run_benchmark_units.slurm
#     sbatch --export=ALL,FOLDER=F,CHUNK=1,UNIT_OFFSET=1000 --array=1-1000%40 run_benchmark_units.slurm
#     ...
# The waves are independent -- a unit depends on nothing but its own instance -- so they can be
# submitted all at once or a few at a time as the queue drains. build_worklist.py prints the
# offsets and the array ranges. Small CHUNK is better for backfill; large CHUNK amortises the
# ~1 minute of conda activation and licence probing per task.
#
# Whatever CHUNK you pass, pass a matching -t: a task killed by the walltime is a missing unit.
#
# PARTITION: broadwell ONLY, for the same reason as the monolithic script -- long jobs elsewhere
# get cancelled by the admin. These jobs are short, so that pressure is gone, but the cores differ
# between partitions and mixing them would put incomparable wall-clock numbers in one CSV. If
# `broadwell` is a node FEATURE here rather than a partition, use `#SBATCH -C broadwell` and your
# usual partition. Override per submission without editing this file:
#     sbatch -p <partition> -C broadwell --array=... run_benchmark_units.slurm
#
# NO SHARED MUTABLE STATE. run_all_benchmarks.slurm warns that two passes writing the same FOLDER
# both write output/<FOLDER>/output_<PROBLEM>/hotstart_state.json and the last writer wins. With
# hundreds of concurrent small jobs that hazard would be the normal case, not the exception, so
# every unit gets its OWN output directory:
#     output/<FOLDER>/units/shards/<PROBLEM>/<INSTANCE>/<SYSTEM>/
# Nothing is written twice by two jobs, and the merge is a pure function of the shards.

set -euo pipefail

FOLDER="${FOLDER:-$(date +%Y%m%d)_V2}"       # output namespace; override with FOLDER=... sbatch
INSTANCE_ROOT="${INSTANCE_ROOT:-../05_instances}"
WORKLIST="${WORKLIST:-output/${FOLDER}/units/worklist.tsv}"
CHUNK="${CHUNK:-1}"                          # units per array task
UNIT_OFFSET="${UNIT_OFFSET:-0}"              # units already covered by earlier WAVES -- see below
TIME_LIMIT="${TIME_LIMIT:-1800}"             # seconds per solver run -- the value used in the
                                             # LPNMR/ATMOS papers, and start_benchmark_caller.py's
                                             # own default. Keep it at 1800 or the numbers stop
                                             # being comparable with the published ones.
MEMORY_LIMIT="${MEMORY_LIMIT:-35}"           # GiB, as in the papers; must stay below --mem above
WANDB_ENABLED="${WANDB_ENABLED:-False}"      # see the note in run_all_benchmarks.slurm
RESULTS_FORMAT="${RESULTS_FORMAT:-csv.gz}"   # uncompressed would be ~2.9 TB across the full grid
RUN_MIP="${RUN_MIP:-auto}"                   # auto | yes | no | only -- see the Gurobi note below
PYTHON_BIN="${PYTHON_BIN:-}"                 # interpreter for the solvers. EMPTY BY DEFAULT, which
                                             # leaves start_benchmark_caller.py's own default in
                                             # place -- the cluster's
                                             # /home/guests/abeiser/miniconda3/envs/potassco/bin/python.
                                             # Set it to run this script anywhere else, e.g. to
                                             # rehearse a submission on a workstation:
                                             #   PYTHON_BIN=$(which python) bash run_benchmark_units.slurm

PYBIN_FLAG=()
[ -n "$PYTHON_BIN" ] && PYBIN_FLAG=(--python-bin="$PYTHON_BIN")

[ -f "$WORKLIST" ] || { echo "[ERROR] $WORKLIST not found -- run ./build_worklist.py first" >&2; exit 1; }
: "${SLURM_ARRAY_TASK_ID:?this script is an array job; submit it with --array=1-N}"

# ---- which units is this task? --------------------------------------------------------------
N_UNITS=$(( $(wc -l < "$WORKLIST") - 1 ))    # line 1 is the header
FIRST_UNIT=$(( UNIT_OFFSET + (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
LAST_UNIT=$(( UNIT_OFFSET + SLURM_ARRAY_TASK_ID * CHUNK ))
[ "$LAST_UNIT" -le "$N_UNITS" ] || LAST_UNIT="$N_UNITS"
if [ "$FIRST_UNIT" -gt "$N_UNITS" ]; then
  # Harmless: the array was submitted wider than the worklist. Say so and stop, rather than
  # failing, so an over-wide --array does not fill the log with errors.
  echo "[SKIP] task ${SLURM_ARRAY_TASK_ID}: units start at ${FIRST_UNIT} but the worklist has ${N_UNITS}"
  exit 0
fi

echo "Job:       ${SLURM_JOB_NAME}  |  array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Worklist:  ${WORKLIST}  (${N_UNITS} units)"
echo "Units:     ${FIRST_UNIT}..${LAST_UNIT}  (CHUNK=${CHUNK}, UNIT_OFFSET=${UNIT_OFFSET})"
echo "Output:    output/${FOLDER}/units/shards/"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco

# ---- Gurobi licence selection: the per-node licence beats WLS -----------------------------
# Identical to run_all_benchmarks.slurm, and for the same reason. GRB_LICENSE_FILE takes
# precedence over Gurobi's own search path, so exporting the WLS file unconditionally would force
# WLS even on the nodes carrying a manually installed licence -- and with it the WLS limit of 2
# concurrent sessions, on precisely the nodes that have no such limit.
#
# The per-node licences live in the SHARED home, named after the node:
#     $HOME/gurobi_<hostname>.lic/gurobi.lic     e.g. gurobi_coppernode25.lic/gurobi.lic
# Because $HOME is shared, the existence of any of those files says nothing about the node this
# task landed on -- only the one matching this hostname does. Hence the hostname-keyed lookup
# rather than a path search.
#
# Override with GRB_LOCAL_LIC=/path/to/gurobi.lic if needed.
GRB_HOST="$(hostname -s 2>/dev/null || hostname)"
GRB_NODE_LIC="${GRB_LOCAL_LIC:-$HOME/gurobi_${GRB_HOST}.lic/gurobi.lic}"
if [ -f "$GRB_NODE_LIC" ]; then
  export GRB_LICENSE_FILE="$GRB_NODE_LIC"
  GRB_WHY="per-node licence $GRB_NODE_LIC -- no session cap"
elif [ -f "$HOME/gurobi_wls.lic" ]; then
  export GRB_LICENSE_FILE="$HOME/gurobi_wls.lic"
  GRB_WHY="WLS licence (shared \$HOME) -- capped at 2 concurrent sessions"
else
  GRB_WHY="no licence found for ${GRB_HOST}; MIP will not run here"
fi
echo "Gurobi:    ${GRB_WHY}"

# ---- Gurobi: only some nodes carry a licence --------------------------------------------
# 04_MIP needs Gurobi and the licence is installed on a SUBSET of the nodes, so MIP is decided
# explicitly, never accidentally:
#   RUN_MIP=auto  probe this node for a usable licence; run MIP only if one is found (default)
#   RUN_MIP=yes   force MIP on, alongside every other solver
#   RUN_MIP=no    force MIP off -- the sweep across the whole partition
#   RUN_MIP=only  run NOTHING BUT MIP
#
# RUN_MIP decides which units EXIST -- that is build_worklist.py's --run-mip, and it must match
# what you pass here, because a unit's command line is rebuilt from it. Build one worklist per
# mode; do not reuse a --run-mip=no worklist with RUN_MIP=yes.
#
# auto is the one mode whose answer is a property of the NODE, and the worklist is built on the
# login node. So the worklist CONTAINS the MIP units and this script decides per task. Where no
# licence is found, a MIP unit is SKIPPED -- recorded as skipped_no_licence in the shard, not run
# and not failed, which is what auto has always meant. The merge then reports 04_MIP as absent for
# that problem rather than as a column of ERROR rows that look like solver failures but are really
# scheduling accidents.
#
# Per-unit scheduling makes the recommended pattern easier, not harder: two worklists, two
# folders, and the MIP one pinned to the licensed nodes.
#   ./build_worklist.py --folder 20260916_V2     --run-mip no
#   ./build_worklist.py --folder 20260916_V2_MIP --run-mip only
#   sbatch --export=ALL,FOLDER=20260916_V2,CHUNK=1,RUN_MIP=no --array=1-N%40 run_benchmark_units.slurm
#   sbatch -p sunnycove -x coppernode[21-24,29-30] \
#          --export=ALL,FOLDER=20260916_V2_MIP,CHUNK=1,RUN_MIP=only --array=1-M%48 run_benchmark_units.slurm
#
# Note -x (exclude), NOT -w (nodelist): this script requests "#SBATCH -N 1" and -w asks for a job
# spanning ALL the listed nodes, which fails with "invalid number of nodes". sunnycove is
# coppernode[21-30] and the licensed nodes are 25-28.
#
# Concurrency on the MIP pass depends on WHICH licence the node uses: a WLS licence permits only 2
# concurrent Gurobi sessions, so %2 there; nodes with a local licence are bounded by memory and
# cores as usual.
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
  yes|no|only) MIP_LICENCE="yes"; MIP_WHY="RUN_MIP=${RUN_MIP} (node licence not consulted)" ;;
  auto)
    if mip_licence_found; then
      MIP_LICENCE="yes"; MIP_WHY="licence found on $(hostname)"
    else
      MIP_LICENCE="no";  MIP_WHY="no licence on $(hostname); MIP skipped, not failed"
    fi ;;
  *) echo "[ERROR] RUN_MIP must be auto, yes, no or only (got: $RUN_MIP)" >&2; exit 1 ;;
esac
echo "MIP:       ${MIP_WHY}"

# Weights & Biases is OFF for this run, deliberately: with tens of thousands of solver runs it
# would mean as many logged W&B runs and network traffic from every node, and it is a HARD failure
# mode -- 01_ASPaeroFlow/main.py raises FileNotFoundError when wandb.key is absent, before wandb is
# ever consulted, so WANDB_MODE=disabled does NOT protect you. The CSVs are authoritative anyway.

PROV_DIR="output/${FOLDER}/units/provenance"
mkdir -p "$PROV_DIR" "logs"
PROV="${PROV_DIR}/task_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}.txt"

# ---- provenance: pin exactly what produced these numbers -----------------------------------
# The optimizer's capacity semantics and arrival-delay metric changed in 2026-09. Results from
# before that change are NOT comparable with these, so record the commit in the output tree. The
# merge folds these per-task files into the per-problem run_provenance_<PROBLEM>.txt that the
# monolithic form writes.
{
  echo "worklist=${WORKLIST}"
  echo "units=${FIRST_UNIT}..${LAST_UNIT}"
  echo "chunk=${CHUNK}"
  echo "time_limit_s=${TIME_LIMIT}"
  echo "memory_limit_gib=${MEMORY_LIMIT}"
  echo "optimizer_commit=$(git -C .. rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "optimizer_dirty=$(test -n "$(git -C .. status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "slurm_job=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
  echo "host=$(hostname)"
  echo "started=$(date -Is)"
  echo "gurobi_licence=${GRB_WHY}"
  echo "run_mip=${RUN_MIP} (${MIP_WHY})"
  echo "results_format=${RESULTS_FORMAT}"
  echo "wandb_enabled=${WANDB_ENABLED}"
} > "$PROV"

# ---- the units ------------------------------------------------------------------------------
N_OK=0; N_FAIL=0; N_SKIP=0
for (( UNIT = FIRST_UNIT; UNIT <= LAST_UNIT; UNIT++ )); do
  IFS=$'\t' read -r UNIT_ID PROBLEM TIME_GRANULARITY REGION CAP_LEVEL FAMILY INSTANCE SYSTEM \
      < <(sed -n "$(( UNIT + 1 ))p" "$WORKLIST")
  if [ "${UNIT_ID:-}" != "$UNIT" ]; then
    echo "[ERROR] worklist line $(( UNIT + 1 )) is unit '${UNIT_ID:-}', expected ${UNIT} -- the "\
         "worklist changed under the array. Regenerate it and resubmit." >&2
    exit 1
  fi

  SHARD="${FOLDER}/units/shards/${PROBLEM}/${INSTANCE}/${SYSTEM}"
  mkdir -p "output/${SHARD}"
  echo "=== unit ${UNIT}/${N_UNITS}  ${PROBLEM}  ${INSTANCE}  ${SYSTEM}  [${FAMILY}] ==="

  # RUN_MIP=auto on a node without a licence: skip, do not fail. Recorded so the merge can tell
  # a deliberate skip from a job that died.
  if [ "$SYSTEM" = "04_MIP" ] && [ "$MIP_LICENCE" = "no" ]; then
    printf '{"unit_id": %s, "problem": "%s", "instance": "%s", "system": "%s", "status": "skipped_no_licence", "host": "%s", "ts": "%s"}\n' \
      "$UNIT" "$PROBLEM" "$INSTANCE" "$SYSTEM" "$(hostname)" "$(date -Is)" > "output/${SHARD}/unit.json"
    echo "[SKIP] no Gurobi licence on $(hostname) (RUN_MIP=auto)"
    N_SKIP=$(( N_SKIP + 1 ))
    continue
  fi

  # The --experiment-* flags run_all_benchmarks.slurm would pass for this problem. They are not
  # strictly needed once --only-system has named the system, and they are passed anyway as a
  # guard: if the worklist and the family rules ever disagree, the caller exits with "no such
  # enabled system" instead of quietly running work the campaign did not ask for.
  mapfile -t EXPERIMENTS < <(./benchmark_families.py --capacity-level "$CAP_LEVEL" \
                                                     --run-mip "$RUN_MIP" \
                                                     --mip-licence "$MIP_LICENCE")

  UNIT_START="$(date -Is)"
  UNIT_RC=0
  # set -e must not take the rest of the chunk down with one failed unit.
  ./start_benchmark_caller.py "${INSTANCE_ROOT}/${PROBLEM}" \
    --output-root=output \
    --output-dir="${SHARD}/" \
    --timestep-granularity="${TIME_GRANULARITY}" \
    --memory-limit="${MEMORY_LIMIT}" \
    --time-limit="${TIME_LIMIT}" \
    --experiment-name="${PROBLEM}" \
    --scaling-experiments=0 \
    --hot-start \
    --wandb-enabled="${WANDB_ENABLED}" \
    --results-format="${RESULTS_FORMAT}" \
    --only-instance="${INSTANCE}" \
    --only-system="${SYSTEM}" \
    ${PYBIN_FLAG[@]+"${PYBIN_FLAG[@]}"} \
    "${EXPERIMENTS[@]}" || UNIT_RC=$?

  if [ "$UNIT_RC" -eq 0 ]; then STATUS="ok"; N_OK=$(( N_OK + 1 ));
  else STATUS="failed"; N_FAIL=$(( N_FAIL + 1 )); fi
  printf '{"unit_id": %s, "problem": "%s", "instance": "%s", "system": "%s", "family": "%s", "status": "%s", "exit_code": %s, "host": "%s", "slurm_job": "%s", "started": "%s", "finished": "%s"}\n' \
    "$UNIT" "$PROBLEM" "$INSTANCE" "$SYSTEM" "$FAMILY" "$STATUS" "$UNIT_RC" "$(hostname)" \
    "${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}" "$UNIT_START" "$(date -Is)" \
    > "output/${SHARD}/unit.json"
  [ "$UNIT_RC" -eq 0 ] || echo "[WARN] unit ${UNIT} exited ${UNIT_RC} -- the merge will list it as missing"
done

{
  echo "finished=$(date -Is)"
  echo "units_ok=${N_OK}"
  echo "units_failed=${N_FAIL}"
  echo "units_skipped=${N_SKIP}"
} >> "$PROV"
echo "[DONE] task ${SLURM_ARRAY_TASK_ID}: ${N_OK} ok, ${N_FAIL} failed, ${N_SKIP} skipped"
