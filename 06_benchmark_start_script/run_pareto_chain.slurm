#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -x coppernode[25-28]
#SBATCH -t 00:30:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -J paretochain
#SBATCH -o logs/%x-%j.out
#
# PARETO CAMPAIGN, UNATTENDED: plan and submit the next round once the previous one has finished.
#
# Between two rounds someone has to check the finished round, plan the next one with
# build_pareto_manifest.py and submit the sbatch lines it prints (TODO-CLUSTER.md, E3, E4, E6). This
# job does exactly that, then submits itself again behind the arrays it started, so a whole
# campaign can be queued in one go and runs without anyone at a terminal. It runs for seconds.
#
# CHAIN_ROUNDS is a "+"-separated list, e.g. E+G+merge. One job of the chain:
#   1. Checks every round already in CHAIN_FOLDER: each manifest row has a result
#      (build_pareto_manifest.py --check exits 0), and no step's .err holds an exception other than
#      the expected ones (no model by the deadline, an unsatisfiable bound, out of memory).
#      Missing rows are resubmitted ONCE (the lines --check prints; rows already done are skipped)
#      with the chain queued behind them; if rows are still missing after that, the chain stops.
#      A missing delay-first end would otherwise make the planner treat that end as open and plan
#      no G for its front.
#   2. With CHAIN_PILOT set: the pilot folder must be complete and its laptop-front table (from
#      merge_pareto_steps.py) must read "yes" on every row, or the chain stops. This is the gate
#      the pilot exists for.
#   3. Plans the first round of CHAIN_ROUNDS and submits the lines the planner prints, unchanged
#      but for --parsable; "merge" runs merge_pareto_steps.py instead.
#   4. Submits itself with the remaining rounds, --dependency=afterany on the arrays of step 3.
# A stop changes nothing: the log (logs/paretochain-<jobid>.out) says why, and the manual commands
# of TODO-CLUSTER.md still apply from that point on.
#
# EVERY ARRAY OF THE FOLDER MUST BE IN THE DEPENDENCY of the first chain job (for the pilot: its E
# and its P2 arrays). Step 1 would otherwise count a row that is still running as missing and run
# it a second time.
#
# SETTINGS travel in the environment (--export=ALL), so values may contain commas and spaces:
#   CHAIN_FOLDER          campaign folder under output/                         (required)
#   CHAIN_ROUNDS          e.g. E+G+merge; rounds E, G, F, P2 and "merge"; each F plans
#                         the next fill round                                  (required)
#   CHAIN_INSTANCE_ROOT   the planner's --instance-root                          (required)
#   CHAIN_PRESET          campaign (default) or pilot
#   CHAIN_PLAN_ARGS       further planner arguments for every round, e.g.
#                         "--partition sunnycove,broadwell --throttle-8g 150 --throttle-40g 40"
#   CHAIN_USC27           round E only: the planner's --usc27-folder
#   CHAIN_PILOT           a pilot folder under output/ that must have passed (gate 2)
#   CHAIN_PILOT_MIN       laptop fronts the pilot must reproduce (default 6, all of them; a
#                         lower value is for a rehearsal on a subset)
#   CHAIN_RETRIED         set by the chain itself after its one resubmission
#   SBATCH_BIN            for a local rehearsal only (a stand-in for sbatch)
#
# EXAMPLE (from 06_benchmark_start_script, after the pilot has been submitted; DEPS are the array
# job ids to wait for, colon-separated):
#   CHAIN_FOLDER=20260928_PARETO CHAIN_ROUNDS=E+G+merge CHAIN_INSTANCE_ROOT="$OPT/05_instances" \
#   CHAIN_USC27="$USC/06_benchmark_start_script/output/USC27" CHAIN_PILOT=PARETO_PILOT \
#   CHAIN_PLAN_ARGS="--partition sunnycove,broadwell --throttle-8g 150 --throttle-40g 40" \
#     sbatch --export=ALL --dependency=afterany:$DEPS run_pareto_chain.slurm

set -euo pipefail

FOLDER="${CHAIN_FOLDER:?set CHAIN_FOLDER}"
ROUNDS="${CHAIN_ROUNDS:?set CHAIN_ROUNDS, e.g. E+G+merge}"
ROOT="${CHAIN_INSTANCE_ROOT:?set CHAIN_INSTANCE_ROOT}"
PRESET="${CHAIN_PRESET:-campaign}"
read -r -a PLAN_ARGS <<< "${CHAIN_PLAN_ARGS:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
ROUND="${ROUNDS%%+*}"
REST=""; [ "$ROUND" != "$ROUNDS" ] && REST="${ROUNDS#*+}"

cd "${SLURM_SUBMIT_DIR:-.}"
[ -f ./build_pareto_manifest.py ] || { echo "[stop] run from 06_benchmark_start_script" >&2; exit 1; }
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate potassco
fi

# The arrays and chain jobs this job submits must not inherit its own job's settings (sbatch
# inside a job exports SLURM_* into the new job), nor any runner setting of the shell the chain
# was started from: run_pareto_steps.slurm reads these names from its environment.
for v in $(compgen -e | grep '^SLURM_' || true); do unset "$v"; done
unset DRY_RUN STEP_TIME_LIMIT MEMORY_LIMIT SKIP_DONE PYTHON_BIN OUTPUT_ROOT TASK_ID \
      SOLVE_DEADLINE_MARGIN MANIFEST CHUNK ROW_OFFSET FOLDER_OVERRIDE 2>/dev/null || true

echo "[chain] $(date -Is) folder ${FOLDER} | preset ${PRESET} | this round ${ROUND} | then ${REST:-nothing}"

submit_lines () {  # stdin: sbatch lines; stdout: the job ids, one per line
  local line words out
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    read -r -a words <<< "$line"            # no eval: coppernode[25-28] must stay literal
    [ "${words[0]}" = "sbatch" ] || { echo "[stop] not an sbatch line: $line" >&2; return 1; }
    echo "[submit] $line" >&2
    out=$("$SBATCH_BIN" --parsable "${words[@]:1}")
    echo "${out%%;*}"
  done
}

chain_next () {  # $1 remaining rounds, $2 colon-separated job ids (may be empty), $3 retried flag
  local rounds="$1" deps="$2" retried="$3" dep_arg=() id
  [ -n "$rounds" ] || { echo "[chain] nothing left to chain."; return 0; }
  [ -n "$deps" ] && dep_arg=( "--dependency=afterany:${deps}" )
  id=$(CHAIN_ROUNDS="$rounds" CHAIN_RETRIED="$retried" \
       "$SBATCH_BIN" --parsable --export=ALL "${dep_arg[@]}" run_pareto_chain.slurm)
  echo "[chain] next: ${rounds} as job ${id%%;*}${deps:+, after ${deps}}"
}

# Exceptions other than the expected ones, over the step .err files of a folder.
unexpected_exceptions () {
  local dir="output/$1/steps"
  [ -d "$dir" ] || return 0
  grep -rh --include='*.err' -E '^[A-Za-z_][A-Za-z_.]*(Error|Exception|Interrupt)' "$dir" \
    | grep -v -E '^RuntimeError: (the solver returned no model before the solve deadline|the program is unsatisfiable)' \
    | grep -v -E 'MemoryError|bad_alloc|Cannot allocate memory|std::length_error' \
    | cut -c1-160 | sort | uniq -c || true
}

# --- 1. the rounds already in this folder ----------------------------------------------------
if compgen -G "output/${FOLDER}/manifests/pareto_*.tsv" > /dev/null; then
  rc=0
  check_out=$(./build_pareto_manifest.py --folder "$FOLDER" --check "${PLAN_ARGS[@]}") || rc=$?
  echo "$check_out"
  if [ "$rc" -eq 2 ]; then
    if [ "${CHAIN_RETRIED:-0}" = "1" ]; then
      echo "[stop] rows are still missing after one resubmission: read their logs" \
           "(logs/pareto-<jobid>_<task>.out) and resubmit with the lines above by hand."
      exit 1
    fi
    ids=$(printf '%s\n' "$check_out" | sed -n 's/^ *\(sbatch .*run_pareto_steps\.slurm\)$/\1/p' | submit_lines)
    [ -n "$ids" ] || { echo "[stop] --check reported missing rows but printed no sbatch line"; exit 1; }
    chain_next "$ROUNDS" "$(printf '%s\n' "$ids" | paste -sd:)" 1
    echo "[chain] resubmitted the missing rows once; this round is planned after them."
    exit 0
  elif [ "$rc" -ne 0 ]; then
    echo "[stop] --check failed (exit $rc)"; exit 1
  fi
fi
bad=$(unexpected_exceptions "$FOLDER")
if [ -n "$bad" ]; then
  echo "[stop] unexpected exceptions in output/${FOLDER}/steps (read those .err files):"
  echo "$bad"; exit 1
fi

# --- 2. the pilot gate -------------------------------------------------------------------------
if [ -n "${CHAIN_PILOT:-}" ]; then
  rc=0; ./build_pareto_manifest.py --folder "$CHAIN_PILOT" --check "${PLAN_ARGS[@]}" > /dev/null || rc=$?
  [ "$rc" -eq 0 ] || { echo "[stop] pilot ${CHAIN_PILOT} is incomplete (--check exit $rc)"; exit 1; }
  bad=$(unexpected_exceptions "$CHAIN_PILOT")
  [ -z "$bad" ] || { echo "[stop] unexpected exceptions in the pilot:"; echo "$bad"; exit 1; }
  ./merge_pareto_steps.py --folder "$CHAIN_PILOT" > /dev/null
  table=$(sed -n '/^## Pilot: laptop fronts/,/^## /p' "output/${CHAIN_PILOT}/pareto_summary.md" \
          | grep -E '^\| V1-' || true)
  echo "$table"
  # A row with steps that does not match is a failure outright; a row without steps is a front
  # the pilot did not run, which only counts against CHAIN_PILOT_MIN (6 = all laptop fronts).
  n_yes=$(printf '%s\n' "$table" | grep -c '| yes |$' || true)
  n_bad=$(printf '%s\n' "$table" | grep '| NO |$' | grep -vc '(no steps)' || true)
  if [ "$n_bad" -gt 0 ] || [ "$n_yes" -lt "${CHAIN_PILOT_MIN:-6}" ]; then
    echo "[stop] pilot gate: ${n_yes} laptop fronts reproduced (${CHAIN_PILOT_MIN:-6} required)," \
         "${n_bad} differ; the campaign is not started."
    exit 1
  fi
  echo "[gate] pilot ${CHAIN_PILOT}: ${n_yes} laptop fronts reproduced, none differs."
fi

# --- 3. this round -----------------------------------------------------------------------------
if [ "$ROUND" = "merge" ]; then
  ./merge_pareto_steps.py --folder "$FOLDER"
  echo "[chain] merged: output/${FOLDER}/pareto_summary.md"
  chain_next "$REST" "" 0
  exit 0
fi
extra=()
[ "$ROUND" = "E" ] && [ -n "${CHAIN_USC27:-}" ] && extra=( --usc27-folder "$CHAIN_USC27" )
plan=$(./build_pareto_manifest.py --folder "$FOLDER" --preset "$PRESET" --round "$ROUND" \
         --instance-root "$ROOT" "${PLAN_ARGS[@]}" "${extra[@]}")
echo "$plan"
ids=$(printf '%s\n' "$plan" | sed -n 's/^ *\(sbatch .*run_pareto_steps\.slurm\)$/\1/p' | submit_lines)
if [ -z "$ids" ]; then
  echo "[chain] round ${ROUND}: nothing submitted."
  # An F round that certifies nothing new ends the fill; the rest (e.g. merge) still runs.
  REST=$(printf '%s\n' "$REST" | tr '+' '\n' | grep -v '^F' | paste -sd+ || true)
fi

# --- 4. the next job of the chain --------------------------------------------------------------
chain_next "$REST" "$(printf '%s\n' "$ids" | grep . | paste -sd: || true)" 0
