#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -t 00:40:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -J pareto
#SBATCH -o logs/%x-%A_%a.out
#
# EXACT (DELAY, SECTORS) PARETO FRONTS -- one epsilon-constraint step per array task.
#
# A row of a manifest written by build_pareto_manifest.py names one solver run of one front:
#     DF     the delay-first end (the encoding's own order, no bound)
#     SF     the sectors-first end (sectors @9 above delay @8, no bound)
#     K<k>   sectors minimised under total arrival delay <= k (sectors-first order)
# This script runs exactly that row with 02_ASP/main.py and nothing else. It does NOT go through
# start_benchmark_caller.py: the caller hard-codes each system's flags and cannot pass a per-row
# bound or objective order.
#
# THE STEP. usc (--solver-profile usc), one clasp thread, clasp seed 11904657 (the caller's), fixed
# horizon (--fixed-horizon: no re-solve loop), --solver-stats and --program-stats on, and
# --solve-deadline = time limit - SOLVE_DEADLINE_MARGIN, so 02_ASP stops its own search and still
# prints the lower bound. The time limit (1800 s; 3600 s for the pilot's P2 rows) comes from the
# row. The process is killed at the limit (timeout -s KILL), exactly as the caller would kill it,
# and runs under an address-space cap (prlimit --as) set by the row's memory class.
#
# WHAT IT WRITES, per row, next to nothing else:
#     output/<FOLDER>/steps/<label>/<round>_<step>.jsonl   every JSON line 02_ASP printed (the model
#                                                          stream and the result line)
#     output/<FOLDER>/steps/<label>/<round>_<step>.prov    provenance: row, command, host, commit,
#                                                          exit code, wall time, outcome
#     output/<FOLDER>/steps/<label>/<round>_<step>.err     02_ASP's stderr
# The .jsonl appears only when the run is over (written as .part, then renamed), so a task that
# SLURM kills mid-run (walltime, the job's memory) leaves no .jsonl, and
# `build_pareto_manifest.py --check` lists its row as missing. A run that 02_ASP ends with an
# exception (no model by the deadline, an unsatisfiable bound) still gets its .jsonl: the
# statistics line is printed before the exception, and it is a result.
#
# The script ALWAYS EXITS 0. An out-of-memory or walltime kill shows up in sacct and as a missing
# row; the run's own outcome is in the .prov file and in the merge.
#
# IDEMPOTENT. A row always writes the same files. With SKIP_DONE=1 (the default) a row whose .jsonl
# already exists is skipped, so resubmitting a whole array only runs what is missing. SKIP_DONE=0
# re-runs and overwrites.
#
# HOW TO RUN IT: build_pareto_manifest.py prints the exact sbatch lines, per memory class. From this
# directory, with logs/ created first (SLURM cannot create it):
#     sbatch -J pareto -p sunnycove -x coppernode[25-28] -t 00:40:00 --cpus-per-task=2 --mem=8G \
#       --export=ALL,FOLDER=<F>,INSTANCE_ROOT=<root>,MANIFEST=output/<F>/manifests/pareto_E_8G.tsv,CHUNK=1,ROW_OFFSET=0 \
#       --array=1-<n>%60 run_pareto_steps.slurm
#
# LOCAL REHEARSAL (no SLURM): TASK_ID=<row> FOLDER=... INSTANCE_ROOT=... MANIFEST=... \
#     PYTHON_BIN=$(which python) STEP_TIME_LIMIT=60 bash run_pareto_steps.slurm
# DRY_RUN=1 prints the command lines and runs nothing.

set -euo pipefail

FOLDER="${FOLDER:?pass FOLDER=<campaign folder under output/> in --export}"
INSTANCE_ROOT="${INSTANCE_ROOT:-../05_instances}"
MANIFEST="${MANIFEST:?pass MANIFEST=output/<FOLDER>/manifests/pareto_<round>_<class>.tsv in --export}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"             # where <FOLDER>/steps/ goes (as the planner's --output-root)
CHUNK="${CHUNK:-1}"                              # manifest rows per array task
ROW_OFFSET="${ROW_OFFSET:-0}"                    # rows before this array's first task (waves)
SOLVE_DEADLINE_MARGIN="${SOLVE_DEADLINE_MARGIN:-30}"  # 02_ASP stops its search this long before
                                                 # the kill, to print the bound (as usc27)
STEP_TIME_LIMIT="${STEP_TIME_LIMIT:-}"           # overrides the row's time limit: LOCAL TESTS ONLY
MEMORY_LIMIT="${MEMORY_LIMIT:-}"                 # GiB for prlimit --as (decimals allowed);
                                                 # empty = the row's class: 7 for 8G, 35 for 40G
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
SEED=11904657                                    # start_benchmark_caller.py's --seed, every run

[ -f ../02_ASP/main.py ] || { echo "[ERROR] run this from 06_benchmark_start_script (../02_ASP/main.py not found)" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "[ERROR] manifest $MANIFEST not found -- run ./build_pareto_manifest.py first" >&2; exit 1; }
if [ ! -d "$INSTANCE_ROOT" ]; then
  echo "[ERROR] INSTANCE_ROOT '$INSTANCE_ROOT' does not exist." >&2
  case "$INSTANCE_ROOT" in
    /05_instances*) echo "        That is \$OPT/05_instances with \$OPT empty: export OPT in the shell you submit from." >&2 ;;
  esac
  exit 1
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && [ -z "$PYTHON_BIN" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate potassco
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
# One solver thread is the experiment; keep numpy's BLAS from starting a pool of its own (it would
# also reserve address space against the prlimit cap).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

COMMIT="$(git -C .. rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY="$(test -n "$(git -C .. status --porcelain --untracked-files=no 2>/dev/null)" && echo yes || echo no)"

run_row () {
  local row_id="$1" line
  line=$(sed -n "$(( row_id + 1 ))p" "$MANIFEST")
  [ -n "$line" ] || { echo "[skip] no manifest row $row_id (past the end)"; return 0; }
  local RID ROUND MEM LABEL PROBLEM INSTANCE VARIANT METRIC TG ORDER BOUND STEP LIMIT
  IFS=$'\t' read -r RID ROUND MEM LABEL PROBLEM INSTANCE VARIANT METRIC TG ORDER BOUND STEP LIMIT <<< "$line"
  if [ "$RID" != "$row_id" ]; then
    echo "[ERROR] manifest line $(( row_id + 1 )) is row '$RID', expected $row_id -- the manifest changed under the array" >&2
    return 0
  fi
  [ -n "$STEP_TIME_LIMIT" ] && LIMIT="$STEP_TIME_LIMIT"
  local as_gib="$MEMORY_LIMIT"
  if [ -z "$as_gib" ]; then
    case "$MEM" in 8G) as_gib=7 ;; 40G) as_gib=35 ;; *) echo "[ERROR] row $row_id: unknown memory class $MEM" >&2; return 0 ;; esac
  fi

  local dir="${OUTPUT_ROOT}/${FOLDER}/steps/${LABEL}"
  local base="${dir}/${ROUND}_${STEP}"
  if [ "$SKIP_DONE" = "1" ] && [ -f "${base}.jsonl" ]; then
    echo "[done] row ${row_id}: ${base}.jsonl exists (SKIP_DONE=1)"; return 0
  fi

  local g r s
  case "$VARIANT" in
    *_nd_*) g=0 ;; *_dp_*) g=1 ;; *_d_*) g=2 ;; *) echo "[ERROR] row $row_id: variant $VARIANT" >&2; return 0 ;;
  esac
  case "$VARIANT" in
    nr_*) r=0 ;; rp_*) r=1 ;; r_*) r=2 ;; *) echo "[ERROR] row $row_id: variant $VARIANT" >&2; return 0 ;;
  esac
  case "$VARIANT" in
    *_ns) s=0 ;; *_sp) s=1 ;; *_s) s=2 ;; *) echo "[ERROR] row $row_id: variant $VARIANT" >&2; return 0 ;;
  esac

  local deadline=$(( LIMIT - SOLVE_DEADLINE_MARGIN ))
  local as_bytes; as_bytes=$(awk -v g="$as_gib" 'BEGIN{printf "%d", g * 1024 * 1024 * 1024}')
  local CMD=( timeout -s KILL "${LIMIT}" prlimit "--as=${as_bytes}"
    "$PYTHON_BIN" -u ../02_ASP/main.py
    "--data-dir=${INSTANCE_ROOT}/${PROBLEM}/${INSTANCE}"
    "--seed=${SEED}" "--timestep-granularity=${TG}" --number-threads=1
    "--arrival-delay-metric=${METRIC}" --verbosity=0 --encoding-path=../02_ASP/encoding.lp
    "--regulation-ground-delay-active=${g}" "--regulation-rerouting-active=${r}"
    "--regulation-dynamic-sectorization=${s}"
    --save-results=false --wandb-enabled=false
    --solver-profile=usc --solver-stats=true "--solve-deadline=${deadline}"
    "--objective-order=${ORDER}" --fixed-horizon --program-stats=true )
  [ "$BOUND" != "none" ] && CMD+=( "--delay-bound=${BOUND}" )

  echo "--------------------------------------------------------------------"
  echo "row ${row_id} | round ${ROUND} | ${LABEL} | ${STEP} (${ORDER}, bound ${BOUND}) | ${LIMIT}s | ${MEM} (as ${as_gib} GiB)"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'; printf ' %q' "${CMD[@]}"; echo; return 0
  fi

  mkdir -p "$dir"
  local started t0 t1 rc=0
  started="$(date -Is)"; t0=$(date +%s.%N)
  "${CMD[@]}" > "${base}.jsonl.part" 2> "${base}.err" || rc=$?
  t1=$(date +%s.%N)
  local wall; wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')

  # What ended the run. The merge reads the JSON lines, not this; it is for a human reading logs.
  local outcome=completed
  if [ "$rc" -ne 0 ]; then
    if grep -q "the program is unsatisfiable" "${base}.err"; then outcome=unsatisfiable
    elif grep -q "no model before the solve deadline" "${base}.err"; then outcome=no_model_by_deadline
    elif grep -qiE "bad_alloc|MemoryError|Cannot allocate memory|std::length_error" "${base}.err"; then outcome=memout
    elif [ "$rc" -eq 137 ] || [ "$rc" -eq 124 ]; then outcome=killed_at_limit
    else outcome=error; fi
  fi
  {
    echo "row=${row_id}"; echo "manifest=${MANIFEST}"; echo "round=${ROUND}"; echo "label=${LABEL}"
    echo "problem=${PROBLEM}"; echo "instance=${INSTANCE}"; echo "variant=${VARIANT}"
    echo "metric=${METRIC}"; echo "order=${ORDER}"; echo "bound=${BOUND}"; echo "step=${STEP}"
    echo "time_limit_s=${LIMIT}"; echo "solve_deadline_s=${deadline}"
    echo "mem_class=${MEM}"; echo "address_space_gib=${as_gib}"
    echo "optimizer_commit=${COMMIT}"; echo "optimizer_dirty=${DIRTY}"
    echo "slurm_job=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
    echo "host=$(hostname)"; echo "started=${started}"; echo "finished=$(date -Is)"
    echo "exit_code=${rc}"; echo "wall_s=${wall}"; echo "outcome=${outcome}"
    printf 'command='; printf ' %q' "${CMD[@]}"; echo
  } > "${base}.prov"
  mv -f "${base}.jsonl.part" "${base}.jsonl"
  echo "row ${row_id}: ${outcome} (exit ${rc}, ${wall} s) -> ${base}.jsonl"
  return 0
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-1}}"
FIRST_ROW=$(( ROW_OFFSET + (TASK_ID - 1) * CHUNK + 1 ))
LAST_ROW=$(( ROW_OFFSET + TASK_ID * CHUNK ))
echo "Job: ${SLURM_JOB_NAME:-local}  |  array ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}  |  host $(hostname)"
echo "Manifest rows ${FIRST_ROW}..${LAST_ROW} of ${MANIFEST}  (CHUNK=${CHUNK}, ROW_OFFSET=${ROW_OFFSET})"
for (( row = FIRST_ROW; row <= LAST_ROW; row++ )); do
  run_row "$row" || true
done
echo "[DONE] task ${TASK_ID}"
exit 0
