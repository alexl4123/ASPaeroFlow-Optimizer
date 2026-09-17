#!/usr/bin/env bash
#SBATCH -p broadwell
#SBATCH -t 02:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -J aspabl
#SBATCH -o logs/%x-%A_%a.out
#
# ASP SOLVER-OPTION ABLATION -- campaign (iii).
#
# The question: which clingo search configuration should someone running these benchmarks
# actually use, and what does that choice cost or buy? The axes are the four named profiles from
# common/clingo_options.py (default, usc, domain, usc-domain) crossed with the exact-ASP systems
# on the SMALL-scaling family. See ABLATION_DESIGN_asp_solver_options.md at the repository root.
#
# NOTHING HERE CHANGES A DEFAULT. `default` remains the default profile everywhere, this script
# never edits a configuration file, and the `default` arm of the ablation is the command line
# every published LPNMR/ATMOS number was produced with, plus --solver-stats (which only adds keys
# to the JSON result lines; it does not touch the search).
#
# ONE ARRAY TASK IS ONE UNIT OF WORK, not one problem directory. run_all_benchmarks.slurm puts a
# whole problem directory in a task, which for the small family is 40 instances x up to 27
# variants -- days of wall clock in a single job. Here a task is (tier, profile, threads, variant
# set, problem, instance), usually 2 solver runs, so a job is an hour at worst. This is
# deliberately the shape the per-run job restructuring wants; if that work has landed and the
# caller understands --only-instance / --only-system, this script uses them automatically.
#
# HOW TO RUN IT (build the index first; it prints these lines itself):
#
#   ./build_ablation_manifest.py
#   sbatch -t 02:00:00 --cpus-per-task=2 --array=1-4%40      run_asp_ablation.slurm   # preflight
#   # ... read the preflight report, THEN:
#   sbatch -t 02:00:00 --cpus-per-task=2 --array=5-804%40    run_asp_ablation.slurm   # tier A
#   sbatch -t 15:00:00 --cpus-per-task=2 --array=805-884%40  run_asp_ablation.slurm   # tier B
#   sbatch -t 02:00:00 --cpus-per-task=5 --array=885-964%40  run_asp_ablation.slurm   # tier C
#
# PARTITION: broadwell, for the same reason run_all_benchmarks.slurm pins it -- long jobs
# elsewhere get cancelled. If broadwell is a node FEATURE here rather than a partition, keep your
# usual partition and add `-C broadwell`. Unlike the main campaign this one needs NO Gurobi
# licence: 04_MIP is off in every tier, because a clingo ablation has nothing to say about MIP.
# That also means no licensed-node constraint and no RUN_MIP modes to get wrong.
#
# MEMORY: 40G matches the main campaign, against the same 35 GiB solver cap. Tasks per node is
#     min(node_memory / 40G, node_cores / cpus-per-task)
# so the %N in --array should be that number times the node count. Check with:
#     sinfo -N -p broadwell -o "%n %c %m"
#
# THREADS: tier C runs four clasp threads, which needs --cpus-per-task=5 (four search threads
# plus the Python parent). It reaches clasp as --solver-arg=--parallel-mode=4, NOT as
# --number-threads: the caller hard-codes SOLVER_SEARCH_THREADS = {"02_ASP": 1} as a module
# constant and exposes no CLI flag for it. build_options() drops the generated --parallel-mode=1
# in favour of the later raw flag, which is what makes the override work at all.

set -euo pipefail

FOLDER="${FOLDER:-$(date +%Y%m%d)_ABLATION}"   # output namespace; override with FOLDER=... sbatch
INSTANCE_ROOT="${INSTANCE_ROOT:-../05_instances}"
MANIFEST="${MANIFEST:-ablation_tasks.tsv}"
TIME_LIMIT="${TIME_LIMIT:-1800}"               # seconds per solver run, as in the papers
PREFLIGHT_TIME_LIMIT="${PREFLIGHT_TIME_LIMIT:-120}"  # tier P only: this tier proves the wiring,
                                                     # it does not measure anything
MEMORY_LIMIT="${MEMORY_LIMIT:-35}"             # GiB, as in the papers; below --mem above
WANDB_ENABLED="${WANDB_ENABLED:-False}"        # thousands of runs; the CSVs are authoritative
RESULTS_FORMAT="${RESULTS_FORMAT:-csv.gz}"     # uncompressed solution matrices are enormous
CHUNK="${CHUNK:-1}"                            # manifest rows per array task
ROW_OFFSET="${ROW_OFFSET:-0}"                  # rows to skip before this array's first task, so
                                               # a manifest longer than the cluster's MaxArraySize
                                               # is submitted as several waves with CHUNK=1
                                               # rather than as fewer, longer tasks
DRY_RUN="${DRY_RUN:-0}"                        # 1 = print the command lines and run nothing
SOLVE_DEADLINE_MARGIN="${SOLVE_DEADLINE_MARGIN:-30}"  # seconds before the limit at which 02_ASP
                                               # stops its own search; "off" = be killed instead

[ -f "$MANIFEST" ] || { echo "[ERROR] $MANIFEST not found -- run ./build_ablation_manifest.py first" >&2; exit 1; }

mkdir -p "output/${FOLDER}" logs

# Same environment the main campaign uses. Guarded so the script can be exercised with
# DRY_RUN=1 on a machine that has no conda install, which is how its command lines get reviewed
# before anything is submitted.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate potassco
else
  echo "[warn] no conda at \$HOME/miniconda3 -- using whatever python is on PATH"
fi

# The caller's --python-bin default is a hard-coded cluster path and run_all_benchmarks.slurm
# relies on it. Same here, overridable for a different install.
PYTHON_BIN="${PYTHON_BIN:-}"

# ---- what the caller can be asked to do ---------------------------------------------------
# --only-instance / --only-system are being added by the per-run job restructuring. Probing for
# them keeps this script working both before and after that lands: with them, the caller is
# pointed straight at the shared instance tree; without them, the task stages a directory holding
# a single symlink, because the caller runs EVERY subdirectory of the directory it is given.
CALLER_HELP="$(./start_benchmark_caller.py --help 2>/dev/null || true)"
HAS_ONLY_INSTANCE=0; HAS_ONLY_SYSTEM=0
case "$CALLER_HELP" in *--only-instance*) HAS_ONLY_INSTANCE=1 ;; esac
case "$CALLER_HELP" in *--only-system*)   HAS_ONLY_SYSTEM=1 ;; esac

run_unit () {
  local row_id="$1"
  local line
  line=$(sed -n "$(( row_id + 1 ))p" "$MANIFEST")
  [ -n "$line" ] || { echo "[skip] no manifest row $row_id (past the end)"; return 0; }

  local TIER PROFILE THREADS VARIANTS PROBLEM INSTANCE TG SYSTEM RUNS
  IFS=$'\t' read -r _ TIER PROFILE THREADS VARIANTS PROBLEM INSTANCE TG SYSTEM RUNS <<< "$line"

  if [ "$TIER" = "PAD" ]; then
    echo "[pad] row ${row_id} is chunk padding -- nothing to do"
    return 0
  fi

  local limit="$TIME_LIMIT"
  if [ "$TIER" = "P" ]; then limit="$PREFLIGHT_TIME_LIMIT"; fi

  # ---- the systems this unit runs --------------------------------------------------------
  # Everything that is not an exact-ASP run is switched off: the ASPaeroFlow heuristics, the
  # baselines, CASA and the MIP. They do not run clingo, so no profile could change them, and
  # running them once per profile would quadruple the campaign to re-measure identical numbers.
  local EXPERIMENTS=(
    --experiment-mip=0
    --experiment-asp-aero-flow=0 --experiment-asp-aero-flow-no-convex=0
    --experiment-asp-aero-flow-nr-nd=0 --experiment-asp-aero-flow-nr-d=0
    --experiment-asp-aero-flow-r-nd=0
    --experiment-casa=0 --experiment-route-delay=0 --experiment-route=0 --experiment-delay=0
  )
  case "$VARIANTS" in
    named2)
      # 05_ASP_rp_dp_sp and 05_ASP_rp_d_sp: the two best-performing exact configurations, and
      # the two the large family also runs, so the ablation speaks about the configurations the
      # published tables actually contain. They stay ON by being left at their default of 1.
      EXPERIMENTS+=( --experiment-all-asp-variants=0 ) ;;
    all27)
      # The full ground-delay x rerouting x dynamic-sectorisation cross. The two named singles
      # are switched OFF here: they are duplicates of two of the 27 under different keys, and
      # running both would pay twice for the same measurement.
      EXPERIMENTS+=( --experiment-all-asp-variants=1
                     --experiment-asp-rp-dp-sp=0 --experiment-asp-rp-d-sp=0 ) ;;
    *) echo "[ERROR] row ${row_id}: unknown variant set '${VARIANTS}'" >&2; return 1 ;;
  esac

  # ---- the solver configuration under test -----------------------------------------------
  # --solver-stats is on for EVERY unit, including the default arm. A timed-out core-guided run
  # and a timed-out branch-and-bound run are not like-for-like on objective value alone: usc
  # optimises the objective levels roughly top-down and can report one model and then nothing,
  # while branch-and-bound improves every level together. The recorded lower bound, cost vector,
  # model count and exhausted flag are what make the two comparable, and the analysis needs all
  # four. It adds keys to the JSON result lines and changes no search decision.
  local SOLVER=( "--solver-profile=${PROFILE}" --solver-stats=True )
  if [ "${THREADS}" -gt 1 ]; then
    SOLVER+=( "--solver-arg=--parallel-mode=${THREADS}" )
  fi
  # ---- the lower bound of a run that does not finish ----------------------------------------
  # The caller kills a run at --time-limit, and a killed clingo never reports its lower bound, so
  # a gap on an instance nobody closes could not be measured. With a margin, 02_ASP stops its own
  # search at limit - margin, still reads its statistics, and prints the bound on its result line.
  # Such a run then EXITS NORMALLY: the caller records a runtime of about limit - margin and no T.
  # The analysis does not read T for that; it reads SOLVER-EXHAUSTED and SOLVER-STOPPED-AT-DEADLINE.
  # This is why the main campaign does not use it. The margin covers reading the statistics and
  # printing the line after the cancel. Translation and grounding cannot be interrupted, so a run
  # still grounding at the limit is killed exactly as without it.
  if [ "$SOLVE_DEADLINE_MARGIN" != "off" ]; then
    SOLVER+=( "--solve-deadline-margin=${SOLVE_DEADLINE_MARGIN}" )
  fi

  # ---- where the results go ----------------------------------------------------------------
  # One output directory per unit, and that is not tidiness: hot-start state lives at
  # <output-dir>/hotstart_state.json, one file per directory, and two concurrent jobs sharing a
  # directory would both write it with last-writer-wins -- the same hazard run_all_benchmarks
  # .slurm documents for its two passes. Per-unit directories make the campaign embarrassingly
  # parallel and make re-running one unit a no-op for every other.
  local tag="${TIER}_${PROFILE}_t${THREADS}"
  if [ "$SYSTEM" != "-" ]; then tag="${tag}/${SYSTEM}"; fi
  local outdir="${FOLDER}/${tag}/${PROBLEM}/${INSTANCE}"

  # ---- point the caller at exactly one instance --------------------------------------------
  local target_dir stage=""
  if [ "$HAS_ONLY_INSTANCE" = "1" ]; then
    target_dir="${INSTANCE_ROOT}/${PROBLEM}"
  else
    # The caller treats every subdirectory of its argument as an instance, so a directory with
    # one symlink in it IS the single-instance selection. Symlink, not copy: the instances are
    # already hard-linked across capacity levels by the expansion script and nothing writes into
    # them. Absolute target, because the caller resolves paths from its own directory.
    stage="output/${FOLDER}/_stage/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${row_id}"
    mkdir -p "$stage"
    ln -sfn "$(readlink -f "${INSTANCE_ROOT}/${PROBLEM}/${INSTANCE}")" "${stage}/${INSTANCE}"
    target_dir="$stage"
  fi

  local SELECT=()
  if [ "$HAS_ONLY_INSTANCE" = "1" ]; then SELECT+=( "--only-instance=${INSTANCE}" ); fi
  if [ "$SYSTEM" != "-" ]; then
    if [ "$HAS_ONLY_SYSTEM" = "1" ]; then
      SELECT+=( "--only-system=${SYSTEM}" )
    else
      echo "[ERROR] row ${row_id} names system '${SYSTEM}' but this start_benchmark_caller.py" >&2
      echo "        has no --only-system. Rebuild the manifest without --per-run-systems." >&2
      return 1
    fi
  fi

  local PYBIN=()
  if [ -n "$PYTHON_BIN" ]; then PYBIN+=( "--python-bin=${PYTHON_BIN}" ); fi

  echo "--------------------------------------------------------------------"
  echo "row ${row_id} | tier ${TIER} | profile ${PROFILE} | ${THREADS} thread(s) | ${VARIANTS}"
  echo "problem ${PROBLEM} | instance ${INSTANCE} | system ${SYSTEM} | limit ${limit}s"
  echo "output  output/${outdir}"

  mkdir -p "output/${outdir}"
  {
    echo "tier=${TIER}"
    echo "solver_profile=${PROFILE}"
    echo "solver_threads=${THREADS}"
    echo "variant_set=${VARIANTS}"
    echo "system=${SYSTEM}"
    echo "problem=${PROBLEM}"
    echo "instance=${INSTANCE}"
    echo "time_granularity=${TG}"
    echo "time_limit_s=${limit}"
    echo "memory_limit_gib=${MEMORY_LIMIT}"
    echo "solver_stats=True"
    echo "solve_deadline_margin_s=${SOLVE_DEADLINE_MARGIN}"
    echo "optimizer_commit=$(git -C .. rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "optimizer_dirty=$(test -n "$(git -C .. status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "slurm_job=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
    echo "manifest_row=${row_id}"
    echo "host=$(hostname)"
    echo "started=$(date -Is)"
  } > "output/${outdir}/run_provenance.txt"

  local CMD=( ./start_benchmark_caller.py "$target_dir"
    --output-root=output
    "--output-dir=${outdir}/"
    "--timestep-granularity=${TG}"
    "--memory-limit=${MEMORY_LIMIT}"
    "--time-limit=${limit}"
    "--experiment-name=${TIER}_${PROFILE}_t${THREADS}_${PROBLEM}_${INSTANCE}"
    # OFF, and it has to stay off: with scaling experiments ON the caller stops attempting a
    # system after its first failure, so one timeout inside a 27-variant unit would silently
    # mark the rest as failed without running them -- and in an ablation the timeouts ARE the
    # measurement.
    --scaling-experiments=0
    --hot-start
    "--wandb-enabled=${WANDB_ENABLED}"
    "--results-format=${RESULTS_FORMAT}"
    "${PYBIN[@]}" "${SELECT[@]}" "${SOLVER[@]}" "${EXPERIMENTS[@]}" )

  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'; printf ' %q' "${CMD[@]}"; echo
  else
    "${CMD[@]}"
    echo "finished=$(date -Is)" >> "output/${outdir}/run_provenance.txt"
  fi

  if [ -n "$stage" ]; then rm -rf "$stage"; fi
  return 0
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-1}}"
FIRST_ROW=$(( ROW_OFFSET + (TASK_ID - 1) * CHUNK + 1 ))
LAST_ROW=$(( ROW_OFFSET + TASK_ID * CHUNK ))
echo "Job: ${SLURM_JOB_NAME:-local}  |  array ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
echo "Manifest rows ${FIRST_ROW}..${LAST_ROW} of ${MANIFEST}  (CHUNK=${CHUNK}, ROW_OFFSET=${ROW_OFFSET})"

for (( row = FIRST_ROW; row <= LAST_ROW; row++ )); do
  run_unit "$row"
done

echo "[DONE] task ${TASK_ID}"
