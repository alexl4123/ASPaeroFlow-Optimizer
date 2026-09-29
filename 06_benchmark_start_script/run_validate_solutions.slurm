#!/usr/bin/env bash
#SBATCH -p broadwell
#SBATCH -t 01:30:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH -J v2valid
#SBATCH -o logs/%x-%A_%a.out
#
# VALIDATE THE SOLUTION MATRICES OF A MERGED CAMPAIGN: ONE ARRAY TASK PER PROBLEM FOLDER.
#
# Task N validates the N-th problem folder (sorted) of <OUTPUT_ROOT>/<FOLDER>/, or CHUNK of them, with
# validate_solutions.py, and writes <OUT>/<FOLDER>/validation_<PROBLEM>.csv (+ .summary.json).
# Nothing under <OUTPUT_ROOT>/<FOLDER>/ is modified.
#
#   FOLDER         campaign folder under OUTPUT_ROOT                  (required, e.g. 20260918_V2)
#   OUTPUT_ROOT    where the campaign folders are                     (default output). Point it at
#                  another checkout's output/ to validate a campaign without touching that checkout.
#   OUT            where the validation CSVs go                       (default output/VALIDATION)
#   INSTANCE_ROOT  folder with <PROBLEM>/<INSTANCE>/                  (default ../05_instances)
#   CHUNK          problem folders per task                           (default 1)
#   SYSTEMS        published | all | comma list                       (default published)
#   EXTRA          further validate_solutions.py flags, e.g. "--no-connectivity"
#
# The task count is the number of problem folders divided by CHUNK:
#     ls -d <OUTPUT_ROOT>/<FOLDER>/output_*/ | wc -l
#
# RESUMABLE. --resume is always passed: rows already written are kept, so a task that hit its
# walltime is simply resubmitted with the same array index and continues where it stopped.
#
# EXIT CODE = the worst verdict of the task, so `sacct -j <job> -o JobID,State,ExitCode` lists the
# problem folders that need a look without opening a log:
#     0 all VALID (NO_MATRIX only where the solver did not finish, or where 03A_CASA overwrote
#       03_DELAY's matrix -- that collision is counted in the CSVs and the summary, not here)
#     1 MISMATCH, or a finished run without its matrix     2 INVALID (a hard constraint violated)
#     3 the validator could not validate (missing instance, instance self-check failed, crash)
# A task killed by the walltime or the memory limit shows as TIMEOUT / OUT_OF_MEMORY instead.
#
# PARTITION: broadwell like the campaign; nothing here is timing-sensitive, so any partition works:
#     sbatch -p <partition> --array=... run_validate_solutions.slurm
#
# THEN, when the arrays have drained (seconds, reads only the CSVs):
#     ./validate_solutions.py --summarize --out-dir <OUT>

set -uo pipefail

: "${FOLDER:?set FOLDER, e.g. sbatch --export=ALL,FOLDER=20260918_V2 ...}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
OUT="${OUT:-output/VALIDATION}"
INSTANCE_ROOT="${INSTANCE_ROOT:-../05_instances}"
CHUNK="${CHUNK:-1}"
SYSTEMS="${SYSTEMS:-published}"
EXTRA="${EXTRA:-}"
: "${SLURM_ARRAY_TASK_ID:?this script is an array job; submit it with --array=1-N}"

mkdir -p logs "$OUT"
[ -d "${OUTPUT_ROOT}/${FOLDER}" ] || { echo "[ERROR] ${OUTPUT_ROOT}/${FOLDER} not found (run from 06_benchmark_start_script)" >&2; exit 3; }
[ -d "$INSTANCE_ROOT" ] || { echo "[ERROR] INSTANCE_ROOT $INSTANCE_ROOT not found" >&2; exit 3; }

mapfile -t PROBLEMS < <(find "${OUTPUT_ROOT}/${FOLDER}" -mindepth 1 -maxdepth 1 -type d -name 'output_*' | sort)
N_PROBLEMS=${#PROBLEMS[@]}
FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK ))
if [ "$FIRST" -ge "$N_PROBLEMS" ]; then
  echo "[SKIP] task ${SLURM_ARRAY_TASK_ID}: only ${N_PROBLEMS} problem folders"
  exit 0
fi

ARGS=()
for (( i = FIRST; i < FIRST + CHUNK && i < N_PROBLEMS; i++ )); do
  ARGS+=(--problem-dir "${PROBLEMS[$i]}")
done

echo "Job:      ${SLURM_JOB_NAME:-v2valid} ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}  host $(hostname)"
echo "Folder:   ${OUTPUT_ROOT}/${FOLDER}  (${N_PROBLEMS} problem folders; this task: $(( ${#ARGS[@]} / 2 )))"
echo "Commit:   $(git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Started:  $(date -Is)"

# CONDA_SH only so the script can be rehearsed off the cluster; the default is the cluster's.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate potassco

RC=0
python -u ./validate_solutions.py "${ARGS[@]}" \
    --instance-root "$INSTANCE_ROOT" --out-dir "$OUT" --systems "$SYSTEMS" --resume $EXTRA || RC=$?

echo "Finished: $(date -Is)  exit ${RC}"
exit "$RC"
