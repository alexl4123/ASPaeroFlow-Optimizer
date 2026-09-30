#!/usr/bin/env bash
#SBATCH -p broadwell
#SBATCH -t 03:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -J matpack
#SBATCH -o logs/%x-%A_%a.out
#
# PACK THE PUBLISHED SOLUTION MATRICES: ONE ARRAY TASK PER ARCHIVE OF THE PLAN.
#
# Task N writes archive N of <PLAN_DIR>/plan.json into <OUT>/ and verifies it (testzip, then the
# sha256 and size of every member against its MANIFEST.csv) before it puts <ARCHIVE>.zip,
# <ARCHIVE>.MANIFEST.csv and <ARCHIVE>.zip.sha256 in place. The results folders are only read.
#
#   PLAN_DIR   the plan step's --plan-dir                          (required)
#   OUT        where the archives go                               (required)
#   PYTHON     interpreter                                         (default: the potassco env)
#   EXTRA      further flags for the pack step, e.g. "--force" to repack a finished archive
#
# First the plan (one small job; it prints the number of archives N):
#     ./pack_solution_matrices.py plan --plan-dir <PLAN_DIR> --validation ... --results-folder ...
# then
#     sbatch --export=ALL,PLAN_DIR=<PLAN_DIR>,OUT=<OUT> --array=1-N%10 run_pack_solution_matrices.slurm
# and, when the array has drained (seconds):
#     ./pack_solution_matrices.py check --plan-dir <PLAN_DIR> --out-dir <OUT>
#
# SIZE: a task reads its runs' files once, writes the archive once and reads it back three times
# (testzip, member hashes, the archive's own sha256); at 20 GB per archive that is about 80 GB of
# I/O, well within -t 03:00:00 at 50 MB/s. Memory stays small: files are streamed in 1 MB blocks.
# %10 keeps the load on the shared file system moderate.
#
# RESUMABLE: an archive whose three files exist is skipped, so a failed or timed-out task is
# simply resubmitted with the same index. A failed task leaves <ARCHIVE>.zip.partial for a look.
#
# EXIT CODE: 0 packed and verified (or already there), 1 archive larger than the limit, 2 the
# archive could not be written or did not verify (the message says which file).
#
# PARTITION: nothing here is timing-sensitive; outside broadwell exclude the Gurobi nodes:
#     sbatch -p broadwell,sunnycove -x coppernode[25-28] ... run_pack_solution_matrices.slurm

set -uo pipefail

: "${PLAN_DIR:?set PLAN_DIR, e.g. sbatch --export=ALL,PLAN_DIR=...,OUT=... }"
: "${OUT:?set OUT, the folder for the archives}"
: "${SLURM_ARRAY_TASK_ID:?this script is an array job; submit it with --array=1-N}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/potassco/bin/python}"
EXTRA="${EXTRA:-}"

mkdir -p logs "$OUT"
[ -f "$PLAN_DIR/plan.json" ] || { echo "[ERROR] $PLAN_DIR/plan.json not found (run the plan step)" >&2; exit 2; }

echo "Job:      ${SLURM_JOB_NAME:-matpack} ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}  host $(hostname)"
echo "Plan:     $PLAN_DIR  ->  $OUT"
echo "Commit:   $(git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Started:  $(date -Is)"

RC=0
"$PYTHON" -u ./pack_solution_matrices.py pack --plan-dir "$PLAN_DIR" --index "$SLURM_ARRAY_TASK_ID" \
    --out-dir "$OUT" $EXTRA || RC=$?

echo "Finished: $(date -Is)  exit ${RC}"
exit "$RC"
