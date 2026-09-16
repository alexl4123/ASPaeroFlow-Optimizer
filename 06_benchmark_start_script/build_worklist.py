#!/usr/bin/env python3
"""Enumerate every (problem, instance, solver system) unit of a campaign into a TSV.

WHY

run_all_benchmarks.slurm schedules one array task per PROBLEM: 21 instances x 12 systems for the
large family, x 39 for the small one, each solver run bounded by --time-limit (1800 s). A task is
therefore tens of hours long, which is what the cluster admins object to and what backfills worst
into a queue saturated by other people's array jobs.

The smallest naturally timeout-bounded piece of work is ONE (problem, instance, system) triple:
one solver run, at most --time-limit seconds. This script writes them all out, in a fixed order,
so that run_benchmark_units.slurm can address them by number.

DETERMINISM IS THE WHOLE POINT

Array index N must mean the same work on Monday and on Friday, or a resubmission reruns the wrong
thing and a merge silently pairs the wrong results. So the order is pinned to three things that
are themselves ordered:

  1. manifest row order      -- the same index space run_all_benchmarks.slurm's --array uses
  2. sorted(instance dirs)   -- the same expression start_benchmark_caller.py sorts with
  3. build_system_config()   -- called here, not re-derived, so keys and column order are the real
                                ones; this is also the CSV column order

Consequently the units of one problem come out instance-outer, system-inner, which is exactly the
loop order of the monolithic run, and the merged CSVs come out with the rows and columns in the
order the monolithic run would have produced.

The manifest is ../05_instances/problems.tsv, written by the generator's
expand_instances_for_benchmark.sh. The worklist is regenerated from it, never edited by hand.

    ./build_worklist.py --folder 20260916_V2 --run-mip no
    ./build_worklist.py --folder 20260916_V2_MIP --run-mip only --chunk 4
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_families as families              # noqa: E402
from start_benchmark_caller import build_arg_parser, build_system_config   # noqa: E402

#: Columns of the worklist TSV. unit_id is 1-based and is the address the array script resolves.
COLUMNS = ("unit_id", "problem_dir", "time_granularity", "region", "capacity_level",
           "family", "instance", "system")

#: SLURM's own default when MaxArraySize is not set in slurm.conf.
DEFAULT_MAX_ARRAY_SIZE = 1001

#: Seconds a unit needs beyond --time-limit: interpreter start-up, instance parsing outside the
#: solver's own clock, and writing the shard. Deliberately generous; a job killed by the walltime
#: is a missing unit that has to be resubmitted.
PER_UNIT_OVERHEAD_S = 120

#: Seconds of slack for the task itself (conda activation, the licence probe, the final echo).
PER_TASK_OVERHEAD_S = 300


def read_manifest(path: Path) -> List[dict]:
    """The problem index, in file order. Columns as written by expand_instances_for_benchmark.sh."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < len(header):
                continue
            rows.append(dict(zip(header, fields)))
    return rows


def systems_for(problem_dir: Path, capacity_level: str, run_mip: str) -> List[str]:
    """The system keys this problem runs, from build_system_config itself.

    The flags are the ones run_all_benchmarks.slurm would pass, so this is not an opinion about
    what the campaign contains -- it is the campaign's own answer.
    """
    flags = families.experiment_flags(capacity_level, run_mip)
    args = build_arg_parser().parse_args([str(problem_dir), *flags])
    base_dir = Path(__file__).resolve().parent
    return [s["key"] for s in build_system_config(base_dir, Path("output"), "worklist", args)]


def detect_max_array_size() -> int:
    """Ask SLURM for MaxArraySize; fall back to SLURM's own default off-cluster."""
    if not shutil.which("scontrol"):
        return DEFAULT_MAX_ARRAY_SIZE
    try:
        out = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return DEFAULT_MAX_ARRAY_SIZE
    for line in out.splitlines():
        if line.strip().startswith("MaxArraySize"):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                break
    return DEFAULT_MAX_ARRAY_SIZE


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=None,
                    help="output namespace, as FOLDER= is for the slurm scripts. The worklist "
                         "lands in output/<FOLDER>/units/worklist.tsv unless --out says otherwise.")
    ap.add_argument("--out", type=Path, default=None, help="where to write the worklist TSV")
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--instance-root", type=Path, default=Path("../05_instances"))
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: <instance-root>/problems.tsv")
    ap.add_argument("--run-mip", default="auto", choices=list(families.RUN_MIP_MODES),
                    help="as RUN_MIP= is for run_all_benchmarks.slurm. auto enumerates the MIP "
                         "units and lets each task decide from its own node's licence.")
    ap.add_argument("--problem", action="append", default=None, metavar="NAME",
                    help="restrict to these problem directories (repeatable). Renumbers the "
                         "units, so treat the result as a campaign of its own.")
    ap.add_argument("--chunk", type=int, default=1,
                    help="units per array task, for the arithmetic printed at the end. The "
                         "runner takes CHUNK as an environment variable, not from the worklist.")
    ap.add_argument("--time-limit", type=int, default=1800,
                    help="seconds per solver run, only used for the walltime recommendation")
    ap.add_argument("--max-array-size", type=int, default=None,
                    help="default: asked of scontrol, else SLURM's own default of 1001")
    a = ap.parse_args()

    instance_root = a.instance_root
    manifest = a.manifest or (instance_root / "problems.tsv")
    if not manifest.exists():
        print(f"[ERROR] {manifest} not found -- run expand_instances_for_benchmark.sh first",
              file=sys.stderr)
        return 1

    if a.out is not None:
        out_path = a.out
    elif a.folder:
        out_path = a.output_root / a.folder / "units" / "worklist.tsv"
    else:
        print("[ERROR] pass --folder (recommended) or --out", file=sys.stderr)
        return 1

    rows = read_manifest(manifest)
    wanted = set(a.problem) if a.problem else None
    if wanted:
        rows = [r for r in rows if r.get("problem_dir") in wanted]
        missing = wanted - {r.get("problem_dir") for r in rows}
        if missing:
            print(f"[ERROR] --problem: not in {manifest}: {sorted(missing)}", file=sys.stderr)
            return 1
    if not rows:
        print(f"[ERROR] no problems selected from {manifest}", file=sys.stderr)
        return 1

    units = []
    per_family = {}
    for row in rows:
        problem = row["problem_dir"]
        cap = row.get("capacity_level", "")
        problem_dir = instance_root / problem
        if not problem_dir.is_dir():
            print(f"[ERROR] {problem_dir} is in the manifest but not on disk", file=sys.stderr)
            return 1
        instances = [p.name for p in sorted(p for p in problem_dir.iterdir() if p.is_dir())]
        declared = row.get("n_instances")
        if declared and declared.isdigit() and int(declared) != len(instances):
            print(f"[WARN] {problem}: manifest says {declared} instances, found "
                  f"{len(instances)} -- the manifest is stale", file=sys.stderr)
        systems = systems_for(problem_dir, cap, a.run_mip)
        family = families.family_name(cap, a.run_mip)
        per_family[family] = per_family.get(family, 0) + len(instances) * len(systems)
        for instance in instances:
            for system in systems:
                units.append((problem, row.get("time_granularity", ""), row.get("region", ""),
                              cap, family, instance, system))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for unit_id, unit in enumerate(units, start=1):
            fh.write("\t".join([str(unit_id), *unit]) + "\n")
    tmp.replace(out_path)

    total = len(units)
    chunk = max(1, a.chunk)
    tasks = math.ceil(total / chunk)
    max_array = a.max_array_size or detect_max_array_size()
    min_chunk = math.ceil(total / max_array) if max_array else chunk
    walltime = chunk * (a.time_limit + PER_UNIT_OVERHEAD_S) + PER_TASK_OVERHEAD_S

    print(f"worklist:  {out_path}")
    print(f"problems:  {len(rows)}   RUN_MIP={a.run_mip}")
    for family, n in sorted(per_family.items()):
        print(f"           {n:>9,} units  {family}")
    print(f"units:     {total:,}   (one solver run each, bounded by {a.time_limit} s)")
    print()
    print(f"  CHUNK={chunk}  ->  {tasks:,} array tasks, "
          f"up to {hms(walltime)} each ({chunk} x {a.time_limit}s + overhead)")
    print(f"  MaxArraySize on this cluster: {max_array:,}"
          f"{'  (assumed -- scontrol not available here)' if not shutil.which('scontrol') else ''}")

    # Waves, not a bigger CHUNK. tasks > MaxArraySize does NOT mean the jobs have to get longer:
    # the same worklist can be submitted as several arrays, each starting where the last ended.
    # Raising CHUNK instead would undo the point of the exercise -- at MaxArraySize=1001 a 59,000
    # unit campaign would need CHUNK >= 59, i.e. ~30-hour tasks.
    waves = math.ceil(tasks / max_array) if max_array else 1
    tasks_per_wave = min(tasks, max_array)
    if waves > 1:
        print(f"  {tasks:,} tasks > MaxArraySize, so submit {waves} WAVES of at most "
              f"{tasks_per_wave:,} tasks. The jobs stay {hms(walltime)}; only the number of "
              f"sbatch calls goes up.")
        print(f"  (Raising CHUNK instead would work too -- CHUNK >= {min_chunk} fits in one "
              f"array -- but each task would then run up to "
              f"{hms(min_chunk * (a.time_limit + PER_UNIT_OVERHEAD_S) + PER_TASK_OVERHEAD_S)}.)")
    else:
        print(f"  [OK] {tasks:,} <= {max_array:,}: one array is enough.")
    print()
    folder = a.folder or "<FOLDER>"
    print("Submit with:")
    for wave in range(waves):
        offset = wave * tasks_per_wave * chunk
        n = min(tasks_per_wave, tasks - wave * tasks_per_wave)
        print(f"  sbatch -t {hms(walltime)} \\")
        print(f"         --export=ALL,FOLDER={folder},CHUNK={chunk},RUN_MIP={a.run_mip}"
              f"{f',UNIT_OFFSET={offset}' if waves > 1 else ''} \\")
        print(f"         --array=1-{n}%40 run_benchmark_units.slurm"
              f"{f'        # units {offset + 1:,}..{min(offset + n * chunk, total):,}' if waves > 1 else ''}")
        if wave == 2 and waves > 4:
            print(f"  ... {waves - 3} more waves, UNIT_OFFSET stepping by "
                  f"{tasks_per_wave * chunk:,} up to {(waves - 1) * tasks_per_wave * chunk:,}")
            break
    print()
    print("Then, when the array has drained:")
    print(f"  ./merge_benchmark_shards.py --folder {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
