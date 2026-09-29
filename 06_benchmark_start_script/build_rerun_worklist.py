#!/usr/bin/env python3
"""Derive the worklist of a rerun campaign from a finished campaign's worklist.

The rerun gets its own folder and a worklist in the same TSV format as build_worklist.py writes,
units renumbered from 1, so run_benchmark_units.slurm and merge_benchmark_shards.py handle it like
any campaign. Every kept row is copied from the source worklist, so problem, time granularity,
region, capacity level and family are exactly the campaign's.

Two modes:

  --systems A,B [--only-finished]   the source's units of these systems; with --only-finished only
                                    those whose run FINISHED in the source campaign, read from the
                                    merged output/<source>/output_<PROBLEM>/progress.jsonl (or,
                                    with --outcome-from individual, from individual_outputs/).
  --as-system S                     one unit per (problem, instance) of the source, for system S --
                                    a new method over exactly the campaign's instances.

    # R-DC: regenerate the matrices 03_DELAY and 03A_CASA lost to their shared results folder
    ./build_rerun_worklist.py --source-folder 20260918_V2 --folder 20260930_V2_RERUN_DC \\
        --systems 03_DELAY,03A_CASA --only-finished
    # R-SEQ: 0_Sequential on every instance of the campaign
    ./build_rerun_worklist.py --source-folder 20260918_V2 --folder 20260930_V2_SEQ \\
        --as-system 0_Sequential

The system(s) must be ones run_benchmark_units.slurm can run for each problem: checked here with
the same benchmark_families.py flags and build_system_config() the unit will use.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_families as families                                        # noqa: E402
from build_worklist import (COLUMNS, DEFAULT_MAX_ARRAY_SIZE, PER_TASK_OVERHEAD_S,  # noqa: E402
                            PER_UNIT_OVERHEAD_S, detect_max_array_size, hms)
from start_benchmark_caller import build_arg_parser, build_system_config     # noqa: E402

OUTCOMES = {"": "ok", "T": "TIMEOUT", "M": "MEMOUT", "E": "ERROR", "P": "UNPARSED"}


def read_worklist(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if tuple(header) != COLUMNS:
            raise SystemExit(f"[ERROR] {path}: header {header} is not {list(COLUMNS)}")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in fh if line.strip()]


def finished_from_progress(problem_dir: Path, systems: Set[str]) -> Dict[Tuple[str, str], str]:
    """(instance, system) -> outcome, from the merged progress.jsonl; the last row of a unit wins.

    The file carries every result line of every run and can be hundreds of MB, so only lines that
    name one of `systems` are parsed.
    """
    outcomes: Dict[Tuple[str, str], str] = {}
    path = problem_dir / "progress.jsonl"
    if not path.is_file():
        return outcomes
    needles = [f'"system": "{s}"' for s in systems]
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not any(n in line for n in needles):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("system") in systems:
                outcomes[(row.get("instance"), row["system"])] = row.get("outcome", "unknown")
    return outcomes


def finished_from_individual(problem_dir: Path, instance: str, system: str) -> str:
    path = problem_dir / "individual_outputs" / f"{instance}_{system}.json"
    try:
        lines = json.loads(path.read_text(encoding="utf-8")).get("object")
    except (OSError, json.JSONDecodeError, AttributeError):
        return "missing"
    if isinstance(lines, list) and lines and isinstance(lines[-1], dict):
        return OUTCOMES.get(lines[-1].get("ERROR"), "unknown")
    return OUTCOMES.get(lines, "unknown") if isinstance(lines, str) else "unknown"


def runnable(system: str, capacity_level: str, run_mip: str) -> bool:
    """Whether a unit of `system` on a problem of this capacity level would find its system."""
    flags = families.experiment_flags(capacity_level, run_mip, system=system)
    args = build_arg_parser().parse_args(["problem", *flags])
    keys = [s["key"] for s in build_system_config(Path(__file__).resolve().parent, Path("output"),
                                                  "rerun", args)]
    return system in keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-folder", required=True, help="the finished campaign, e.g. 20260918_V2")
    ap.add_argument("--source-worklist", type=Path, default=None,
                    help="default: <output-root>/<source-folder>/units/worklist.tsv")
    ap.add_argument("--folder", required=True, help="the rerun's own folder, e.g. 20260930_V2_SEQ")
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--systems", default=None, help="comma-separated system keys to keep")
    mode.add_argument("--as-system", default=None, help="one unit per (problem, instance) for this system")
    ap.add_argument("--only-finished", action="store_true",
                    help="with --systems: keep only units whose run finished in the source campaign")
    ap.add_argument("--outcome-from", default="progress", choices=["progress", "individual"],
                    help="where --only-finished reads the outcome (default: the merged progress.jsonl)")
    ap.add_argument("--run-mip", default="no", choices=list(families.RUN_MIP_MODES),
                    help="the RUN_MIP the rerun is submitted with (checked, not written)")
    ap.add_argument("--time-limit", type=int, default=1800, help="for the walltime printed at the end")
    ap.add_argument("--max-array-size", type=int, default=None)
    a = ap.parse_args()

    source = a.source_worklist or (a.output_root / a.source_folder / "units" / "worklist.tsv")
    if not source.is_file():
        print(f"[ERROR] {source} not found", file=sys.stderr)
        return 1
    rows = read_worklist(source)
    out_path = a.output_root / a.folder / "units" / "worklist.tsv"
    if out_path.exists():
        print(f"[ERROR] {out_path} exists; a rerun folder gets its worklist once", file=sys.stderr)
        return 1

    kept: List[Dict[str, str]] = []
    dropped: Dict[str, int] = {}
    if a.as_system:
        seen = set()
        for row in rows:
            key = (row["problem_dir"], row["instance"])
            if key not in seen:
                seen.add(key)
                kept.append(dict(row, system=a.as_system))
        wanted = {a.as_system}
    else:
        wanted = {s.strip() for s in a.systems.split(",") if s.strip()}
        known = {row["system"] for row in rows}
        unknown = wanted - known
        if unknown:
            print(f"[ERROR] not in the source worklist: {sorted(unknown)}", file=sys.stderr)
            return 1
        candidates = [row for row in rows if row["system"] in wanted]
        if not a.only_finished:
            kept = candidates
        else:
            cache: Dict[str, Dict[Tuple[str, str], str]] = {}
            for row in candidates:
                problem_dir = a.output_root / a.source_folder / f"output_{row['problem_dir']}"
                if a.outcome_from == "progress":
                    if row["problem_dir"] not in cache:
                        cache[row["problem_dir"]] = finished_from_progress(problem_dir, wanted)
                    outcome = cache[row["problem_dir"]].get((row["instance"], row["system"]), "missing")
                else:
                    outcome = finished_from_individual(problem_dir, row["instance"], row["system"])
                if outcome == "ok":
                    kept.append(row)
                else:
                    dropped[outcome] = dropped.get(outcome, 0) + 1

    if not kept:
        print("[ERROR] no unit left", file=sys.stderr)
        return 1

    # every kept unit must find its system when run_benchmark_units.slurm runs it
    checked = {}
    for row in kept:
        key = (row["system"], row["capacity_level"])
        if key not in checked:
            checked[key] = runnable(row["system"], row["capacity_level"], a.run_mip)
        if not checked[key]:
            print(f"[ERROR] {row['system']} is not enabled for capacity level {row['capacity_level']} "
                  f"with RUN_MIP={a.run_mip}", file=sys.stderr)
            return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tsv.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for unit_id, row in enumerate(kept, start=1):
            fh.write("\t".join([str(unit_id)] + [row[c] for c in COLUMNS[1:]]) + "\n")
    tmp.replace(out_path)

    per_system: Dict[str, int] = {}
    per_family: Dict[str, int] = {}
    for row in kept:
        per_system[row["system"]] = per_system.get(row["system"], 0) + 1
        per_family[row["family"]] = per_family.get(row["family"], 0) + 1
    note = {
        "written_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_worklist": str(source), "source_units": len(rows),
        "mode": f"as-system {a.as_system}" if a.as_system else
                f"systems {sorted(wanted)}" + (f", only finished (from {a.outcome_from})" if a.only_finished else ""),
        "units": len(kept), "per_system": per_system, "dropped_by_source_outcome": dropped,
    }
    (out_path.parent / "worklist_source.json").write_text(json.dumps(note, indent=2), encoding="utf-8")

    print(f"worklist:  {out_path}  ({len(kept):,} units from {len(rows):,} in {source})")
    for system, n in sorted(per_system.items()):
        print(f"           {n:>9,} units  {system}")
    for family, n in sorted(per_family.items()):
        print(f"           {n:>9,} units  {family}")
    if dropped:
        print("dropped:   " + ", ".join(f"{n:,} {o}" for o, n in sorted(dropped.items()))
              + " (not finished in the source campaign)")

    max_array = a.max_array_size or detect_max_array_size() or DEFAULT_MAX_ARRAY_SIZE
    walltime = a.time_limit + PER_UNIT_OVERHEAD_S + PER_TASK_OVERHEAD_S
    waves = math.ceil(len(kept) / max_array)
    print(f"\nOne unit per task (CHUNK=1): {len(kept):,} tasks of up to {hms(walltime)}; "
          f"{waves} array(s) at MaxArraySize {max_array:,}.")
    for wave in range(waves):
        offset = wave * max_array
        n = min(max_array, len(kept) - offset)
        print(f"  sbatch -t {hms(walltime)} --export=ALL,FOLDER={a.folder},CHUNK=1,RUN_MIP={a.run_mip}"
              f"{f',UNIT_OFFSET={offset}' if waves > 1 else ''} --array=1-{n}%40 run_benchmark_units.slurm")
    print(f"\nThen: ./merge_benchmark_shards.py --folder {a.folder}"
          f"{' --allow-incomplete' if a.only_finished else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
