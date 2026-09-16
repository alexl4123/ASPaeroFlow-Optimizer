#!/usr/bin/env python3
"""Concatenate the per-unit shards into the CSVs the monolithic benchmark writes.

run_benchmark_units.slurm runs every (problem, instance, system) unit in its own job and its own
output directory, so that hundreds of concurrent small jobs share no mutable state. This turns
that back into what run_all_benchmarks.slurm produces:

    output/<FOLDER>/output_<PROBLEM>/execution_time.csv, ram_usage.csv, overload.csv,
                                     arrival-delay.csv, ..., error.csv
                                     individual_outputs/<INSTANCE>_<SYSTEM>.json
                                     hotstart_state.json, progress.jsonl, solver_outputs/
    output/<FOLDER>/run_provenance_<PROBLEM>.txt

Same files, same paths, same rows in the same order, same columns in the same order.

WHY THIS CALLS THE WRITER INSTEAD OF WRITING CSVs

start_benchmark_caller.py's sol_value_to_rows() discovers each metric CSV's header WHILE it walks
the instances: a system that does not report OVERLOAD on the first instance is absent from that
header, and if it reports OVERLOAD on a later instance the column appears from there on, so rows
can differ in length. That is the current output, quirk included. Re-implementing the writer would
mean re-implementing the quirk and hoping; this imports write_result_csvs() and calls it, so
"identical" is structural rather than a matter of care.

WHAT A SHARD CONTAINS

Each unit ran with --hot-start, so its directory holds hotstart_state.json, whose one record
carries exactly what the writer needs: execution_time, ram_usage and solution_value (the full list
of the solver's JSON output lines). That file IS the shard payload; no new format was invented.

IDEMPOTENT AND RE-RUNNABLE

Merging is a pure function of the shards plus whatever was merged before, so it can be run while
the array is still draining, again when it finishes, and again after the stragglers have been
resubmitted. Each run rewrites the CSVs from scratch. Records already merged are kept even if
their shard has since been pruned, because the merged hotstart_state.json is read back in as a
base -- which also means a merged folder can be handed straight back to run_all_benchmarks.slurm
with --hot-start and it will skip the finished work.

MISSING UNITS

A unit whose job died leaves no record. Rather than write a CSV with a hole in it, this reports
the gap and, with --write-missing-worklist, writes a worklist of exactly the missing units to
resubmit. A problem is only written when its rectangle is complete; --allow-incomplete relaxes
that by dropping systems that are not complete across all instances, which keeps the CSVs
rectangular and reproduces what the monolithic form writes when a system is switched off.

    ./merge_benchmark_shards.py --folder 20260916_V2
    ./merge_benchmark_shards.py --folder 20260916_V2 --write-missing-worklist retry.tsv
    ./merge_benchmark_shards.py --folder 20260916_V2 --allow-incomplete --prune-shards
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_worklist import COLUMNS                                    # noqa: E402
from start_benchmark_caller import write_result_csvs                  # noqa: E402

#: Statuses a unit can be in after the array has run.
DONE, MISSING, SKIPPED = "done", "missing", "skipped"

#: Keys the per-task provenance files carry that belong in the per-problem provenance file, so
#: that a merged folder says the same things about itself as a monolithic one.
PROVENANCE_KEYS = ("time_limit_s", "memory_limit_gib", "optimizer_commit", "optimizer_dirty",
                   "gurobi_licence", "run_mip", "results_format", "wandb_enabled")


def read_worklist(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != list(COLUMNS):
            raise SystemExit(f"[ERROR] {path} has columns {header}, expected {list(COLUMNS)}")
        for line in fh:
            line = line.rstrip("\n")
            if line:
                rows.append(dict(zip(header, line.split("\t"))))
    return rows


def shard_dir(units_root: Path, unit: dict) -> Path:
    return units_root / "shards" / unit["problem_dir"] / unit["instance"] / unit["system"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def unit_record(shard: Path, instance: str, system: str) -> Tuple[str, dict | None]:
    """(status, record) for one unit, read from its shard.

    The record comes from hotstart_state.json, but the solver's output lines are taken from the
    shard's individual_outputs/ copy when it is there. Both hold the same values; they differ in
    the ORDER OF THE JSON KEYS, because save_hotstart_state_atomic() dumps with sort_keys=True and
    that sorts the nested output dicts too. Reading them back and writing them out again would
    then give merged individual_outputs/ files whose keys are alphabetical where a monolithic
    run's are in the order the solver printed them -- same data, different bytes. Preferring the
    shard's own copy keeps the bytes identical as well. (The CSVs never noticed either way: they
    look values up by key.)
    """
    state = load_json(shard / "hotstart_state.json")
    if isinstance(state, dict):
        rec = state.get("records", {}).get(instance, {}).get(system)
        if rec is not None:
            raw = load_json(shard / "individual_outputs" / f"{instance}_{system}.json")
            if isinstance(raw, dict) and "object" in raw:
                rec = dict(rec, solution_value=raw["object"])
            return DONE, rec
    marker = load_json(shard / "unit.json")
    if isinstance(marker, dict) and marker.get("status") == "skipped_no_licence":
        return SKIPPED, None
    return MISSING, None


def progress_row(shard: Path, instance: str, system: str, rec: dict) -> dict:
    """The unit's own progress line, or one rebuilt from its record if the shard is gone."""
    prog = shard / "progress.jsonl"
    if prog.exists():
        for line in reversed(prog.read_text(errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("instance") == instance and row.get("system") == system:
                return row
    sol = rec.get("solution_value")
    err = sol[-1].get("ERROR") if isinstance(sol, list) and sol and isinstance(sol[-1], dict) else None
    outcome = {"T": "TIMEOUT", "M": "MEMOUT", "E": "ERROR", "P": "UNPARSED"}.get(err, "ok")
    return {"ts": rec.get("timestamp_utc"), "instance": instance, "system": system,
            "outcome": outcome, "runtime_s": rec.get("execution_time"),
            "ram_mb": rec.get("ram_usage"), "objective": sol, "error_code": err,
            "reused": False}


def relocate_solver_outputs(shard: Path, target: Path, mode: str) -> int:
    """Move (or copy) a shard's solver_outputs/<SYSTEM>/<INSTANCE>/ tree next to the CSVs.

    The per-run result matrices are the bulk of the output -- ~11 GB compressed across the full
    grid -- so the default is a rename, which costs nothing on one filesystem. The leaf is unique
    per (system, instance), so nothing overwrites anything.
    """
    src = shard / "solver_outputs"
    if mode == "leave" or not src.is_dir():
        return 0
    moved = 0
    for system_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            dst = target / "solver_outputs" / system_dir.name / run_dir.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            if mode == "copy":
                shutil.copytree(run_dir, dst)
            else:
                shutil.move(str(run_dir), str(dst))
            moved += 1
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, help="output namespace, as FOLDER= at submission")
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--worklist", type=Path, default=None,
                    help="default: output/<FOLDER>/units/worklist.tsv")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a problem even if units are missing, dropping any SYSTEM that is "
                         "not complete across all of its instances so the CSVs stay rectangular")
    ap.add_argument("--write-missing-worklist", type=Path, default=None, metavar="TSV",
                    help="write the missing units as a worklist of their own, renumbered from 1, "
                         "ready to resubmit with WORKLIST=<TSV> and CHUNK=1")
    ap.add_argument("--solver-outputs", default="move", choices=["move", "copy", "leave"],
                    help="what to do with each shard's solver_outputs/ tree (default: move)")
    ap.add_argument("--prune-shards", action="store_true",
                    help="delete the shard directories of every problem written in full. The "
                         "merged hotstart_state.json keeps the results, so a later merge still "
                         "works, but the per-unit logs and CSVs are gone.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    folder_root = a.output_root / a.folder
    units_root = folder_root / "units"
    worklist_path = a.worklist or (units_root / "worklist.tsv")
    if not worklist_path.exists():
        print(f"[ERROR] {worklist_path} not found -- was this campaign built with "
              f"build_worklist.py?", file=sys.stderr)
        return 1

    units = read_worklist(worklist_path)

    # Canonical order: first appearance in the worklist, which build_worklist.py emitted as
    # manifest order x sorted(instance dirs) x build_system_config() order -- the same rows and
    # the same columns the monolithic run produces.
    problems: Dict[str, dict] = {}
    for unit in units:
        problem = problems.setdefault(unit["problem_dir"], {
            "instances": [], "systems": [], "units": [], "row": unit})
        if unit["instance"] not in problem["instances"]:
            problem["instances"].append(unit["instance"])
        if unit["system"] not in problem["systems"]:
            problem["systems"].append(unit["system"])
        problem["units"].append(unit)

    missing_units: List[dict] = []
    n_done = n_skipped = n_moved = 0
    written: List[str] = []
    incomplete: List[Tuple[str, int, List[str]]] = []

    for problem_name, problem in problems.items():
        target = folder_root / f"output_{problem_name}"

        # Anything merged earlier is the base, so a folder whose shards were pruned still merges.
        base_state = load_json(target / "hotstart_state.json") or {}
        base_records = base_state.get("records", {}) if isinstance(base_state, dict) else {}

        records: Dict[str, Dict[str, dict]] = {}
        progress: Dict[Tuple[str, str], dict] = {}
        status: Dict[Tuple[str, str], str] = {}
        for unit in problem["units"]:
            instance, system = unit["instance"], unit["system"]
            shard = shard_dir(units_root, unit)
            state, rec = unit_record(shard, instance, system)
            if state != DONE:
                rec = base_records.get(instance, {}).get(system)
                if rec is not None:
                    state = DONE
            if state == DONE:
                records.setdefault(instance, {})[system] = rec
                progress[(instance, system)] = progress_row(shard, instance, system, rec)
                n_moved += relocate_solver_outputs(shard, target, a.solver_outputs)
                n_done += 1
            else:
                if state == SKIPPED:
                    n_skipped += 1
                else:
                    missing_units.append(unit)
                status[(instance, system)] = state

        instances = problem["instances"]
        complete = [s for s in problem["systems"]
                    if all(s in records.get(i, {}) for i in instances)]
        gaps = [s for s in problem["systems"] if s not in complete]

        if gaps and not a.allow_incomplete:
            n_gap = sum(1 for i in instances for s in gaps if s not in records.get(i, {}))
            incomplete.append((problem_name, n_gap, gaps))
            continue
        if not complete:
            incomplete.append((problem_name, len(instances) * len(problem["systems"]),
                               problem["systems"]))
            continue

        target.mkdir(parents=True, exist_ok=True)
        exec_time = {i: {s: records[i][s].get("execution_time") for s in complete} for i in instances}
        ram_usage = {i: {s: records[i][s].get("ram_usage") for s in complete} for i in instances}
        sol_value = {i: {s: records[i][s].get("solution_value") for s in complete} for i in instances}
        write_result_csvs(target, instances, complete, exec_time, ram_usage, sol_value)

        # The merged hot-start state: every record found, including systems left out of the CSVs
        # because they were incomplete. A monolithic re-run with --hot-start then skips them.
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        merged_records = {i: dict(base_records.get(i, {})) for i in instances}
        for i, per_system in records.items():
            merged_records.setdefault(i, {}).update(per_system)
        (target / "hotstart_state.json").write_text(json.dumps({
            "schema_version": 1,
            "created_utc": base_state.get("created_utc", now) if isinstance(base_state, dict) else now,
            "updated_utc": now,
            "records": merged_records,
        }, indent=2, sort_keys=True), encoding="utf-8")

        # progress.jsonl, in canonical order, renumbered so benchmark_progress.py reads a merged
        # folder exactly as it reads a monolithic one.
        rows = [progress[(i, s)] for i in instances for s in complete if (i, s) in progress]
        with (target / "progress.jsonl").open("w", encoding="utf-8") as fh:
            for n, row in enumerate(rows, start=1):
                row = dict(row, done=n, total=len(rows))
                fh.write(json.dumps(row) + "\n")

        write_provenance(folder_root, units_root, problem_name, problem["row"],
                         len(instances), complete, gaps)
        written.append(problem_name)

        if a.prune_shards and not gaps:
            for unit in problem["units"]:
                shutil.rmtree(shard_dir(units_root, unit), ignore_errors=True)

    # ---- report -------------------------------------------------------------------------
    total = len(units)
    print(f"worklist:  {worklist_path}  ({total:,} units)")
    print(f"merged:    {n_done:,} units into {len(written)} problem folder(s) under {folder_root}")
    if n_skipped:
        print(f"skipped:   {n_skipped:,} unit(s) -- RUN_MIP=auto found no Gurobi licence on that node")
    if n_moved and a.solver_outputs != "leave":
        print(f"matrices:  {n_moved:,} solver_outputs tree(s) {a.solver_outputs}d into place")
    if incomplete:
        print(f"\nNOT WRITTEN -- {len(incomplete)} problem(s) are incomplete "
              f"(re-run with --allow-incomplete to write what is there):")
        for name, n_gap, gaps in incomplete[:20]:
            print(f"  {name:<52} {n_gap:>6} unit(s) short, system(s): {', '.join(gaps[:6])}"
                  f"{' ...' if len(gaps) > 6 else ''}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")
    if missing_units:
        print(f"\nMISSING -- {len(missing_units):,} unit(s) produced no result. First 15:")
        for unit in missing_units[:15]:
            print(f"  unit {unit['unit_id']:>7}  {unit['problem_dir']:<40} "
                  f"{unit['instance']:<22} {unit['system']}")
        if a.write_missing_worklist:
            path = a.write_missing_worklist
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                fh.write("\t".join(COLUMNS) + "\n")
                for new_id, unit in enumerate(missing_units, start=1):
                    fh.write("\t".join([str(new_id)] + [unit[c] for c in COLUMNS[1:]]) + "\n")
            print(f"\n  wrote {path} -- {len(missing_units):,} units, renumbered from 1. Resubmit:")
            print(f"    sbatch --export=ALL,FOLDER={a.folder},CHUNK=1,WORKLIST={path} \\")
            print(f"           --array=1-{len(missing_units)}%40 run_benchmark_units.slurm")
        else:
            print("  pass --write-missing-worklist <TSV> to get a resubmittable worklist of these")
    if not incomplete and not missing_units:
        print("\nComplete: every unit in the worklist has a result.")
    return 0


def write_provenance(folder_root: Path, units_root: Path, problem: str, row: dict,
                     n_instances: int, systems: List[str], gaps: List[str]) -> None:
    """The per-problem provenance file the monolithic form writes, with the same keys.

    The per-unit path has no single host, job id or start time for a problem -- its units ran on
    many nodes over many jobs -- so those keys carry the aggregate and point at
    units/provenance/, where each task wrote its own. The keys themselves are unchanged because
    things downstream read them; benchmark_progress.py, for one, looks for "MIP ONLY" in
    experiment_family.
    """
    # Settings the tasks recorded. Where the tasks disagree -- a resubmission with a different
    # TIME_LIMIT, say -- every value is listed, because silently reporting one of them would
    # misdescribe half the numbers in the CSVs.
    harvested: Dict[str, List[str]] = {}
    for task_file in sorted((units_root / "provenance").glob("task_*.txt")):
        for line in task_file.read_text(errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key in PROVENANCE_KEYS and value not in harvested.setdefault(key, []):
                harvested[key].append(value)

    def recorded(key: str) -> str:
        values = harvested.get(key) or ["unknown"]
        return values[0] if len(values) == 1 else " | ".join(values)

    lines = [
        f"problem={problem}",
        f"region={row.get('region', '')}",
        f"capacity_level={row.get('capacity_level', '')}",
        f"time_granularity={row.get('time_granularity', '')}",
        f"time_limit_s={recorded('time_limit_s')}",
        f"memory_limit_gib={recorded('memory_limit_gib')}",
        f"optimizer_commit={recorded('optimizer_commit')}",
        f"optimizer_dirty={recorded('optimizer_dirty')}",
        f"slurm_job=many (per-unit array; see {units_root}/provenance/)",
        f"host=many (per-unit array; see {units_root}/provenance/)",
        f"merged={datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        f"gurobi_licence={recorded('gurobi_licence')}",
        f"mip={recorded('run_mip')}",
        f"results_format={recorded('results_format')}",
        f"wandb_enabled={recorded('wandb_enabled')}",
        f"experiment_family={row.get('family', '')}",
        f"n_instances={n_instances}",
        f"systems={','.join(systems)}",
    ]
    if gaps:
        lines.append(f"systems_dropped_incomplete={','.join(gaps)}")
    (folder_root / f"run_provenance_{problem}.txt").write_text("\n".join(lines) + "\n",
                                                               encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
