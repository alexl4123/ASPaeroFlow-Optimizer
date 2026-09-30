#!/usr/bin/env python3
"""Pack the validated solution matrices of a benchmark campaign into zip archives for Zenodo.

Three steps, so that the slow part runs as many small SLURM jobs:

  plan   one small job. Reads the validator's per-run CSVs (validation_<PROBLEM>.csv), selects
         the runs whose matrices are published, stats their files, and assigns them to archives:
         one group per family (large, small) and licence (CC-BY-4.0: grid regions; GPL-2.0-or-later:
         regions built on X-Plane waypoints), each group cut into archives of at most
         --max-archive-gb in sorted path order. Writes <plan-dir>/plan.json,
         <plan-dir>/archives/<ARCHIVE>.runs.json, selected_runs.csv, unpublished_finished_runs.csv,
         and prints the plan. --selection-only stops before the stat (no matrices needed).
  pack   one SLURM array task per archive (--index = the task id). Writes <out-dir>/<ARCHIVE>.zip
         and verifies it: zipfile.testzip() (CRC-32 of every member), then the sha256 and size of
         every member against MANIFEST.csv. Only then are <ARCHIVE>.zip, <ARCHIVE>.MANIFEST.csv and
         <ARCHIVE>.zip.sha256 put in place; a failed task leaves <ARCHIVE>.zip.partial.
  check  seconds, after the array. Every selected run is in exactly one archive and no other run
         is in any; every archive is present and within the size limit. Writes <out-dir>/SHA256SUMS.

WHICH RUNS (from the validation CSVs, nothing is guessed)

  A validation row is published if
    - its system is published (--systems) and its folder is the one the system is taken from
      (--take; V2 default: 03_DELAY and 03A_CASA from 20260930_V2_RERUN_DC, whose matrices replace
      the campaign's, where 03A_CASA overwrote 03_DELAY's; 0_Sequential from 20260930_V2_SEQ;
      every other system from 20260918_V2 and 20260918_V2_MIP),
    - the run finished: outcome ok, and for a row validated in rerun mode campaign_outcome ok too,
    - status VALID and matrix_owner equal to the system.
  A (problem, instance, system) selected twice is an error. Every run that finished but is not
  selected is listed in unpublished_finished_runs.csv with the reason of each of its rows.

WHICH FILES OF A RUN (exactly what validate_solutions.py reads)

  converted_navpoint_matrix, converted_instance_matrix, navaid_sector_time_assignment,
  capacity_time_matrix (the first of .csv.gz, .csv, .npz that exists, as the validator picks it),
  manifest.json, and result_line.json: the solver's last result line, the claims the validator
  checked the matrix against. The line is not stored in the run folder; it is read from
  individual_outputs/ (or the unit shard, or hotstart_state.json) of the results folder, and for
  a row validated in rerun mode from the campaign the rerun was judged against (--claims-from),
  and it must equal the claimed_* values of the validation row. Anything else in a run folder is
  left out and counted in plan.json.

ARCHIVE LAYOUT

  <ARCHIVE>/README.md                                 licence, contents, how to validate
  <ARCHIVE>/MANIFEST.csv                              problem, instance, system, file, bytes, sha256
  <ARCHIVE>/<PROBLEM>/<SYSTEM>/<INSTANCE>/<file>      the run files above
  validate_solutions.py --problem-dir <ARCHIVE>/<PROBLEM> --instance-root <extracted instances>
  validates an unzipped archive as it is.

  Members are written with a fixed timestamp (the plan's date) and fixed permissions, stored
  without recompression where already compressed (.gz, .npz), deflated otherwise, so that packing
  the same plan twice gives the same bytes.

USAGE (from 06_benchmark_start_script/)

  ./pack_solution_matrices.py plan --plan-dir P \\
      --validation validation_20260918 --validation validation_reruns/20260930_V2_RERUN_DC \\
      --results-folder 20260918_V2=$O/20260918_V2 --results-folder 20260918_V2_MIP=$O/20260918_V2_MIP \\
      --results-folder 20260930_V2_RERUN_DC=output/20260930_V2_RERUN_DC \\
      --claims-from 20260930_V2_RERUN_DC=20260918_V2 --instance-root $OPT/05_instances
  sbatch --export=ALL,PLAN_DIR=P,OUT=Z --array=1-<archives> run_pack_solution_matrices.slurm
  ./pack_solution_matrices.py check --plan-dir P --out-dir Z
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_solutions as vs  # noqa: E402

# ---------------------------------------------------------------------------------------------
# What is published, and under which licence
# ---------------------------------------------------------------------------------------------

CC_BY = "CC-BY-4.0"
GPL = "GPL-2.0-or-later"

#: Regions of the large-scaling dataset and the licence of their instances (LICENSING.md of the
#: instance record): synthetic grid navpoints are CC-BY-4.0, graphs built on X-Plane waypoints
#: GPL-2.0-or-later. A solution matrix carries the licence of its instance.
LARGE_REGIONS: Dict[str, str] = {
    "00-0-CENTRAL-EUROPE-7x7": CC_BY, "01-0-USA-EAST-COAST-20x10": CC_BY,
    "02-0-MAJOR-EUROPE-40x20": CC_BY, "03-0-EAST-ASIA-40x40": CC_BY,
    "04-0-DACH": GPL, "05-0-EUROPE": GPL, "06-0-USA-MAINLAND": GPL, "08-0-CENTRAL-EUROPE": GPL,
}

#: Every problem of the small-scaling dataset starts with this; all its regions are grids.
SMALL_PREFIX = "30-"

#: The instance archives a reader validates against (README of each matrix archive).
INSTANCE_RECORDS = {
    "large": ("large-scaling", "10.5281/zenodo.23039645", "experiment_data_V2_large_scaling_TG<g>.zip"),
    "small": ("small-scaling", "10.5281/zenodo.23039835", "experiment_data_V2_small_scaling.zip"),
}

#: Where each system's published matrices come from (V2). 03A_CASA wrote into 03_DELAY's folder
#: in the campaign, so both come from the rerun R-DC.
DEFAULT_TAKE: Tuple[str, ...] = (
    "03_DELAY,03A_CASA=20260930_V2_RERUN_DC",
    "0_Sequential=20260930_V2_SEQ",
    "*=20260918_V2,20260918_V2_MIP",
)

MANIFEST_FILE = "manifest.json"
RESULT_LINE_FILE = vs.RESULT_LINE_FILE

#: The run files a reader needs (besides result_line.json), in validator order of preference.
MATRIX_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    name: tuple(f"{name}{ext}" for ext in vs.MATRIX_EXTENSIONS) for name in vs.MATRIX_NAMES}

#: Summary files the validator writes next to the per-problem CSVs; never read as runs.
VALIDATION_SUMMARY_FILES = {"validation_summary.csv", "validation_flagged_runs.csv",
                            "validation_rerun_deviations.csv"}

#: Columns kept from a validation row.
ROW_COLUMNS = ("folder", "problem", "instance", "system", "outcome", "status", "failed_checks",
               "matrix_dir", "matrix_owner", "campaign_outcome", "claimed_computation_finished"
               ) + tuple(f"claimed_{m}" for m in vs.METRICS)

GB = 10 ** 9                            # Zenodo counts decimal gigabytes
RESULT_LINE_ESTIMATE = 4096             # bytes; a result line is well under 2 KB
ARCHIVE_FIXED_ESTIMATE = 64 * 1024      # README.md, MANIFEST header, end of central directory
ZIP64_LIMIT = (1 << 31) - 1

MANIFEST_COLUMNS = ("problem", "instance", "system", "file", "bytes", "sha256")


class PlanError(Exception):
    """The plan cannot be made from these inputs."""


class PackError(Exception):
    """An archive could not be written or did not verify."""


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------

def classify(problem: str) -> Tuple[str, str]:
    """(family, licence) of a problem; a region this table does not know is an error."""
    if problem.startswith(SMALL_PREFIX):
        return "small", CC_BY
    for region, licence in LARGE_REGIONS.items():
        if problem.startswith(region + "-"):
            return "large", licence
    raise PlanError(f"{problem}: unknown region (not in LARGE_REGIONS, not a small-scaling problem)")


def parse_take(specs: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """['SYS1,SYS2=FOLDER1,FOLDER2', '*=...'] -> {system: folders}; '*' covers every other system."""
    rules: Dict[str, Tuple[str, ...]] = {}
    for spec in specs:
        if "=" not in spec:
            raise PlanError(f"--take {spec!r}: expected SYSTEMS=FOLDERS")
        systems, folders = spec.split("=", 1)
        sources = tuple(f.strip() for f in folders.split(",") if f.strip())
        if not sources:
            raise PlanError(f"--take {spec!r}: no folder")
        for system in (s.strip() for s in systems.split(",")):
            if not system:
                continue
            if system in rules:
                raise PlanError(f"--take names {system} twice")
            rules[system] = sources
    return rules


def source_folders(rules: Dict[str, Tuple[str, ...]], system: str) -> Tuple[str, ...]:
    return rules.get(system, rules.get("*", ()))


def is_rerun_row(row: dict) -> bool:
    """A row validated in rerun mode (--campaign-folder) carries the campaign's outcome."""
    return bool(row.get("campaign_outcome"))


def row_finished(row: dict) -> bool:
    """The run finished in the campaign: the campaign's outcome for a rerun row, its own otherwise."""
    return (row.get("campaign_outcome") if is_rerun_row(row) else row.get("outcome")) == "ok"


def exclusion_reason(row: dict, rules: Dict[str, Tuple[str, ...]], systems: Iterable[str]) -> str:
    """Why this validation row is not published; '' if it is."""
    system = row["system"]
    if system not in systems:
        return "system not published"
    allowed = source_folders(rules, system)
    if row["folder"] not in allowed:
        return "taken from " + ("+".join(allowed) if allowed else "no folder")
    if row.get("outcome") != "ok":
        return f"not finished ({row.get('outcome') or 'no outcome'})"
    if is_rerun_row(row) and row["campaign_outcome"] != "ok":
        return f"not finished in the campaign ({row['campaign_outcome']})"
    if row.get("status") != vs.VALID:
        first = (row.get("failed_checks") or "").split(";")[0].split(":")[0]
        return f"status {row.get('status') or 'missing'}" + (f" ({first})" if first else "")
    if row.get("matrix_owner") != system:
        return f"matrix owner {row.get('matrix_owner') or 'unknown'}"
    return ""


def select_runs(rows: Iterable[dict], rules: Dict[str, Tuple[str, ...]],
                systems: Sequence[str]) -> Tuple[List[dict], Counter, List[Tuple[Tuple[str, str, str], str]]]:
    """(selected rows sorted by archive path, excluded counts by (system, folder, reason),
    finished runs without a selected row with the reasons of their rows)."""
    systems = set(systems)
    selected: Dict[Tuple[str, str, str], dict] = {}
    seen: Dict[Tuple[str, str, str, str], int] = {}
    excluded: Counter = Counter()
    finished: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for row in rows:
        key = (row["problem"], row["instance"], row["system"])
        where = (row["folder"],) + key
        if where in seen:
            raise PlanError(f"{row['folder']}: {key} has two validation rows (a CSV passed twice?)")
        seen[where] = 1
        reason = exclusion_reason(row, rules, systems)
        if reason:
            excluded[(row["system"], row["folder"], reason)] += 1
        elif key in selected:
            raise PlanError(f"{key} selected from {selected[key]['folder']} and {row['folder']}")
        else:
            selected[key] = row
        if row["system"] in systems and row_finished(row):
            finished[key].append(f"{row['folder']}: {reason or 'selected'}")
    unpublished = [(key, "; ".join(notes)) for key, notes in sorted(finished.items())
                   if key not in selected]
    ordered = sorted(selected.values(), key=lambda r: (r["problem"], r["system"], r["instance"]))
    return ordered, excluded, unpublished


def validation_csvs(paths: Sequence[Path]) -> List[Path]:
    """The per-problem validation CSVs: files as given, directories searched recursively."""
    out = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.rglob("validation_*.csv")
                           if p.name not in VALIDATION_SUMMARY_FILES)
            if not found:
                raise PlanError(f"{path}: no validation_*.csv below it")
            out += found
        elif path.is_file():
            out.append(path)
        else:
            raise PlanError(f"{path}: no such file or directory")
    unique = sorted({p.resolve() for p in out})
    return unique


def read_validation_rows(paths: Sequence[Path]) -> List[dict]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = {"folder", "problem", "instance", "system", "status"} - set(reader.fieldnames or ())
            if missing:
                raise PlanError(f"{path}: not a validation CSV (columns {sorted(missing)} missing)")
            for row in reader:
                rows.append({c: row.get(c) or "" for c in ROW_COLUMNS})
    return rows


# ---------------------------------------------------------------------------------------------
# Files of a run, archive sizes, splitting
# ---------------------------------------------------------------------------------------------

def run_files(run_dir: Path) -> Tuple[List[Tuple[str, int]], List[str]]:
    """(files to publish with their sizes, names left out) of one run folder."""
    present = {}
    for entry in os.scandir(run_dir):
        present[entry.name] = entry.stat().st_size if entry.is_file() else None
    keep: List[Tuple[str, int]] = []
    for name, candidates in MATRIX_CANDIDATES.items():
        for candidate in candidates:
            if present.get(candidate) is not None:
                keep.append((candidate, present[candidate]))
                break
    if present.get(MANIFEST_FILE) is not None:
        keep.append((MANIFEST_FILE, present[MANIFEST_FILE]))
    kept = {name for name, _ in keep}
    left_out = sorted(name + ("/" if size is None else "") for name, size in present.items()
                      if name not in kept)
    return sorted(keep), left_out


def member_overhead(path: str) -> int:
    """Upper bound of the zip bytes a member adds beyond its data (headers, zip64 extras, the
    central-directory entry) plus its MANIFEST.csv row."""
    return 200 + 3 * len(path.encode("utf-8")) + 64 + 20


def run_bytes(archive_path_prefix: str, files: Sequence[Tuple[str, int]]) -> int:
    """Estimated bytes one run adds to an archive (never less than the real size)."""
    total = RESULT_LINE_ESTIMATE + member_overhead(f"{archive_path_prefix}/{RESULT_LINE_FILE}")
    for name, size in files:
        total += size + member_overhead(f"{archive_path_prefix}/{name}")
    return total


def split_sizes(sizes: Sequence[int], max_bytes: int) -> List[List[int]]:
    """Cut items (already in their final order) into consecutive parts of at most max_bytes each:
    the fewest parts next-fit allows, then as even as that number of parts allows. An item larger
    than max_bytes gets a part of its own."""
    def next_fit(capacity: int) -> List[List[int]]:
        parts: List[List[int]] = []
        current: List[int] = []
        filled = 0
        for i, size in enumerate(sizes):
            if current and filled + size > capacity:
                parts.append(current)
                current, filled = [], 0
            current.append(i)
            filled += size
        if current:
            parts.append(current)
        return parts

    if not sizes:
        return []
    fewest = next_fit(max_bytes)
    n = len(fewest)
    largest = max(sizes)
    if n <= 1 or largest > max_bytes:
        return fewest
    # next-fit never needs more parts for a larger capacity: find the least capacity that still
    # gives n parts
    lo, hi = max(math.ceil(sum(sizes) / n), largest), max_bytes
    while lo < hi:
        mid = (lo + hi) // 2
        if len(next_fit(mid)) <= n:
            hi = mid
        else:
            lo = mid + 1
    return next_fit(lo)


def archive_name(prefix: str, family: str, licence: str, part: int) -> str:
    return f"{prefix}_{family}_{licence}_{part:02d}"


def git_commit(path: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_pairs(specs: Sequence[str], what: str) -> Dict[str, str]:
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise PlanError(f"{what} {spec!r}: expected NAME=VALUE")
        name, value = spec.split("=", 1)
        if name in out:
            raise PlanError(f"{what} names {name} twice")
        out[name.strip()] = value.strip()
    return out


def licence_of_instance(instance_root: Path, problem: str, instance: str) -> Optional[str]:
    """The licence field of the instance's instance_info.json, or None if it has none."""
    info = vs.instance_dir(instance_root, problem, instance) / "instance_info.json"
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("licence") if isinstance(data, dict) else None
    return str(value) if value else None


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------

def human(n: float) -> str:
    return f"{n / GB:,.2f} GB" if n >= GB else f"{n / 10 ** 6:,.2f} MB"


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def print_selection(selected: List[dict], excluded: Counter, unpublished: list) -> None:
    per = Counter((classify(r["problem"]) + (r["system"], r["folder"])) for r in selected)
    print(f"\nSELECTED: {len(selected):,} runs")
    print(f"  {'family':<6} {'licence':<17} {'system':<28} {'folder':<22} {'runs':>7}")
    for (family, licence, system, folder), n in sorted(per.items()):
        print(f"  {family:<6} {licence:<17} {system:<28} {folder:<22} {n:>7,}")
    print("\nNOT SELECTED (validation rows):")
    print(f"  {'system':<28} {'folder':<22} {'reason':<44} {'rows':>7}")
    for (system, folder, reason), n in sorted(excluded.items()):
        print(f"  {system:<28} {folder:<22} {reason[:44]:<44} {n:>7,}")
    by_system = Counter(key[2] for key, _ in unpublished)
    print(f"\nFINISHED BUT NOT PUBLISHED: {len(unpublished):,} runs"
          + ("" if not unpublished else " (unpublished_finished_runs.csv)"))
    for system, n in sorted(by_system.items()):
        example = next(note for key, note in unpublished if key[2] == system)
        print(f"  {system:<28} {n:>7,}   e.g. {example[:90]}")


def cmd_plan(a: argparse.Namespace) -> int:
    plan_dir: Path = a.plan_dir
    plan_dir.mkdir(parents=True, exist_ok=True)
    rules = parse_take(a.take or DEFAULT_TAKE)
    systems = [s.strip() for s in a.systems.split(",")] if a.systems else list(vs.PUBLISHED_SYSTEMS)
    folders = {k: Path(v).resolve() for k, v in parse_pairs(a.results_folder, "--results-folder").items()}
    claims_from = parse_pairs(a.claims_from, "--claims-from")
    max_bytes = int(a.max_archive_gb * GB)
    t0 = time.time()

    csvs = validation_csvs(a.validation)
    rows = read_validation_rows(csvs)
    print(f"[plan] {len(csvs)} validation CSVs, {len(rows):,} rows ({time.time() - t0:.1f} s)")
    print("[plan] taken from: " + "; ".join(f"{s} <- {'+'.join(f)}" for s, f in rules.items()))
    selected, excluded, unpublished = select_runs(rows, rules, systems)
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in selected:
        groups[classify(row["problem"])].append(row)
    print_selection(selected, excluded, unpublished)
    write_csv(plan_dir / "unpublished_finished_runs.csv", ("problem", "instance", "system", "notes"),
              (key + (note,) for key, note in unpublished))
    if a.selection_only:
        write_csv(plan_dir / "selected_runs.csv", ("problem", "instance", "system", "folder"),
                  ((r["problem"], r["instance"], r["system"], r["folder"]) for r in selected))
        print(f"\n[plan] --selection-only: {plan_dir}/selected_runs.csv written, no stat, no plan.json")
        return 0

    # every folder a selected row or its claims live in must be given
    needed = {r["folder"] for r in selected}
    for row in selected:
        if is_rerun_row(row):
            if row["folder"] not in claims_from:
                raise PlanError(f"{row['folder']} was validated in rerun mode; pass --claims-from "
                                f"{row['folder']}=<the campaign folder it reran>")
            needed.add(claims_from[row["folder"]])
    missing = sorted(needed - set(folders))
    if missing:
        raise PlanError("--results-folder missing for " + ", ".join(missing))

    # licence cross-check against the instances' own instance_info.json, one instance per problem
    licence_notes: Counter = Counter()
    if a.instance_root:
        first_instance = {}
        for row in selected:
            first_instance.setdefault(row["problem"], row["instance"])
        for problem, instance in sorted(first_instance.items()):
            declared = licence_of_instance(a.instance_root, problem, instance)
            expected = classify(problem)[1]
            if declared is None:
                licence_notes["instance_info.json without a licence field"] += 1
            elif not declared.startswith(expected):
                raise PlanError(f"{problem}: instance_info.json says {declared!r}, the region table {expected}")
            else:
                licence_notes[f"instance_info.json agrees ({expected})"] += 1

    # stat every selected run
    t1 = time.time()
    errors: List[str] = []
    left_out: Counter = Counter()
    runs_by_group: Dict[Tuple[str, str], List[dict]] = {}
    for gkey in sorted(groups):
        entries = []
        for row in groups[gkey]:
            src = folders[row["folder"]] / f"output_{row['problem']}" / row["matrix_dir"]
            if not row["matrix_dir"] or not src.is_dir():
                errors.append(f"{src}: run folder missing ({row['folder']} {row['instance']} {row['system']})")
                continue
            files, extra = run_files(src)
            names = {n for n, _ in files}
            if not names & set(MATRIX_CANDIDATES["converted_navpoint_matrix"]):
                errors.append(f"{src}: no converted_navpoint_matrix")
                continue
            for name in extra:
                left_out[name] += 1
            prefix = f"{row['problem']}/{row['system']}/{row['instance']}"
            entries.append({
                "problem": row["problem"], "instance": row["instance"], "system": row["system"],
                "folder": row["folder"],
                "claims_folder": claims_from[row["folder"]] if is_rerun_row(row) else row["folder"],
                "src": row["matrix_dir"],
                "files": [[n, s] for n, s in files],
                "claims": {k: row[k] for k in ROW_COLUMNS if k.startswith("claimed_")},
                "bytes": run_bytes(prefix, files),
            })
        runs_by_group[gkey] = entries
    print(f"\n[plan] stat of {sum(len(v) for v in runs_by_group.values()):,} run folders: "
          f"{time.time() - t1:.1f} s")
    if errors:
        (plan_dir / "plan_errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"[ERROR] {len(errors)} selected runs cannot be packed; first: {errors[0]}\n"
              f"        all in {plan_dir}/plan_errors.txt; no plan written", file=sys.stderr)
        return 2

    # archives
    archive_dir = plan_dir / "archives"
    archive_dir.mkdir(exist_ok=True)
    for old in archive_dir.glob("*.runs.json"):
        old.unlink()
    capacity = max_bytes - ARCHIVE_FIXED_ESTIMATE
    archives = []
    selected_rows = []
    for (family, licence), entries in sorted(runs_by_group.items()):
        parts = split_sizes([e["bytes"] for e in entries], capacity)
        for k, idx in enumerate(parts, 1):
            name = archive_name(a.prefix, family, licence, k)
            chunk = [entries[i] for i in idx]
            payload = sum(s for e in chunk for _, s in e["files"])
            estimate = sum(e["bytes"] for e in chunk) + ARCHIVE_FIXED_ESTIMATE
            archives.append({
                "index": len(archives) + 1, "name": name, "family": family, "licence": licence,
                "part": k, "parts": len(parts), "runs": len(chunk),
                "files": sum(len(e["files"]) + 1 for e in chunk) + 2,
                "payload_bytes": payload, "estimated_bytes": estimate,
                "systems": dict(sorted(Counter(e["system"] for e in chunk).items())),
                "problems": len({e["problem"] for e in chunk}),
                "first": f"{chunk[0]['problem']}/{chunk[0]['system']}/{chunk[0]['instance']}",
                "last": f"{chunk[-1]['problem']}/{chunk[-1]['system']}/{chunk[-1]['instance']}",
                "over_limit": estimate > max_bytes,
            })
            write_json(archive_dir / f"{name}.runs.json", chunk)
            selected_rows += [(e["problem"], e["instance"], e["system"], e["folder"], name,
                               sum(s for _, s in e["files"])) for e in chunk]

    n_archives = len(archives)
    total_payload = sum(x["payload_bytes"] for x in archives)
    total_estimate = sum(x["estimated_bytes"] for x in archives)
    fits_files = n_archives + a.other_files <= a.max_files
    within_record = total_estimate <= a.record_limit_gb * GB
    plan = {
        "created": date.today().isoformat(),
        "tool": "pack_solution_matrices.py", "optimizer_commit": git_commit(Path(__file__).resolve().parent),
        "parameters": {"max_archive_gb": a.max_archive_gb, "prefix": a.prefix, "systems": systems,
                       "take": {k: list(v) for k, v in rules.items()}, "claims_from": claims_from,
                       "validation": [str(p) for p in csvs][:5] + (["..."] if len(csvs) > 5 else []),
                       "n_validation_csvs": len(csvs), "instance_root": str(a.instance_root or "")},
        "folders": {k: str(v) for k, v in folders.items()},
        "archives": archives,
        "totals": {"runs": len(selected), "payload_bytes": total_payload,
                   "estimated_bytes": total_estimate, "archives": n_archives},
        "zenodo": {"max_files": a.max_files, "other_files": a.other_files,
                   "files_in_record": n_archives + a.other_files, "fits_file_limit": fits_files,
                   "record_limit_gb": a.record_limit_gb, "within_record_limit": within_record},
        "excluded": [{"system": s, "folder": f, "reason": r, "rows": n}
                     for (s, f, r), n in sorted(excluded.items())],
        "unpublished_finished_runs": len(unpublished),
        "left_out_files": dict(sorted(left_out.items())),
        "licence_check": dict(licence_notes),
    }
    write_csv(plan_dir / "selected_runs.csv", ("problem", "instance", "system", "folder", "archive", "bytes"),
              selected_rows)
    write_json(plan_dir / "plan.json", plan)

    print(f"\nARCHIVES (max {a.max_archive_gb:g} GB each, {n_archives} in all):")
    print(f"  {'#':>3} {'archive':<46} {'runs':>7} {'files':>8} {'size':>14}  first .. last problem")
    for x in archives:
        flag = "  OVER THE LIMIT (one run larger than the limit)" if x["over_limit"] else ""
        print(f"  {x['index']:>3} {x['name']:<46} {x['runs']:>7,} {x['files']:>8,} "
              f"{human(x['estimated_bytes']):>14}  {x['first'].split('/')[0][:28]} .. "
              f"{x['last'].split('/')[0][:28]}{flag}")
    for (family, licence) in sorted(runs_by_group):
        group = [x for x in archives if (x["family"], x["licence"]) == (family, licence)]
        print(f"  group {family}/{licence}: {len(group)} archive(s), "
              f"{sum(x['runs'] for x in group):,} runs, {human(sum(x['estimated_bytes'] for x in group))}")
    print(f"  TOTAL: {len(selected):,} runs, {n_archives} archives, {human(total_estimate)} "
          f"(file data {human(total_payload)})")
    if left_out:
        print("  left out of the run folders: " + ", ".join(f"{k} x{v:,}" for k, v in sorted(left_out.items())))
    if licence_notes:
        print("  licence check: " + ", ".join(f"{k}: {v}" for k, v in sorted(licence_notes.items())))
    print(f"\nZENODO: {n_archives} archives + {a.other_files} other files = {n_archives + a.other_files} "
          f"of {a.max_files} files per record: {'fits' if fits_files else 'DOES NOT FIT'}")
    print(f"        {human(total_estimate)} of the {a.record_limit_gb:g} GB record limit: "
          f"{'within' if within_record else 'EXCEEDS the limit (quota increase or a second record needed)'}")
    if plan["optimizer_commit"] == "unknown" or plan["optimizer_commit"].endswith("-dirty"):
        print(f"\n[WARN] optimizer commit {plan['optimizer_commit']!r}: the archives' README names it; "
              "plan from a clean git checkout", file=sys.stderr)
    print(f"\n[plan] {plan_dir}/plan.json written; pack with --array=1-{n_archives}")
    if not fits_files:
        print("[ERROR] too many archives for one record: raise --max-archive-gb", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------

def load_plan(plan_dir: Path) -> dict:
    path = plan_dir / "plan.json"
    if not path.is_file():
        raise PackError(f"{path} not found: run the plan step first")
    return json.loads(path.read_text(encoding="utf-8"))


def zip_date(plan: dict) -> Tuple[int, int, int, int, int, int]:
    y, m, d = (int(x) for x in plan["created"].split("-"))
    return (y, m, d, 0, 0, 0)


def write_member(zf: zipfile.ZipFile, arcname: str, when, source=None, data: bytes = None) -> Tuple[str, int]:
    """Write one member from a file (streamed) or from bytes; return its (sha256, bytes)."""
    info = zipfile.ZipInfo(arcname, date_time=when)
    info.compress_type = zipfile.ZIP_STORED if arcname.endswith((".gz", ".npz")) else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    digest = hashlib.sha256()
    n = 0
    size = len(data) if data is not None else Path(source).stat().st_size
    with zf.open(info, "w", force_zip64=size > ZIP64_LIMIT) as dst:
        if data is not None:
            digest.update(data)
            dst.write(data)
            n = len(data)
        else:
            with open(source, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
                    dst.write(chunk)
                    n += len(chunk)
    return digest.hexdigest(), n


def result_line(run: dict, folders: Dict[str, Path], cache: Dict[Path, "vs.ClaimSource"]) -> dict:
    """The solver's last result line the validator checked, equal to the row's claimed_* values."""
    problem_dir = folders[run["claims_folder"]] / f"output_{run['problem']}"
    source = cache.get(problem_dir)
    if source is None:
        cache.clear()                         # runs are sorted by problem: keep one at a time
        source = cache[problem_dir] = vs.ClaimSource(problem_dir, run["problem"])
    lines, where = source.lines(run["instance"], run["system"])
    if not isinstance(lines, list) or not lines or not isinstance(lines[-1], dict):
        raise PackError(f"{problem_dir}: no result line for {run['instance']} {run['system']}")
    last = lines[-1]
    for m in vs.METRICS:
        if str(last.get(m, "")) != run["claims"].get(f"claimed_{m}", ""):
            raise PackError(f"{problem_dir} {run['instance']} {run['system']}: result line {m}="
                            f"{last.get(m)!r}, validation row {run['claims'].get(f'claimed_{m}')!r}")
    return last


def manifest_bytes(rows: List[Sequence]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


LICENCE_TEXT = {
    CC_BY: ("**CC-BY-4.0** (https://creativecommons.org/licenses/by/4.0/). These regions use "
            "synthetic grid navpoints and contain no X-Plane navigation data."),
    GPL: ("**GPL-2.0-or-later** (https://www.gnu.org/licenses/old-licenses/gpl-2.0.html). A solution "
          "matrix describes flights through the navigation graph of its instance and carries the "
          "instance's licence; the graphs of these regions are built on waypoints from the X-Plane "
          "navigation data, which is licensed GPL-2.0-or-later (see LICENSING.md of the "
          "large-scaling instance record). Redistribute under the same terms."),
}


def readme_text(plan: dict, meta: dict, runs: List[dict]) -> str:
    name, family, licence = meta["name"], meta["family"], meta["licence"]
    dataset, doi, zip_name = INSTANCE_RECORDS[family]
    systems = Counter(r["system"] for r in runs)
    problems = sorted({r["problem"] for r in runs})
    rerun = sorted({r["system"] for r in runs if r["claims_folder"] != r["folder"]})
    first, last = meta["first"].split("/")[0], meta["last"].split("/")[0]
    example_root = "experiment_data_V2_small_scaling" if family == "small" else \
        "experiment_data_V2_large_scaling_TG60"
    lines = [
        f"# {name}", "",
        f"Solution matrices of {len(runs):,} benchmark runs on the V2 joint ATFCM instances "
        f"({dataset} dataset), licence {licence}. Archive {meta['part']} of {meta['parts']} with "
        f"this dataset and licence; "
        + (f"1 problem, `{first}`. " if len(problems) == 1 else
           f"{len(problems)} problems, from `{first}` to `{last}` in sorted order. ")
        + "Part of the benchmark results record; its README describes the methods, the "
        "campaign and the independent check (Sections 5 and 6).", "",
        "Included are the runs that finished within the limits (1800 s, 35 GiB) and whose "
        "matrices passed the independent check (status VALID): every hard constraint of the "
        "model holds, and the matrices reproduce the six objective values the method reported.", "",
        "## Licence", "", LICENCE_TEXT[licence], "",
        "## Layout", "",
        "    README.md       this file",
        "    MANIFEST.csv    one row per file: problem, instance, system, file, bytes, sha256",
        "    <problem>/<method>/<instance>/",
        "        converted_navpoint_matrix.csv.gz      flights x timesteps: the vertex flight f reaches at timestep t, -1 at every other timestep",
        "        converted_instance_matrix.csv.gz      flights x timesteps: the sector flight f is in at t, -1 before departure and after landing",
        "        navaid_sector_time_assignment.csv.gz  vertices x timesteps: the sector of vertex v at t, named by its representative vertex",
        "        capacity_time_matrix.csv.gz           vertices x timesteps: the capacity of the sector named s at t, 0 where no sector is named s",
        "        manifest.json                         matrix shapes, time granularity and arrival-delay metric of the run",
        "        result_line.json                      the method's final result line: the six objective values, status and times",
        "",
        "Matrices are comma-separated integers, gzip-compressed, one row per flight or vertex (ids as "
        "in the instance files) and one column per timestep of the instance files, from timestep 0 "
        "(the results record's README, Section 4, gives the time convention). "
        "A method that writes another format has `.csv` or `.npz` in place of `.csv.gz`.", "",
        "| method | runs |", "|---|---:|",
    ] + [f"| `{s}` | {n:,} |" for s, n in sorted(systems.items())] + [""]
    if rerun:
        lines += [
            "`" + "`, `".join(rerun) + "`: the benchmark campaign stored the matrices of `03_DELAY` "
            "and `03A_CASA` in one folder, so `03A_CASA` overwrote most of those of `03_DELAY`. "
            "Both methods were rerun with the same solver code, options and limits; their matrices "
            "come from that rerun, which reported the same objective values as the campaign in "
            "every run included here. `result_line.json` is the campaign's line.", ""]
    lines += [
        "## How to validate", "",
        f"1. Download the instances: `{zip_name}` of the {dataset} dataset record "
        f"(DOI {doi}), and extract it.",
        "2. Get `06_benchmark_start_script/validate_solutions.py` of the ASPaeroFlow-Optimizer "
        f"(https://github.com/alexl4123/ASPaeroFlow-Optimizer, commit `{plan['optimizer_commit']}` "
        "or later); it needs Python 3 with numpy, pandas and scipy.",
        "3. Validate one problem of this archive:", "",
        f"       python validate_solutions.py --problem-dir {name}/<problem> \\",
        f"           --instance-root {example_root} --out-dir check", "",
        f"   The result is one row per run in `check/{name}/validation_<problem>.csv`; every run "
        "should be VALID. A large-scaling problem `<region>-TG<g>-PCAP<c>` needs the instance "
        "archive of its time granularity `<g>`; the validator finds its instances under "
        "`<region>-TG<g>/PCAP<c>/`.", "",
        "`MANIFEST.csv` holds the sha256 of every file; `sha256sum` of a file must match its row.", "",
    ]
    return "\n".join(lines)


def verify_archive(path: Path, name: str, rows: List[Sequence]) -> None:
    """testzip, the member list, and the sha256 and size of every member against MANIFEST.csv."""
    want = {f"{name}/README.md", f"{name}/MANIFEST.csv"} | {
        f"{name}/{p}/{s}/{i}/{f}" for p, i, s, f, _, _ in rows}
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise PackError(f"{path.name}: CRC error in {bad}")
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise PackError(f"{path.name}: a member name appears twice")
        if set(names) != want:
            extra, lost = sorted(set(names) - want), sorted(want - set(names))
            raise PackError(f"{path.name}: members differ from the plan (extra {extra[:3]}, missing {lost[:3]})")
        with zf.open(f"{name}/MANIFEST.csv") as fh:
            listed = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        if len(listed) != len(rows):
            raise PackError(f"{path.name}: MANIFEST.csv has {len(listed)} rows, {len(rows)} files written")
        for row in listed:
            member = f"{name}/{row['problem']}/{row['system']}/{row['instance']}/{row['file']}"
            digest = hashlib.sha256()
            n = 0
            with zf.open(member) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
                    n += len(chunk)
            if digest.hexdigest() != row["sha256"] or n != int(row["bytes"]):
                raise PackError(f"{path.name}: {member} does not match its MANIFEST.csv row")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_pack(a: argparse.Namespace) -> int:
    plan = load_plan(a.plan_dir)
    if not plan["zenodo"]["fits_file_limit"] and not a.force:
        raise PackError("the plan has more archives than a record may hold; re-plan (or --force)")
    if not 1 <= a.index <= len(plan["archives"]):
        print(f"[SKIP] index {a.index}: the plan has {len(plan['archives'])} archives")
        return 0
    meta = plan["archives"][a.index - 1]
    name = meta["name"]
    out: Path = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    zip_path, sha_path = out / f"{name}.zip", out / f"{name}.zip.sha256"
    man_path, partial = out / f"{name}.MANIFEST.csv", out / f"{name}.zip.partial"
    if zip_path.exists() and sha_path.exists() and man_path.exists() and not a.force:
        print(f"[SKIP] {name}: already packed ({sha_path.name} present); --force repacks")
        return 0
    for stale in (zip_path, sha_path, man_path, partial):
        if stale.exists():
            stale.unlink()

    runs = json.loads((a.plan_dir / "archives" / f"{name}.runs.json").read_text(encoding="utf-8"))
    folders = {k: Path(v) for k, v in plan["folders"].items()}
    when = zip_date(plan)
    t0 = time.time()
    print(f"[pack] {name}: {len(runs):,} runs, estimated {human(meta['estimated_bytes'])} -> {zip_path}",
          flush=True)
    rows: List[Sequence] = []
    cache: Dict[Path, vs.ClaimSource] = {}
    with zipfile.ZipFile(partial, "w", allowZip64=True) as zf:
        write_member(zf, f"{name}/README.md", when, data=readme_text(plan, meta, runs).encode("utf-8"))
        for k, run in enumerate(runs, 1):
            src = folders[run["folder"]] / f"output_{run['problem']}" / run["src"]
            prefix = f"{name}/{run['problem']}/{run['system']}/{run['instance']}"
            for fname, size in run["files"]:
                path = src / fname
                actual = path.stat().st_size
                if actual != size:
                    raise PackError(f"{path}: {actual} bytes now, {size} in the plan (changed since)")
                digest, n = write_member(zf, f"{prefix}/{fname}", when, source=path)
                rows.append((run["problem"], run["instance"], run["system"], fname, n, digest))
            data = (json.dumps(result_line(run, folders, cache), indent=2) + "\n").encode("utf-8")
            digest, n = write_member(zf, f"{prefix}/{RESULT_LINE_FILE}", when, data=data)
            rows.append((run["problem"], run["instance"], run["system"], RESULT_LINE_FILE, n, digest))
            if k % 500 == 0:
                print(f"  {k:,}/{len(runs):,} runs  {time.time() - t0:.0f} s", flush=True)
        manifest = manifest_bytes(rows)
        write_member(zf, f"{name}/MANIFEST.csv", when, data=manifest)
    t_write = time.time() - t0
    verify_archive(partial, name, rows)
    t_verify = time.time() - t0 - t_write
    os.replace(partial, zip_path)
    man_path.write_bytes(manifest)
    sha = sha256_file(zip_path)
    tmp = sha_path.with_suffix(".tmp")
    tmp.write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
    tmp.replace(sha_path)
    size = zip_path.stat().st_size
    print(f"[OK] {name}.zip: {len(runs):,} runs, {len(rows):,} files, {human(size)} "
          f"(estimate {human(meta['estimated_bytes'])}); write {t_write:.0f} s, verify {t_verify:.0f} s; "
          f"sha256 {sha}")
    if size > plan["parameters"]["max_archive_gb"] * GB and not meta["over_limit"]:
        print(f"[ERROR] {name}.zip is larger than the limit", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------------

def cmd_check(a: argparse.Namespace) -> int:
    plan = load_plan(a.plan_dir)
    out: Path = a.out_dir
    problems: List[str] = []
    where: Dict[Tuple[str, str, str], str] = {}
    sums: List[str] = []
    total = 0
    print(f"  {'#':>3} {'archive':<46} {'runs':>7} {'files':>8} {'size':>14}")
    for meta in plan["archives"]:
        name = meta["name"]
        zip_path, sha_path, man_path = out / f"{name}.zip", out / f"{name}.zip.sha256", out / f"{name}.MANIFEST.csv"
        absent = [p.name for p in (zip_path, sha_path, man_path) if not p.exists()]
        if absent:
            problems.append(f"{name}: missing {', '.join(absent)}")
            continue
        with man_path.open(newline="", encoding="utf-8") as fh:
            listed = list(csv.DictReader(fh))
        keys = {(r["problem"], r["instance"], r["system"]) for r in listed}
        planned = {(r["problem"], r["instance"], r["system"]) for r in json.loads(
            (a.plan_dir / "archives" / f"{name}.runs.json").read_text(encoding="utf-8"))}
        if keys != planned:
            problems.append(f"{name}: {len(keys - planned)} runs not in its plan, "
                            f"{len(planned - keys)} planned runs missing")
        for key in keys:
            if key in where:
                problems.append(f"{key} is in {where[key]} and {name}")
            where[key] = name
        line = sha_path.read_text(encoding="utf-8").strip()
        if a.rehash and line.split()[0] != sha256_file(zip_path):
            problems.append(f"{name}.zip: sha256 differs from {sha_path.name}")
        size = zip_path.stat().st_size
        total += size
        if size > plan["parameters"]["max_archive_gb"] * GB and not meta["over_limit"]:
            problems.append(f"{name}.zip: {human(size)}, over the limit")
        sums.append(line)
        print(f"  {meta['index']:>3} {name:<46} {len(keys):>7,} {len(listed):>8,} {human(size):>14}")
    with (a.plan_dir / "selected_runs.csv").open(newline="", encoding="utf-8") as fh:
        selected = {(r["problem"], r["instance"], r["system"]) for r in csv.DictReader(fh)}
    if not problems:
        lost, extra = selected - set(where), set(where) - selected
        if lost:
            problems.append(f"{len(lost)} selected runs are in no archive, e.g. {sorted(lost)[0]}")
        if extra:
            problems.append(f"{len(extra)} packed runs were not selected, e.g. {sorted(extra)[0]}")
    n = len(plan["archives"])
    print(f"  TOTAL {len(where):,} runs of {len(selected):,} selected, {n} archives, {human(total)}; "
          f"Zenodo files: {n} + {plan['zenodo']['other_files']} other = {n + plan['zenodo']['other_files']} "
          f"of {plan['zenodo']['max_files']}")
    if problems:
        for p in problems:
            print(f"[ERROR] {p}", file=sys.stderr)
        return 1
    (out / "SHA256SUMS").write_text("\n".join(sorted(sums, key=lambda s: s.split()[-1])) + "\n",
                                    encoding="utf-8")
    print(f"[OK] every selected run is in exactly one archive; {out}/SHA256SUMS written")
    return 0


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("WHICH RUNS"):])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="select the runs, stat their files, assign them to archives")
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--validation", type=Path, action="append", required=True,
                   help="a validation_<PROBLEM>.csv, or a folder searched for them (repeatable)")
    p.add_argument("--results-folder", action="append", default=[], metavar="NAME=PATH",
                   help="where the results folder NAME (the rows' folder column) is (repeatable)")
    p.add_argument("--claims-from", action="append", default=[], metavar="RERUN=CAMPAIGN",
                   help="for a folder validated in rerun mode: the folder whose result lines it was "
                        "judged against")
    p.add_argument("--take", action="append", default=None, metavar="SYSTEMS=FOLDERS",
                   help="where each system's matrices come from; replaces the V2 default "
                        + " ".join(DEFAULT_TAKE))
    p.add_argument("--systems", default="",
                   help="comma-separated published systems (default: the validator's PUBLISHED_SYSTEMS)")
    p.add_argument("--instance-root", type=Path, default=None,
                   help="check the licence against instance_info.json of one instance per problem")
    p.add_argument("--max-archive-gb", type=float, default=20.0, help="per archive, 1 GB = 10^9 bytes")
    p.add_argument("--max-files", type=int, default=100, help="Zenodo's file limit per record")
    p.add_argument("--other-files", type=int, default=10, help="the record's files besides the archives")
    p.add_argument("--record-limit-gb", type=float, default=50.0, help="Zenodo's size limit per record")
    p.add_argument("--prefix", default="solution_matrices", help="archive names: <prefix>_<family>_<licence>_NN")
    p.add_argument("--selection-only", action="store_true",
                   help="report the selection and write selected_runs.csv; no stat, no plan.json")

    p = sub.add_parser("pack", help="write and verify one archive of the plan")
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--index", type=int, required=True, help="1-based archive number (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="repack an archive that is already there")

    p = sub.add_parser("check", help="after packing: coverage, sizes, SHA256SUMS")
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rehash", action="store_true", help="recompute the sha256 of every archive")

    a = ap.parse_args(argv)
    try:
        return {"plan": cmd_plan, "pack": cmd_pack, "check": cmd_check}[a.cmd](a)
    except (PlanError, PackError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
