#!/usr/bin/env python3
"""Read the usc run over all 27 exact-ASP variants (manifest tier U): how far does usc carry?

    ./analyze_usc27.py --folder 20261001_USC27
    ./analyze_usc27.py --folder 20261001_USC27 --missing-only     # just the resubmission list

analyze_asp_ablation.py compares PROFILES against each other and has nothing to compare when
there is only one. This script asks the other question tier U exists for: per VARIANT, and per
FLIGHT COUNT, how many instances usc closes, how long that takes, and how the runs that do not
close end (time limit, memory, no incumbent at all). It reads the runs with the ablation's own
collect(), so closed means exactly what it means there: clingo's SOLVER-EXHAUSTED, never the
exit status.

A run in which usc found no model before its solve deadline prints its statistics line (lower
bound included) and then raises in 02_ASP, so the caller records it as an error. That is an
outcome of the search, not a crash, and it is counted apart as no_model_by_deadline; the error
column holds only the errors it does not explain. Its traceback in the job log is expected.

It also lists the manifest rows that produced no run at all -- a job that never started, was
killed by SLURM, or crashed before its result JSON was written -- as array ranges ready for a
resubmission. Read the job logs for those rows first: a crash that happens once happens again.

Writes usc27_variants.csv, usc27_scaling.csv and usc27_summary.md into the campaign folder.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ablation = _load("analyze_asp_ablation")
manifest_builder = _load("build_ablation_manifest")

#: The 27 variants in the caller's own order (ground delay outer, then rerouting, then
#: sectorisation), as <rerouting>_<delay>_<sectorisation>.
VARIANTS: List[str] = [key.split("_ASP_", 1)[1] for key in manifest_builder.all27_systems()]


def read_manifest(path: Path, tier: str) -> List[Dict[str, str]]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [row for row in rows if row["tier"] == tier]


def ranges(numbers: Sequence[int]) -> str:
    """1,2,3,7,9,10 -> '1-3,7,9-10', the form sbatch --array takes."""
    out, numbers = [], sorted(numbers)
    start = prev = None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def missing_rows(manifest: List[Dict[str, str]], runs) -> List[int]:
    have = {(r.variant, r.problem, r.instance) for r in runs}
    missing = []
    for row in manifest:
        variant = row["system"].split("_ASP_", 1)[1]
        if (variant, row["problem"], row["instance"]) not in have:
            missing.append(int(row["task_id"]))
    return missing


def no_model_by_deadline(run) -> bool:
    """usc stopped at its deadline without any model; 02_ASP then raises by design."""
    return bool(run.stopped) and not run.has_incumbent


def per_variant(runs, limit: float) -> List[Dict]:
    by_variant = defaultdict(list)
    for run in runs:
        by_variant[run.variant].append(run)
    rows = []
    for variant in VARIANTS:
        mine = by_variant.get(variant, [])
        closed = [r for r in mine if r.closed]
        by_size = defaultdict(list)
        for run in mine:
            by_size[run.size].append(run)
        # The largest flight count at which EVERY attempted instance closed, and the largest at
        # which ANY did: the first is where usc is dependable, the second where it still reaches.
        all_closed = [size for size, rs in by_size.items() if rs and all(r.closed for r in rs)]
        any_closed = [size for size, rs in by_size.items() if any(r.closed for r in rs)]
        rows.append({
            "variant": variant,
            "runs": len(mine),
            "closed": len(closed),
            "closed_pct": round(100.0 * len(closed) / len(mine), 1) if mine else None,
            "timeout": sum(1 for r in mine if r.outcome == "timeout"),
            "memout": sum(1 for r in mine if r.outcome == "memout"),
            "error": sum(1 for r in mine if r.outcome == "error" and not no_model_by_deadline(r)),
            "stopped_at_deadline": sum(1 for r in mine if r.stopped),
            "no_model_by_deadline": sum(1 for r in mine if no_model_by_deadline(r)),
            "no_incumbent": sum(1 for r in mine if not r.has_incumbent),
            "median_s_closed": round(statistics.median(
                [r.wall_s for r in closed if r.wall_s is not None]), 1)
                if any(r.wall_s is not None for r in closed) else None,
            "par2_s": round(ablation.par2(mine, limit), 1) if mine else None,
            "max_flights_all_closed": max(all_closed) if all_closed else None,
            "max_flights_any_closed": max(any_closed) if any_closed else None,
        })
    return rows


def per_size(runs) -> Tuple[List[int], Dict[Tuple[str, int], Tuple[int, int]]]:
    cells: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
    sizes = sorted({r.size for r in runs if r.size is not None})
    for run in runs:
        cell = cells[(run.variant, run.size)]
        cell[0] += int(bool(run.closed))
        cell[1] += 1
    return sizes, {k: (v[0], v[1]) for k, v in cells.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--folder", type=str, required=True,
                        help="Campaign folder under --output-root, as passed to sbatch as FOLDER")
    parser.add_argument("--manifest", type=Path, default=Path("usc27_tasks.tsv"))
    parser.add_argument("--tier", type=str, default="U")
    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="Per-run limit the run used; PAR2 needs it")
    parser.add_argument("--missing-only", action="store_true",
                        help="Only list the manifest rows without a result, then exit")
    args = parser.parse_args()

    folder = args.output_root / args.folder
    runs = [r for r in ablation.collect(folder) if r.tier == args.tier] if folder.is_dir() else []
    manifest = read_manifest(args.manifest, args.tier) if args.manifest.exists() else []
    print(f"[folder]   {folder}: {len(runs)} tier-{args.tier} runs")

    if manifest:
        missing = missing_rows(manifest, runs)
        print(f"[manifest] {args.manifest}: {len(manifest)} tier-{args.tier} rows, "
              f"{len(manifest) - len(missing)} with a result, {len(missing)} without")
        if missing:
            print(f"           rows without a result: {ranges(missing)}")
            print("           read their logs before resubmitting (logs/<job name>-<job>_<row>.out);"
                  "\n           with CHUNK=1 and ROW_OFFSET=0 the row number IS the array index:")
            print(f"           --array={ranges(missing)}%40")
    else:
        print(f"[manifest] {args.manifest} not found: cannot say which rows are missing")
    if args.missing_only or not runs:
        return 0 if runs else 1

    variants = per_variant(runs, args.time_limit)
    sizes, cells = per_size(runs)

    with (folder / "usc27_variants.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(variants[0].keys()))
        writer.writeheader()
        writer.writerows(variants)
    with (folder / "usc27_scaling.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", "flights", "closed", "attempted"])
        for variant in VARIANTS:
            for size in sizes:
                if (variant, size) in cells:
                    writer.writerow([variant, size, *cells[(variant, size)]])

    def fmt(value):
        return "-" if value is None else str(value)

    lines = [
        f"# {runs[0].profile if runs else 'usc'} over all 27 exact-ASP variants (tier {args.tier})", "",
        f"{len(runs)} runs, per-run limit {args.time_limit:.0f} s. Closed = clingo's "
        "SOLVER-EXHAUSTED (optimum proven). PAR2 counts an unclosed run as twice the limit.", "",
        "| variant | runs | closed | closed % | median s (closed) | PAR2 s | timeout | memout | "
        "error | no model by deadline | no incumbent | all closed up to | any closed up to |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in variants:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['closed']} | {fmt(row['closed_pct'])} | "
            f"{fmt(row['median_s_closed'])} | {fmt(row['par2_s'])} | {row['timeout']} | "
            f"{row['memout']} | {row['error']} | {row['no_model_by_deadline']} | "
            f"{row['no_incumbent']} | "
            f"{fmt(row['max_flights_all_closed'])} | {fmt(row['max_flights_any_closed'])} |")
    lines += ["", "Closed / attempted per flight count:", "",
              "| variant | " + " | ".join(str(s) for s in sizes) + " |",
              "|---|" + "---|" * len(sizes)]
    for variant in VARIANTS:
        cols = [("{}/{}".format(*cells[(variant, s)]) if (variant, s) in cells else "-")
                for s in sizes]
        lines.append(f"| {variant} | " + " | ".join(cols) + " |")
    (folder / "usc27_summary.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines[6:6 + len(variants)]))
    print(f"\nwrote {folder}/usc27_variants.csv, usc27_scaling.csv, usc27_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
