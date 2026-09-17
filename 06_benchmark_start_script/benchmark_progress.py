#!/usr/bin/env python3
"""
How far along is the benchmark? Run this after SSH-ing in.

Each array task appends to output/<FOLDER>/output_<PROBLEM>/progress.jsonl; this reads all of
them and summarises. Nothing here touches the running jobs.

    ./benchmark_progress.py                      # newest output folder
    ./benchmark_progress.py --folder 20260911_V2
    ./benchmark_progress.py --detail             # per-task table
    ./benchmark_progress.py --failures           # what went wrong, and where

It reads BOTH campaign shapes and picks by what the folder contains:

  monolithic   run_all_benchmarks.slurm -- one array task per problem, writing
               output/<FOLDER>/output_<PROBLEM>/progress.jsonl. The denominator is estimated from
               ../05_instances/problems.tsv times 39 or 12 systems per instance.

  per-unit     run_benchmark_units.slurm -- one job per (problem, instance, system), writing
               output/<FOLDER>/units/shards/<PROBLEM>/<INSTANCE>/<SYSTEM>/progress.jsonl. Here the
               denominator is EXACT: units/worklist.tsv lists every unit the campaign consists of,
               so nothing has to be guessed from a systems-per-instance count.

A per-unit folder is recognised by units/worklist.tsv. After merge_benchmark_shards.py has run,
output_<PROBLEM>/progress.jsonl exists there too and reads exactly like a monolithic folder; the
worklist view is preferred while the campaign is in flight, since it is the one that knows about
units that have not started.
"""
import argparse, json, re, time
from collections import Counter, defaultdict
from pathlib import Path


CODES = {"T": "TIMEOUT", "M": "MEMOUT", "E": "ERROR", "P": "UNPARSED"}


#: The part of a problem directory name that every problem in a campaign shares: the data window,
#: the en-route capacity and the cluster size. Dropping it leaves what tells problems apart --
#: region, graph, granularity and capacity level -- e.g.
#:   04-0-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-50-GABRIEL-GRAPH-V2-TG15-PCAP100
#:   -> 04-0-DACH-GABRIEL-GRAPH-V2-TG15-PCAP100
_SHARED_PART = re.compile(r"-\d{4}-\d{2}-\d{2}--\d{4}-\d{2}-\d{2}-CAP-ENROUTE-\d+-CLUSTERSIZE-\d+")


def display_name(name, full=False):
    """A problem name short enough to read in a table, with nothing distinguishing cut off."""
    return name if full else _SHARED_PART.sub("", name)


def _true_outcome(row):
    """Outcome from the solver's ERROR key, falling back to the recorded label."""
    err = row.get("error_code")
    if err is None:
        sol = row.get("objective")
        if isinstance(sol, list) and sol and isinstance(sol[-1], dict):
            err = sol[-1].get("ERROR")
        elif isinstance(sol, str):
            err = sol
    if err in CODES:
        return CODES[err]
    if err in ("", None):
        return row.get("outcome", "ok") if err is None else "ok"
    return row.get("outcome", "?")


def hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def expected_runs(manifest, root, sys_small, sys_large):
    """Solver runs the WHOLE campaign implies, read from the problem index.

    The other denominator in this report -- the sum of each task's own "total" -- only
    covers tasks that have STARTED, because progress.jsonl is what carries it. That
    number therefore reports how far the launched work has got, not how far the campaign
    has got, and it grows as tasks launch. This reads ../05_instances/problems.tsv
    instead, which lists every problem whether or not it has run.

    Columns: problem_dir, time_granularity, region, capacity_level, n_instances.
    capacity_level == "NONE" marks the small-scaling family, which runs 39 systems per
    instance against the large family's 12. A MIP-only pass runs exactly one.
    """
    if not manifest.exists():
        return None, None
    rows = []
    with open(manifest) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_cap, i_n = header.index("capacity_level"), header.index("n_instances")
        except ValueError:
            return None, None
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(i_cap, i_n):
                try:
                    rows.append((f[i_cap], int(f[i_n])))
                except ValueError:
                    continue
    if not rows:
        return None, None
    # RUN_MIP=only writes "MIP ONLY" into the provenance; that pass runs one system.
    mip_only = any("MIP ONLY" in prov.read_text(errors="replace")
                   for prov in root.glob("run_provenance_*.txt"))
    total = sum(n * (1 if mip_only else (sys_small if cap == "NONE" else sys_large))
                for cap, n in rows)
    return total, len(rows)


def read_worklist(path):
    """[(problem, instance, system)] in campaign order, or None if this is not a per-unit folder."""
    if not path.exists():
        return None
    units = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_p, i_i, i_s = (header.index("problem_dir"), header.index("instance"),
                             header.index("system"))
        except ValueError:
            return None
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(i_p, i_i, i_s):
                units.append((f[i_p], f[i_i], f[i_s]))
    return units or None


def report_units(root, units, a):
    """Progress of a per-unit campaign, counted against the worklist.

    Every unit is one solver run in one job, so "done" is simply how many shards have written a
    progress line. A unit that has not started is not missing -- it is queued -- which is why the
    denominator comes from the worklist rather than from the shards that happen to exist.
    """
    shards = root / "units" / "shards"
    per_problem, outcomes, failures = defaultdict(lambda: [0, 0]), Counter(), []
    newest, skipped = 0.0, 0
    for problem, instance, system in units:
        per_problem[problem][1] += 1
        shard = shards / problem / instance / system
        prog = shard / "progress.jsonl"
        if not prog.exists():
            marker = shard / "unit.json"
            if marker.exists():
                try:
                    if json.loads(marker.read_text(errors="replace")).get("status") == "skipped_no_licence":
                        skipped += 1
                except (OSError, json.JSONDecodeError):
                    pass
            continue
        rows = []
        for line in prog.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue        # a half-written last line while the unit is running
        if not rows:
            continue
        r = rows[-1]
        per_problem[problem][0] += 1
        outcome = _true_outcome(r)
        outcomes[outcome] += 1
        if outcome != "ok":
            failures.append((problem, r.get("instance"), r.get("system"), outcome))
        newest = max(newest, prog.stat().st_mtime)

    tot_done = sum(d for d, _ in per_problem.values())
    tot_all = len(units)
    finished = sum(1 for d, t in per_problem.values() if t and d >= t)
    merged = len(list(root.glob("output_*/execution_time.csv")))

    print(f"  campaign shape     per-unit (units/worklist.tsv)")
    print(f"  problems           {len(per_problem)}")
    print(f"  problems complete  {finished}   ({merged} merged into output_<PROBLEM>/)")
    print(f"  solver runs        {tot_done:,} / {tot_all:,}  "
          f"({100 * tot_done / tot_all if tot_all else 0:.1f}%)  OF THE CAMPAIGN")
    if skipped:
        print(f"  skipped            {skipped:,}  (RUN_MIP=auto, no Gurobi licence on that node)")
    if newest:
        print(f"  last activity      {hms(time.time() - newest)} ago")
    else:
        print(f"  last activity      nothing has finished yet")
    print(f"\n  outcomes: " + "  ".join(f"{k}={v:,}" for k, v in outcomes.most_common()))
    if tot_done:
        print(f"  solved within limits: {100 * outcomes.get('ok', 0) / tot_done:.1f}% of finished runs")

    if a.detail:
        names = {p: display_name(p, a.full_names) for p in per_problem}
        width = max([len("problem")] + [len(n) for n in names.values()]) + 2
        print(f"\n  {'problem':<{width}}{'done':>8}{'total':>8}  progress")
        for problem, (d, t) in sorted(per_problem.items(), key=lambda kv: (kv[1][0] / kv[1][1]) if kv[1][1] else 0):
            bar = "#" * int(20 * d / t) if t else ""
            print(f"  {names[problem]:<{width}}{d:>8}{t:>8}  {bar:<20} {100 * d / t if t else 0:5.1f}%")

    if a.failures:
        if not failures:
            print("\n  no failures recorded")
        else:
            print(f"\n  {len(failures)} non-ok runs; by system:")
            by_sys = defaultdict(Counter)
            for _, _, system, outcome in failures:
                by_sys[system][outcome] += 1
            for system, c in sorted(by_sys.items(), key=lambda kv: -sum(kv[1].values())):
                print(f"    {system:<30} " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
            print("\n  first 15:")
            for problem, inst, system, outcome in failures[:15]:
                print(f"    {outcome:<9} {system:<28} {inst:<22} {display_name(problem, a.full_names)}")
    print("\n  Units that never produced a result are named by:"
          "\n    ./merge_benchmark_shards.py --folder %s --write-missing-worklist retry.tsv"
          % root.name)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--folder", default=None, help="default: most recently modified")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--full-names", action="store_true",
                    help="print problem names in full; by default the part every problem shares "
                         "(data window, en-route capacity, cluster size) is left out")
    ap.add_argument("--manifest", type=Path, default=Path("../05_instances/problems.tsv"),
                    help="problem index, for the true campaign denominator")
    ap.add_argument("--systems-small", type=int, default=39,
                    help="solver systems per small-family instance (capacity_level NONE)")
    ap.add_argument("--systems-large", type=int, default=12,
                    help="solver systems per large-family instance")
    a = ap.parse_args()

    root = a.output_root / a.folder if a.folder else None
    if root is None:
        cands = [p for p in a.output_root.iterdir() if p.is_dir()] if a.output_root.is_dir() else []
        if not cands:
            raise SystemExit(f"no output folders under {a.output_root}")
        root = max(cands, key=lambda p: p.stat().st_mtime)
    print(f"run folder: {root}\n")

    # A per-unit campaign counts against its worklist; a monolithic one against the problem index.
    units = read_worklist(root / "units" / "worklist.tsv")
    if units is not None:
        return report_units(root, units, a)

    files = sorted(root.glob("output_*/progress.jsonl"))
    if not files:
        # Distinguish "not started" from "started, nothing finished yet". progress.jsonl gets
        # its first line when the first (instance, solver) COMPLETES, so a pass whose solvers
        # each take up to the time limit looks identical to one that never launched.
        started = sorted(root.glob("output_*"))
        if started:
            raise SystemExit(
                f"{len(started)} task(s) have started under {root}, but no solver run has "
                f"finished yet, so there is nothing to summarise.\n"
                f"That is normal early on when the time limit is high -- the first line "
                f"appears when the first (instance, solver) completes.\n"
                f"Follow a task directly meanwhile:  tail -f logs/v2bench-*_1.out")
        raise SystemExit(f"nothing under {root} yet -- no task has started")

    per_task, outcomes, failures = {}, Counter(), []
    newest = 0.0
    for f in files:
        task = f.parent.name.replace("output_", "")
        done = total = 0
        rows = []
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue        # a half-written last line while a job is running
            rows.append(r)
            # Recover the true outcome from the solver's own ERROR key. Rows written before
            # 2026-09-12 recorded outcome="ok" unconditionally, because the writer keyed off
            # the runtime -- which is always a float -- instead of the output's ERROR field.
            # The raw output is stored under "objective", so those rows can still be read
            # correctly rather than discarded.
            outcome = _true_outcome(r)
            outcomes[outcome] += 1
            if outcome != "ok":
                failures.append((task, r.get("instance"), r.get("system"), outcome))
        if rows:
            done, total = rows[-1].get("done", len(rows)), rows[-1].get("total", 0)
        per_task[task] = (done, total)
        newest = max(newest, f.stat().st_mtime)

    tot_done = sum(d for d, _ in per_task.values())
    tot_all = sum(t for _, t in per_task.values())
    finished = sum(1 for d, t in per_task.values() if t and d >= t)
    pct = 100 * tot_done / tot_all if tot_all else 0

    exp_runs, n_problems = expected_runs(a.manifest, root, a.systems_small, a.systems_large)

    started = f"{len(per_task)}" + (f" / {n_problems}" if n_problems else "")
    print(f"  tasks started      {started}")
    print(f"  tasks finished     {finished}")
    if exp_runs:
        print(f"  solver runs        {tot_done:,} / {exp_runs:,}  "
              f"({100 * tot_done / exp_runs:.1f}%)  OF THE CAMPAIGN")
        print(f"                     {tot_done:,} / {tot_all:,}  ({pct:.1f}%)  "
              f"of the {len(per_task)} started task(s)")
    else:
        print(f"  solver runs        {tot_done:,} / {tot_all:,}  ({pct:.1f}%)  "
              f"of the {len(per_task)} started task(s)")
        print(f"  campaign total     unknown -- {a.manifest} not found; pass --manifest")
    print(f"  last activity      {hms(time.time() - newest)} ago")
    print(f"\n  outcomes: " + "  ".join(f"{k}={v:,}" for k, v in outcomes.most_common()))
    if tot_all and tot_done:
        ok = outcomes.get("ok", 0)
        print(f"  solved within limits: {100 * ok / tot_done:.1f}% of finished runs")

    if a.detail:
        names = {k: display_name(k, a.full_names) for k in per_task}
        width = max([len("task")] + [len(n) for n in names.values()]) + 2
        print(f"\n  {'task':<{width}}{'done':>8}{'total':>8}  progress")
        for task, (d, t) in sorted(per_task.items(), key=lambda kv: (kv[1][0] / kv[1][1]) if kv[1][1] else 0):
            bar = "#" * int(20 * d / t) if t else ""
            print(f"  {names[task]:<{width}}{d:>8}{t:>8}  {bar:<20} {100 * d / t if t else 0:5.1f}%")

    if a.failures:
        if not failures:
            print("\n  no failures recorded")
        else:
            print(f"\n  {len(failures)} non-ok runs; by system:")
            by_sys = defaultdict(Counter)
            for _, _, system, outcome in failures:
                by_sys[system][outcome] += 1
            for system, c in sorted(by_sys.items(), key=lambda kv: -sum(kv[1].values())):
                print(f"    {system:<30} " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
            print("\n  first 15:")
            for task, inst, system, outcome in failures[:15]:
                print(f"    {outcome:<9} {system:<28} {inst:<22} {display_name(task, a.full_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
