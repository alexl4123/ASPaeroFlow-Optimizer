#!/usr/bin/env python3
"""
How far along is the benchmark? Run this after SSH-ing in.

Each array task appends to output/<FOLDER>/output_<PROBLEM>/progress.jsonl; this reads all of
them and summarises. Nothing here touches the running jobs.

    ./benchmark_progress.py                      # newest output folder
    ./benchmark_progress.py --folder 20260911_V2
    ./benchmark_progress.py --detail             # per-task table
    ./benchmark_progress.py --failures           # what went wrong, and where
"""
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path


def hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--folder", default=None, help="default: most recently modified")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--failures", action="store_true")
    a = ap.parse_args()

    root = a.output_root / a.folder if a.folder else None
    if root is None:
        cands = [p for p in a.output_root.iterdir() if p.is_dir()] if a.output_root.is_dir() else []
        if not cands:
            raise SystemExit(f"no output folders under {a.output_root}")
        root = max(cands, key=lambda p: p.stat().st_mtime)
    print(f"run folder: {root}\n")

    files = sorted(root.glob("output_*/progress.jsonl"))
    if not files:
        raise SystemExit(f"no progress.jsonl under {root} -- nothing has started yet")

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
            outcomes[r.get("outcome", "?")] += 1
            if r.get("outcome") not in ("ok", None):
                failures.append((task, r.get("instance"), r.get("system"), r.get("outcome")))
        if rows:
            done, total = rows[-1].get("done", len(rows)), rows[-1].get("total", 0)
        per_task[task] = (done, total)
        newest = max(newest, f.stat().st_mtime)

    tot_done = sum(d for d, _ in per_task.values())
    tot_all = sum(t for _, t in per_task.values())
    finished = sum(1 for d, t in per_task.values() if t and d >= t)
    pct = 100 * tot_done / tot_all if tot_all else 0

    print(f"  tasks started      {len(per_task)}")
    print(f"  tasks finished     {finished}")
    print(f"  solver runs        {tot_done:,} / {tot_all:,}  ({pct:.1f}%)")
    print(f"  last activity      {hms(time.time() - newest)} ago")
    print(f"\n  outcomes: " + "  ".join(f"{k}={v:,}" for k, v in outcomes.most_common()))
    if tot_all and tot_done:
        ok = outcomes.get("ok", 0)
        print(f"  solved within limits: {100 * ok / tot_done:.1f}% of finished runs")

    if a.detail:
        print(f"\n  {'task':<62}{'done':>8}{'total':>8}  progress")
        for task, (d, t) in sorted(per_task.items(), key=lambda kv: (kv[1][0] / kv[1][1]) if kv[1][1] else 0):
            bar = "#" * int(20 * d / t) if t else ""
            print(f"  {task[:60]:<62}{d:>8}{t:>8}  {bar:<20} {100 * d / t if t else 0:5.1f}%")

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
                print(f"    {outcome:<9} {system:<28} {inst:<22} {task[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
