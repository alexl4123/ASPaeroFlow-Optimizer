#!/usr/bin/env python3
"""Plan one round of the exact (delay, sectors) Pareto-front campaign and write its manifests.

The campaign (pareto_research/CAMPAIGN_DRAFT.md in the JOAS paper folder, decisions of 2026-09-24)
computes, per instance and variant, the front of (total arrival delay, active sectors) at the least
overload, by epsilon-constraint steps: sectors minimised under a hard bound K on the delay. One
step is one solver run is one manifest row is one SLURM array task (run_pareto_steps.slurm). The
rounds, each planned from the results of the previous one:

    E   both ends of every front: DF (delay-first order, no bound) and SF (sectors-first, no bound).
        A delay-first end the usc27 run already PROVED at horizon 24 is imported instead of re-run
        (--usc27-folder).
    G   a grid of bounds per front whose delay-first end is proven:
          both ends proven: K = D_min - 1 (the overload point; certifies the left end), and
                            K = D_S - 1 - j*stride for j >= 0 while K > D_min, with
                            stride = ceil((D_S - D_min - 1) / 24) -- every interior K when there
                            are at most 24;
          sectors-first end open: K = D_min - 1 and D_min + 1 .. D_min + 24;
          delay-first end open: nothing.
        A (group, flights) cell whose round-E pass rate (both ends proven) is below 10 % gets no G
        at all, and the ends-only flight counts (group A at 80 and 100) never get one.
    F   (optional, repeatable) for every uncertified stretch between two proven front points, the
        K values in it no step has tried yet, top of the stretch first, at most 12 per front;
        stops by itself when the previous F round certified nothing new, or when the rounds' worst
        case reaches 1,000 core-hours.
    P2  pilot only: the seven known-hard steps of the laptop sweeps, once each at 3600 s.

The pilot is the same rounds E and G in its own FOLDER with --preset pilot (V1 MAJOR-EUROPE-10x10
and the BSc central-europe instance, under `floored` and `signed`), plus P2.

Every step runs 1800 s (P2: 3600 s), single-threaded usc, fixed horizon, lower bound recorded.
Memory classes: 40G for group B (full rerouting), 8G for groups A and C; one manifest per round and
class, output/<FOLDER>/manifests/pareto_<round>_<class>.tsv. The script prints the exact sbatch
lines. It never overwrites a manifest (an array may be reading it); --check lists the rows of the
existing manifests that have no result yet, as --array ranges.

Usage (from 06_benchmark_start_script):
    ./build_pareto_manifest.py --folder F --round E --instance-root "$OPT/05_instances" \\
        [--usc27-folder output/USC27]
    ./build_pareto_manifest.py --folder F --round G --instance-root "$OPT/05_instances"
    ./build_pareto_manifest.py --folder F --round F --instance-root "$OPT/05_instances"
    ./build_pareto_manifest.py --folder F --check
    ./build_pareto_manifest.py --folder P --preset pilot --round E --instance-root <pilot root>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pareto_lib as pl  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ablation_manifest = _load("build_ablation_manifest")   # read_problems, list_instances, plan_waves


# ---------------------------------------------------------------------------------------------
# Which fronts exist
# ---------------------------------------------------------------------------------------------

class FrontSpec:
    __slots__ = ("label", "problem", "instance", "variant", "metric", "tg", "flights", "seed",
                 "group", "kind")

    def __init__(self, problem, instance, variant, metric, tg, kind):
        self.problem, self.instance, self.variant, self.metric = problem, instance, variant, metric
        self.tg, self.kind = str(tg), kind                   # kind: "full" or "ends"
        self.flights, self.seed = pl.parse_instance(instance)
        self.group = pl.group_of(variant)
        self.label = pl.make_label(problem, instance, variant, metric)

    @property
    def mem_class(self) -> str:
        return pl.mem_class_of(self.variant)

    def row(self, round_name, order, bound, time_limit) -> pl.Row:
        return pl.Row(round=round_name, mem_class=self.mem_class, label=self.label,
                      problem=self.problem, instance=self.instance, variant=self.variant,
                      metric=self.metric, time_granularity=self.tg, order=order, bound=bound,
                      time_limit=time_limit)


def campaign_fronts(instance_root: Path, flights: Optional[Sequence[int]] = None
                    ) -> List[FrontSpec]:
    """Table 1 of the draft on the V2 small family: every region, every seed.

    `flights` replaces the table's flight counts (each gets a full front); for tests and subsets.
    """
    problems = ablation_manifest.read_problems(instance_root / "problems.tsv")
    fronts = []
    for group in pl.GROUPS.values():
        for variant in group["variants"]:
            wanted = {size: "full" for size in pl.full_sizes(variant)}
            wanted.update({size: "ends" for size in pl.end_sizes(variant)})
            if flights:
                wanted = {size: "full" for size in flights}
            for prob in problems:
                name = prob["problem_dir"]
                for inst in ablation_manifest.list_instances(instance_root / name):
                    size, _ = pl.parse_instance(inst)
                    if size in wanted:
                        fronts.append(FrontSpec(name, inst, variant, pl.CAMPAIGN_METRIC,
                                                prob.get("time_granularity", "1"), wanted[size]))
    return fronts


def select(fronts: List[FrontSpec], args) -> List[FrontSpec]:
    """Apply --regions/--variants/--seeds/--metrics (and --flights for the pilot)."""
    def wanted(text):
        return {x.strip() for x in text.split(",") if x.strip()} if text else None
    regions, variants = wanted(args.regions), wanted(args.variants)
    seeds, metrics = wanted(args.seeds), wanted(args.metrics)
    flights = {int(x) for x in wanted(args.flights)} if args.flights else None
    return [f for f in fronts
            if (regions is None or pl.region_of(f.problem) in regions)
            and (variants is None or f.variant in variants)
            and (seeds is None or str(f.seed) in seeds)
            and (metrics is None or f.metric in metrics)
            and (flights is None or f.flights in flights)]


def with_results(fronts: List[FrontSpec], args) -> List[FrontSpec]:
    """The fronts round E produced step files for; says how many of the others are missing."""
    root = pl.steps_dir(args.output_root, args.folder)
    have = [f for f in fronts if (root / f.label).is_dir()]
    if len(have) < len(fronts):
        print(f"[warn] {len(fronts) - len(have)} of {len(fronts)} selected fronts have no step "
              f"files at all (E run on a subset, or not finished: run --check first)")
    return have


def pilot_fronts(instance_root: Path) -> List[FrontSpec]:
    fronts = []
    for problem, instance, variant, metric in pl.PILOT_FRONTS:
        if not (instance_root / problem / instance).is_dir():
            raise SystemExit(f"[ERROR] pilot instance {instance_root / problem / instance} not "
                             f"found; rsync it first (TODO-CLUSTER.md, section E1)")
        fronts.append(FrontSpec(problem, instance, variant, metric, 1, "full"))
    return fronts


# ---------------------------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------------------------

def import_usc27_end(spec: FrontSpec, usc27: Path, output_root: Path, folder: str) -> str:
    """Copy the usc27 run's delay-first end into this campaign's steps, if it is a proof at 24.

    usc27 ran the same 02_ASP program (same encoding, `signed`, clasp seed, usc, one thread,
    1800 s with the deadline 30 s before) in the encoding's own order. Where it PROVED its optimum
    at horizon 24 with SOLVER-HORIZON-FINAL true, that is exactly the result a fixed-horizon
    delay-first step would give, so it is imported, not recomputed. Returns what happened.
    """
    target = pl.step_path(output_root, folder, spec.label, "E", "DF")
    if target.exists():
        return "exists"
    if spec.metric != "signed":
        return "metric"
    hits = sorted(usc27.glob(f"U_usc_t1/*_ASP_{spec.variant}/{spec.problem}/{spec.instance}/"
                             f"individual_outputs/*.json"))
    if not hits:
        return "absent"
    try:
        lines = json.loads(hits[-1].read_text()).get("object", [])
    except (OSError, ValueError):
        return "unreadable"
    lines = [line for line in lines if isinstance(line, dict)]
    finals = [line for line in lines if "SOLVER-STOPPED-AT-DEADLINE" in line]
    if not finals:
        return "no result line"
    last = finals[-1]
    proven = (bool(last.get("COMPUTATION-FINISHED")) and "OVERLOAD" in last
              and last.get("SOLVER-MAX-TIME") == pl.HORIZON
              and last.get("SOLVER-HORIZON-FINAL") is True
              and last.get("ARRIVAL-DELAY-METRIC") == spec.metric)
    if not proven:
        return "not proven at 24"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".jsonl.part")
    tmp.write_text("".join(json.dumps(line) + "\n" for line in lines))
    target.with_suffix(".prov").write_text(
        f"imported_from={hits[-1].resolve()}\nround=E\nlabel={spec.label}\nstep=DF\n"
        f"order=delay-first\nbound=none\noutcome=imported_usc27\n")
    tmp.replace(target)
    return "imported"


def round_e(fronts, args) -> List[pl.Row]:
    rows, imported = [], defaultdict(int)
    for spec in fronts:
        df_needed = True
        if args.usc27_folder is not None:
            what = import_usc27_end(spec, args.usc27_folder, args.output_root, args.folder)
            imported[what] += 1
            df_needed = what not in ("imported", "exists")
        if df_needed:
            rows.append(spec.row("E", "delay-first", None, args.time_limit))
        rows.append(spec.row("E", "sectors-first", None, args.time_limit))
    if args.usc27_folder is not None:
        print("[usc27] delay-first ends: " + ", ".join(f"{k} {v}" for k, v in sorted(imported.items())))
    return rows


def tried_bounds(steps: Sequence[pl.Step]) -> set:
    return {s.bound for s in steps if s.bound is not None}


def pass_rates(fronts, results) -> Dict[Tuple[str, int], Tuple[int, int]]:
    """(group, flights) -> (fronts, fronts with both ends proven), over the full-front sizes."""
    cells: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
    for spec in fronts:
        if spec.kind != "full":
            continue
        ends = results[spec.label][1]
        cells[(spec.group, spec.flights)][0] += 1
        cells[(spec.group, spec.flights)][1] += int(ends["df_proven"] and ends["sf_proven"])
    return {k: (v[0], v[1]) for k, v in sorted(cells.items())}


def grid_bounds(ends: Dict, metric: str) -> List[int]:
    """Round G's K values for one front (see the module docstring)."""
    if not ends["df_proven"]:
        return []
    d_min = ends["d_min"]
    ks = []
    # K = D_min - 1 cannot be satisfied at all when delays are floored at 0 and D_min is 0.
    if not (metric == "floored" and d_min == 0):
        ks.append(d_min - 1)
    if ends["sf_proven"]:
        d_s = ends["d_s"]
        interior = d_s - d_min - 1
        if interior > 0:
            stride = math.ceil(interior / pl.GRID_CAP)
            k = d_s - 1
            while k > d_min:
                ks.append(k)
                k -= stride
    else:
        ks.extend(range(d_min + 1, d_min + pl.GRID_CAP + 1))
    return ks


def load_results(fronts, args):
    """label -> (Front, end summary, steps), from the step files written so far."""
    results = {}
    for spec in fronts:
        steps = pl.load_front_steps(pl.steps_dir(args.output_root, args.folder) / spec.label)
        front = pl.certify(spec.label, steps, spec.metric)
        results[spec.label] = (front, pl.end_summary(front), steps)
    return results


def pruned_cells(fronts, results, args, show: bool = True) -> set:
    """(group, flights) cells whose round-E pass rate is below 10 %: no G, and no F either."""
    rates = pass_rates(fronts, results)
    if show:
        print("\nround-E pass rate (both ends proven) per (group, flights):")
        print(f"  {'group':<6}{'flights':>8}{'fronts':>8}{'passed':>8}{'rate':>8}   G?")
    skip_cells = set()
    for (group, flights), (n, passed) in rates.items():
        rate = passed / n if n else 0.0
        ok = rate >= pl.PASS_RATE_MIN or args.no_prune
        if not ok:
            skip_cells.add((group, flights))
        if show:
            print(f"  {group:<6}{flights:>8}{n:>8}{passed:>8}{rate:>8.0%}   "
                  f"{'yes' if ok else 'NO (below 10 %)'}")
    return skip_cells


def round_g(fronts, args) -> List[pl.Row]:
    results = load_results(fronts, args)
    skip_cells = pruned_cells(fronts, results, args)
    rows, kinds = [], defaultdict(int)
    for spec in fronts:
        if spec.kind != "full" or (spec.group, spec.flights) in skip_cells:
            continue
        front, ends, steps = results[spec.label]
        if not ends["df_proven"]:
            kinds["delay-first end open: no G"] += 1
            continue
        kinds["both ends proven" if ends["sf_proven"] else "sectors-first end open"] += 1
        done = tried_bounds(steps)
        for k in grid_bounds(ends, spec.metric):
            if k not in done:
                rows.append(spec.row("G", "sectors-first", k, args.time_limit))
    print("\nfronts: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    return rows


def _manifests(args, round_prefix: str) -> List[Path]:
    return sorted((args.output_root / args.folder / "manifests").glob(f"pareto_{round_prefix}*.tsv"))


def certified_total(results) -> int:
    return sum(sum(front.stretch_ok) + int(front.left_ok) + int(front.right_ok)
               for front, _, _ in results.values())


def round_f(fronts, args) -> Tuple[List[pl.Row], str, Dict]:
    results = load_results(fronts, args)
    mdir = args.output_root / args.folder / "manifests"
    previous = sorted(mdir.glob("pareto_F*.plan.json"))
    index = len(previous) + 1
    now = certified_total(results)
    spent = 0.0
    for plan in previous:
        info = json.loads(plan.read_text())
        spent += float(info.get("worst_core_h", 0.0))
    if previous and not args.force:
        last = json.loads(previous[-1].read_text())
        if now <= int(last.get("certified_at_planning", -1)):
            print(f"[stop] F{index - 1} certified nothing new ({now} certified stretches and ends, "
                  f"{last.get('certified_at_planning')} before it). No F{index}.")
            return [], f"F{index}", {}
    budget_rows = int(max(0.0, args.fill_budget - spent) * 3600 // args.time_limit)
    skip_cells = pruned_cells(fronts, results, args, show=False)
    per_front: Dict[str, List[pl.Row]] = {}
    for spec in fronts:
        if spec.kind != "full" or (spec.group, spec.flights) in skip_cells:
            continue
        front, _, steps = results[spec.label]
        if front.cls in ("exact", "none") or not front.o_known:
            continue
        done = tried_bounds(steps)
        todo = []
        for i, ok in enumerate(front.stretch_ok):
            if ok or not (front.point_proven[i] and front.point_proven[i + 1]):
                continue
            (d0, _), (d1, _) = front.corners[i], front.corners[i + 1]
            todo.extend(k for k in range(d1 - 1, d0, -1) if k not in done)
        if todo:
            per_front[spec.label] = [spec.row("F", "sectors-first", k, args.time_limit)
                                     for k in todo[:pl.FILL_CAP]]
    # Round-robin over fronts, so a budget cut keeps the top of every front's first stretch.
    rows, depth = [], 0
    while any(depth < len(v) for v in per_front.values()):
        for label in sorted(per_front):
            if depth < len(per_front[label]):
                rows.append(per_front[label][depth])
        depth += 1
    if len(rows) > budget_rows:
        print(f"[budget] {len(rows)} fill rows, {budget_rows} fit in the remaining "
              f"{args.fill_budget - spent:.0f} of {args.fill_budget:.0f} core-hours; cut")
        rows = rows[:budget_rows]
    plan = {"certified_at_planning": now, "rows": len(rows),
            "worst_core_h": len(rows) * args.time_limit / 3600.0, "spent_before": spent}
    return rows, f"F{index}", plan


def round_p2(args) -> List[pl.Row]:
    rows = []
    for problem, instance, variant, metric, k in pl.PILOT_P2:
        spec = FrontSpec(problem, instance, variant, metric, 1, "full")
        if not (args.instance_root / problem / instance).is_dir():
            raise SystemExit(f"[ERROR] pilot instance {args.instance_root / problem / instance} "
                             f"not found")
        rows.append(spec.row("P2", "sectors-first", k, pl.P2_TIME_LIMIT))
    return rows


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def walltime(limit_s: int) -> str:
    """The -t request: the step's limit plus ten minutes for start-up and teardown."""
    minutes = math.ceil(limit_s / 60) + 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def sbatch_lines(manifest: Path, rows: Sequence[pl.Row], args, ids: Optional[Sequence[int]] = None
                 ) -> List[str]:
    mem = rows[0].mem_class
    limit = max(r.time_limit for r in rows)
    throttle = args.throttle_40g if mem == "40G" else args.throttle_8g
    exclude = f" -x {args.exclude}" if args.exclude else ""
    head = (f"sbatch -J pareto -p {args.partition}{exclude} -t {walltime(limit)} "
            f"--cpus-per-task=2 --mem={pl.MEM_CLASSES[mem]['sbatch_mem']}")
    root = args.instance_root_text
    lines = []
    if ids is None:
        waves = ablation_manifest.plan_waves(1, len(rows), 1, args.max_array_size)
        for offset, n in waves:
            lines.append(f"{head} --export=ALL,FOLDER={args.folder},INSTANCE_ROOT={root},"
                         f"MANIFEST={manifest},CHUNK=1,ROW_OFFSET={offset} "
                         f"--array=1-{n}%{throttle} run_pareto_steps.slurm")
    else:
        cap = ablation_manifest.max_array_tasks(args.max_array_size)
        by_wave: Dict[int, List[int]] = defaultdict(list)
        for row_id in ids:
            by_wave[(row_id - 1) // cap * cap].append(row_id)
        for offset, members in sorted(by_wave.items()):
            lines.append(f"{head} --export=ALL,FOLDER={args.folder},INSTANCE_ROOT={root},"
                         f"MANIFEST={manifest},CHUNK=1,ROW_OFFSET={offset} "
                         f"--array={pl.ranges(i - offset for i in members)}%{throttle} "
                         f"run_pareto_steps.slurm")
    return lines


def write_round(round_name: str, rows: List[pl.Row], args, plan: Optional[Dict] = None) -> None:
    if not rows:
        print(f"\nround {round_name}: nothing to run.")
        return
    mdir = args.output_root / args.folder / "manifests"
    by_class: Dict[str, List[pl.Row]] = defaultdict(list)
    for row in rows:
        by_class[row.mem_class].append(row)
    paths = {mem: mdir / f"pareto_{round_name}_{mem}.tsv" for mem in by_class}
    clash = [str(p) for p in paths.values() if p.exists()]
    if clash:
        raise SystemExit("[ERROR] refusing to overwrite " + ", ".join(clash) + ": an array may be "
                         "reading it. Use --check for the rows still missing, or delete it by hand "
                         "if it was never submitted.")
    print(f"\nround {round_name}: {len(rows)} steps")
    print(f"  {'class':<6}{'rows':>7}{'worst core-h':>14}   manifest")
    for mem in sorted(by_class):
        pl.write_manifest(paths[mem], by_class[mem])
        core_h = sum(r.time_limit for r in by_class[mem]) / 3600.0
        print(f"  {mem:<6}{len(by_class[mem]):>7}{core_h:>14,.0f}   {paths[mem]}")
    if plan is not None:
        (mdir / f"pareto_{round_name}.plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # --check prints resubmission lines later and needs the same INSTANCE_ROOT.
    (mdir / "instance_root.txt").write_text(args.instance_root_text + "\n")
    print("\n  Worst case = every step uses its full limit; solver core-hours (the allocation is "
          "twice that at 2 cpus per task).")
    print("\nSubmit (nothing is submitted by this script); one array per memory class:")
    for mem in sorted(by_class):
        for line in sbatch_lines(paths[mem], by_class[mem], args):
            print("  " + line)


def check(args) -> int:
    mdir = args.output_root / args.folder / "manifests"
    pattern = f"pareto_{args.round}*_*.tsv" if args.round else "pareto_*.tsv"
    manifests = sorted(mdir.glob(pattern))
    if not manifests:
        print(f"[check] no manifests matching {mdir / pattern}")
        return 1
    missing_total = 0
    for manifest in manifests:
        rows = pl.read_manifest(manifest)
        missing, killed, outcomes = [], [], defaultdict(int)
        for row in rows:
            path = pl.step_path(args.output_root, args.folder, row["label"], row["round"],
                                row["step"])
            if not path.exists():
                missing.append(int(row["row_id"]))
                continue
            step = pl.load_step(path)
            outcomes[step.status] += 1
            if step.final is None:
                killed.append(int(row["row_id"]))
        missing_total += len(missing)
        rel = manifest
        print(f"\n{rel}: {len(rows)} rows, {len(rows) - len(missing)} with a result, "
              f"{len(missing)} without")
        if outcomes:
            print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
        if killed:
            print(f"  no result line (killed at the limit or out of memory; read the .err/.prov "
                  f"files, do not resubmit blindly): rows {pl.ranges(killed)}")
        if missing:
            print(f"  missing rows: {pl.ranges(missing)}")
            print("  read their logs (logs/pareto-<jobid>_<task>.out) first, then resubmit:")
            specs = [pl.Row(round=r["round"], mem_class=r["mem_class"], label=r["label"],
                            problem=r["problem"], instance=r["instance"], variant=r["variant"],
                            metric=r["metric"], time_granularity=r["time_granularity"],
                            order=r["order"], bound=None if r["bound"] == "none" else int(r["bound"]),
                            time_limit=int(r["time_limit"])) for r in rows]
            for line in sbatch_lines(rel, specs, args, ids=missing):
                print("    " + line)
    return 0 if missing_total == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, help="Campaign folder under output/")
    parser.add_argument("--round", type=str, default=None, choices=["E", "G", "F", "P2"],
                        help="Round to plan (with --check: only this round's manifests)")
    parser.add_argument("--preset", choices=["campaign", "pilot"], default="campaign")
    parser.add_argument("--instance-root", type=str, default=None,
                        help="The 05_instances tree (campaign) or the pilot instance root. Passed "
                             "to the runner as INSTANCE_ROOT, so give it as an absolute path. "
                             "--check reuses the one the manifests were planned with.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--usc27-folder", type=Path, default=None,
                        help="Round E: import the delay-first ends the usc27 run proved at "
                             "horizon 24 (e.g. output/USC27 of the usc27 clone)")
    parser.add_argument("--time-limit", type=int, default=pl.STEP_TIME_LIMIT,
                        help="Seconds per step in E, G and F (default 1800, as decided)")
    parser.add_argument("--no-prune", action="store_true",
                        help="Round G: ignore the 10 %% pass-rate rule")
    parser.add_argument("--force", action="store_true",
                        help="Round F: plan even when the previous F round certified nothing new")
    parser.add_argument("--fill-budget", type=float, default=pl.FILL_BUDGET_CORE_H,
                        help="Round F: worst-case core-hours over all F rounds (default 1000)")
    parser.add_argument("--partition", default="sunnycove")
    parser.add_argument("--exclude", default="coppernode[25-28]",
                        help="Nodes to leave out (default: the Gurobi nodes)")
    parser.add_argument("--throttle-8g", type=int, default=60)
    parser.add_argument("--throttle-40g", type=int, default=24)
    parser.add_argument("--max-array-size", type=int, default=50000,
                        help="The cluster's MaxArraySize (50000); the highest index is one less")
    parser.add_argument("--regions", default=None,
                        help="Only these regions (comma-separated, e.g. EAST-ASIA-3x3)")
    parser.add_argument("--variants", default=None, help="Only these variants (comma-separated)")
    parser.add_argument("--seeds", default=None, help="Only these seeds (comma-separated)")
    parser.add_argument("--metrics", default=None, help="Only these metrics (pilot: floored,signed)")
    parser.add_argument("--flights", default=None,
                        help="Campaign: REPLACE the table's flight counts by these (each a full "
                             "front) -- for tests and subsets. Pilot: only these.")
    parser.add_argument("--check", action="store_true",
                        help="List the rows of the existing manifests that have no result yet")
    args = parser.parse_args()

    if args.instance_root is None:
        recorded = args.output_root / args.folder / "manifests" / "instance_root.txt"
        args.instance_root = (recorded.read_text().strip() if recorded.exists()
                              else "../05_instances")
    args.instance_root_text = args.instance_root
    args.instance_root = Path(args.instance_root)
    if args.check:
        return check(args)
    if args.round is None:
        parser.error("--round is required (or --check)")
    if args.preset == "campaign" and args.round == "P2":
        parser.error("P2 is part of the pilot: --preset pilot")
    if not args.instance_root.is_dir():
        raise SystemExit(f"[ERROR] instance root {args.instance_root} does not exist")

    if args.round == "P2":
        write_round("P2", round_p2(args), args)
        return 0
    flights = [int(x) for x in args.flights.split(",")] if args.flights else None
    fronts = (pilot_fronts(args.instance_root) if args.preset == "pilot"
              else campaign_fronts(args.instance_root, flights))
    fronts = select(fronts, args)
    if args.round in ("G", "F"):
        fronts = with_results(fronts, args)
    kinds = defaultdict(int)
    for spec in fronts:
        kinds[(spec.group, spec.kind)] += 1
    print(f"[{args.preset}] {len(fronts)} fronts: " +
          ", ".join(f"group {g} {k} {n}" for (g, k), n in sorted(kinds.items())))

    if args.round == "E":
        write_round("E", round_e(fronts, args), args)
    elif args.round == "G":
        write_round("G", round_g(fronts, args), args)
    elif args.round == "F":
        rows, name, plan = round_f(fronts, args)
        write_round(name, rows, args, plan if rows else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
