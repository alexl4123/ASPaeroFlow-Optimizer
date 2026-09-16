#!/usr/bin/env python3
"""Build the job index for the ASP solver-option ablation campaign.

The ablation asks ONE question: which clingo search configuration should someone running these
benchmarks actually use, and what does that choice cost or buy? It answers it by crossing the
four named profiles from common/clingo_options.py

    default  usc  domain  usc-domain

with the exact-ASP systems on the SMALL-scaling family -- the family that exists precisely so
exact methods can close instances.

WHY A MANIFEST AND NOT A NESTED LOOP INSIDE THE SLURM SCRIPT
run_all_benchmarks.slurm indexes a job array into ../05_instances/problems.tsv, one array task
per problem directory. This campaign follows that convention but indexes ONE INSTANCE, not one
problem directory: a task here is (tier, profile, threads, variant set, problem, instance), so a
task runs at most 27 solver runs and usually 2. That keeps every job short enough for a cluster
that objects to long-running scripts, and it is the shape the per-run job restructuring wants.

The row is the unit of work. run_asp_ablation.slurm reads exactly one row and does exactly what
it says, so re-running a single task is `sbatch --array=<task_id>` and nothing else.

TIERS. Each is a separate sbatch submission with its own -t and --cpus-per-task, because their
job shapes differ. The script prints the exact submission lines for the manifest it just wrote.

    P  preflight   4 jobs.   One instance, four profiles, short limit. Proves the flags reach
                   clingo and that --solver-stats lands in the recorded JSON before anything
                   large is launched.
    A  main grid   4 profiles x every small instance, running the two exact configurations the
                   large family also runs (05_ASP_rp_dp_sp, 05_ASP_rp_d_sp). This tier produces
                   the recommendation.
    B  breadth     4 profiles x all 27 exact-ASP variants on a small stratified subsample.
                   Answers "does the tier-A recommendation generalise across the variant grid?"
                   and carries a built-in control: the encoding's only live #heuristic directive
                   grounds to nothing unless dynamic sectorisation is PARTIAL, so `domain` must
                   behave exactly like `default` on every _ns and _s variant. If it does not,
                   something in the pipeline is wrong.
    C  threads     A deliberately small thread probe, {default, usc-domain} at 4 threads on one
                   region, compared against the 1-thread cells tier A already has. Separate
                   because 4 threads changes what is being measured -- see the design document.

WHY tier A RUNS ONLY THE TWO NAMED VARIANTS
Not only because they are the best-performing exact configurations. The benchmark caller cannot
express any other subset: the per-variant flags (--experiment-asp-r-d-s and friends) are parsed
and then never read in build_system_config(), so the exact-ASP systems are selectable only as
"all 27" or as those two named singles. A four-variant tier A is not submittable today.

Usage:
    ./build_ablation_manifest.py                          # reads ../05_instances/problems.tsv
    ./build_ablation_manifest.py --tiers P                # preflight rows only
    ./build_ablation_manifest.py --assume-grid            # size the campaign before the data
                                                          # has been regenerated
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

#: Columns of the emitted manifest. run_asp_ablation.slurm reads them positionally.
COLUMNS = (
    "task_id", "tier", "profile", "threads", "variants",
    "problem", "instance", "time_granularity", "runs",
)

#: Profiles under test. All four, because that is the ablation.
ALL_PROFILES = ("default", "usc", "domain", "usc-domain")

#: How many solver runs one job performs, per variant set. "named2" is the two exact
#: configurations run_all_benchmarks.slurm also puts on the large family; "all27" is the full
#: ground-delay x rerouting x dynamic-sectorisation cross.
RUNS_PER_VARIANT_SET = {"named2": 2, "all27": 27}

#: capacity_level of the small-scaling family in problems.tsv. That family has no PCAP sweep,
#: so the expansion script writes NONE, and this is how the two families are told apart.
SMALL_FAMILY_CAP_LEVEL = "NONE"


def read_problems(manifest: Path) -> List[Dict[str, str]]:
    """The small-scaling rows of ../05_instances/problems.tsv, in file order."""
    if not manifest.exists():
        raise SystemExit(
            f"[ERROR] {manifest} not found. Run ASPaeroFlow-DataGenerator/"
            "expand_instances_for_benchmark.sh first, or pass --assume-grid to size the "
            "campaign before the data exists."
        )
    rows: List[Dict[str, str]] = []
    with manifest.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row.get("capacity_level") == SMALL_FAMILY_CAP_LEVEL:
                rows.append(row)
    if not rows:
        raise SystemExit(
            f"[ERROR] no small-scaling rows (capacity_level={SMALL_FAMILY_CAP_LEVEL}) in "
            f"{manifest}. The ablation is defined on the small family only."
        )
    return rows


def list_instances(problem_dir: Path) -> List[str]:
    """Instance directory names inside one problem directory, sorted as the caller sorts them."""
    if not problem_dir.is_dir():
        raise SystemExit(f"[ERROR] problem directory {problem_dir} does not exist")
    return sorted(p.name for p in problem_dir.iterdir() if p.is_dir())


def synthetic_instances(sizes: Sequence[int], seeds: Sequence[int]) -> List[str]:
    """Instance names the generator WOULD produce, for sizing the campaign before it runs.

    The generator names an instance <7-digit flight count>_SEED<seed>, e.g. 0000040_SEED42.
    """
    return sorted(f"{size:07d}_SEED{seed}" for size in sizes for seed in seeds)


def parse_instance(name: str):
    """(flight_count, seed) from an instance directory name, or (None, None)."""
    try:
        size_text, seed_text = name.split("_SEED", 1)
        return int(size_text), int(seed_text)
    except (ValueError, AttributeError):
        return None, None


def select(instances: Sequence[str], sizes, seeds) -> List[str]:
    """Instances whose flight count and seed are in the given sets (None = no filter)."""
    kept = []
    for name in instances:
        size, seed = parse_instance(name)
        if sizes is not None and size not in sizes:
            continue
        if seeds is not None and seed not in seeds:
            continue
        kept.append(name)
    return kept


def build_rows(problems, instances_of, args) -> List[List[str]]:
    """One row per array task, grouped by tier so each tier is a contiguous --array range."""
    rows: List[List[str]] = []

    def emit(tier, profile, threads, variants, problem, instance, granularity):
        rows.append([
            str(len(rows) + 1), tier, profile, str(threads), variants,
            problem, instance, str(granularity), str(RUNS_PER_VARIANT_SET[variants]),
        ])

    for tier in args.tiers:
        for prob in problems:
            name = prob["problem_dir"]
            granularity = prob.get("time_granularity", "1")
            available = instances_of[name]

            if tier == "P":
                # Preflight: the first region only, its smallest instance, all four profiles.
                if prob is not problems[0]:
                    continue
                chosen = select(available, {min(args.sizes_all)}, {args.preflight_seed})[:1]
                for profile in ALL_PROFILES:
                    for inst in chosen:
                        emit("P", profile, 1, "named2", name, inst, granularity)

            elif tier == "A":
                for profile in ALL_PROFILES:
                    for inst in available:
                        emit("A", profile, 1, "named2", name, inst, granularity)

            elif tier == "B":
                chosen = select(available, set(args.tier_b_sizes), {args.tier_b_seed})
                for profile in ALL_PROFILES:
                    for inst in chosen:
                        emit("B", profile, 1, "all27", name, inst, granularity)

            elif tier == "C":
                if args.tier_c_region not in name:
                    continue
                for profile in args.tier_c_profiles:
                    for inst in available:
                        emit("C", profile, args.tier_c_threads, "named2", name, inst, granularity)

    return rows


def summarise(rows, time_limit: int) -> None:
    """Run counts, worst-case core-hours and the submission lines, per tier.

    Core-hours are counted on SOLVER cores (threads actually used), and separately on the
    ALLOCATION (--cpus-per-task), because those differ and the queue charges the second one.
    """
    tiers: Dict[str, Dict[str, float]] = {}
    for row in rows:
        tier, threads, runs = row[1], int(row[3]), int(row[8])
        acc = tiers.setdefault(tier, {"jobs": 0, "runs": 0, "core_h": 0.0, "threads": threads,
                                      "first": int(row[0]), "last": int(row[0])})
        acc["jobs"] += 1
        acc["runs"] += runs
        acc["core_h"] += runs * threads * time_limit / 3600.0
        acc["first"] = min(acc["first"], int(row[0]))
        acc["last"] = max(acc["last"], int(row[0]))

    cpus = {"P": 2, "A": 2, "B": 2, "C": 5}
    walltime = {"P": "00:30:00", "A": "02:00:00", "B": "16:00:00", "C": "02:00:00"}
    label = {"P": "preflight", "A": "main grid", "B": "variant breadth", "C": "thread probe"}

    print()
    print(f"{'tier':<6}{'jobs':>7}{'runs':>8}{'worst-case core-h':>20}   array range")
    print("-" * 72)
    total_jobs = total_runs = 0
    total_core_h = 0.0
    for tier in ("P", "A", "B", "C"):
        if tier not in tiers:
            continue
        acc = tiers[tier]
        print(f"{tier:<6}{acc['jobs']:>7}{acc['runs']:>8}{acc['core_h']:>20,.0f}"
              f"   {acc['first']}-{acc['last']}   ({label[tier]})")
        total_jobs += acc["jobs"]
        total_runs += acc["runs"]
        total_core_h += acc["core_h"]
    print("-" * 72)
    print(f"{'ALL':<6}{total_jobs:>7}{total_runs:>8}{total_core_h:>20,.0f}")
    print(f"\nWorst case assumes every run uses the full {time_limit} s. The small family is "
          f"built so exact\nmethods CLOSE instances, so the real cost is well below this -- see "
          f"the design document\nfor the expected-case arithmetic.")

    print("\nSubmit one tier at a time (nothing is submitted by this script):")
    for tier in ("P", "A", "B", "C"):
        if tier not in tiers:
            continue
        acc = tiers[tier]
        print(f"  # tier {tier} -- {label[tier]}")
        print(f"  sbatch -t {walltime[tier]} --cpus-per-task={cpus[tier]} "
              f"--array={acc['first']}-{acc['last']}%40 run_asp_ablation.slurm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance-root", type=Path, default=Path("../05_instances"),
                        help="Where the expansion script put the instances (default ../05_instances)")
    parser.add_argument("--out", type=Path, default=Path("ablation_tasks.tsv"),
                        help="Manifest to write (default ablation_tasks.tsv)")
    parser.add_argument("--tiers", type=str, default="P,A,B,C",
                        help="Comma-separated tiers to emit, in order. Default P,A,B,C.")
    parser.add_argument("--time-limit", type=int, default=1800,
                        help="Per-run limit used ONLY for the cost estimate printed here "
                             "(default 1800, the value the papers use)")

    parser.add_argument("--tier-b-sizes", type=str, default="10,20,30,40",
                        help="Flight counts in the tier-B subsample (default 10,20,30,40: the "
                             "sizes where 27 variants x 4 profiles is affordable)")
    parser.add_argument("--tier-b-seed", type=int, default=11904657,
                        help="Single seed for tier B (default 11904657, the generator's own)")
    parser.add_argument("--tier-c-region", type=str, default="CENTRAL-EUROPE",
                        help="Substring of the problem directory used for the thread probe")
    parser.add_argument("--tier-c-profiles", type=str, default="default,usc-domain",
                        help="Profiles in the thread probe (default: the two extremes)")
    parser.add_argument("--tier-c-threads", type=int, default=4,
                        help="clasp search threads for the probe (default 4, the count measured "
                             "at 0.18 s on an instance one thread did not close in 25 s)")
    parser.add_argument("--preflight-seed", type=int, default=11904657)

    parser.add_argument("--assume-grid", action="store_true",
                        help="Do not read the instance tree; assume the generator's grid. Use "
                             "this to size the campaign BEFORE the data is regenerated.")
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated problem directory names, for --assume-grid without "
                             "a problems.tsv")
    parser.add_argument("--sizes", type=str, default="10,20,30,40,50,60,70,80,90,100",
                        help="Flight counts in the small family (used by --assume-grid)")
    parser.add_argument("--seeds", type=str, default="42,11904657,150699,13",
                        help="Seeds in the small family (used by --assume-grid)")
    args = parser.parse_args()

    args.tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    for tier in args.tiers:
        if tier not in ("P", "A", "B", "C"):
            raise SystemExit(f"[ERROR] unknown tier {tier!r}; expected P, A, B or C")
    args.tier_b_sizes = [int(x) for x in args.tier_b_sizes.split(",") if x.strip()]
    args.tier_c_profiles = [p.strip() for p in args.tier_c_profiles.split(",") if p.strip()]
    for profile in args.tier_c_profiles:
        if profile not in ALL_PROFILES:
            raise SystemExit(f"[ERROR] unknown profile {profile!r}; expected one of "
                             f"{', '.join(ALL_PROFILES)}")
    args.sizes_all = [int(x) for x in args.sizes.split(",") if x.strip()]
    seeds_all = [int(x) for x in args.seeds.split(",") if x.strip()]

    if args.assume_grid:
        if args.problems:
            names = [p.strip() for p in args.problems.split(",") if p.strip()]
        else:
            manifest = args.instance_root / "problems.tsv"
            names = [r["problem_dir"] for r in read_problems(manifest)]
        problems = [{"problem_dir": n, "time_granularity": "1"} for n in names]
        grid = synthetic_instances(args.sizes_all, seeds_all)
        instances_of = {n: list(grid) for n in names}
        print(f"[assume-grid] {len(names)} problem directories x {len(grid)} assumed instances")
    else:
        problems = read_problems(args.instance_root / "problems.tsv")
        instances_of = {}
        for prob in problems:
            instances_of[prob["problem_dir"]] = list_instances(
                args.instance_root / prob["problem_dir"])
        counts = ", ".join(f"{p['problem_dir']}={len(instances_of[p['problem_dir']])}"
                           for p in problems)
        print(f"[small family] {len(problems)} problem directories: {counts}")

    rows = build_rows(problems, instances_of, args)
    if not rows:
        raise SystemExit("[ERROR] no tasks generated -- check --tiers and the subsample filters")

    with args.out.open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    print(f"wrote {args.out} ({len(rows)} array tasks)")

    summarise(rows, args.time_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
