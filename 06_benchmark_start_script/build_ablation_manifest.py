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
#: `system` is "-" for "every system the variant set enables", or a single system key once the
#: caller can isolate one (see --per-run-systems).
COLUMNS = (
    "task_id", "tier", "profile", "threads", "variants",
    "problem", "instance", "time_granularity", "system", "runs",
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

#: The two exact configurations run_all_benchmarks.slurm also puts on the large family, under
#: the keys build_system_config() gives them.
NAMED2_SYSTEMS = ("05_ASP_rp_dp_sp", "05_ASP_rp_d_sp")


def all27_systems() -> List[str]:
    """The 27 keys --experiment-all-asp-variants builds, in the order it builds them.

    Mirrored from build_system_config(): ground delay is the OUTER loop, then rerouting, then
    dynamic sectorisation, and the running index starts at 5. Verified against the caller itself
    with --verify-system-keys, which is the answer to "what if that loop changes".
    """
    ground = {0: "nd", 1: "dp", 2: "d"}
    reroute = {0: "nr", 1: "rp", 2: "r"}
    sector = {0: "ns", 1: "sp", 2: "s"}
    keys, index = [], 5
    for g in (0, 1, 2):
        for r in (0, 1, 2):
            for s in (0, 1, 2):
                keys.append(f"{index}_ASP_{reroute[r]}_{ground[g]}_{sector[s]}")
                index += 1
    return keys


def verify_system_keys() -> int:
    """Compare all27_systems() with what start_benchmark_caller.py actually builds."""
    import types
    from types import SimpleNamespace
    import importlib.util

    sys.modules.setdefault("psutil", types.ModuleType("psutil"))  # import-time only
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_caller_for_keys", here / "start_benchmark_caller.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = SimpleNamespace(results_format="csv", wandb_enabled="False",
                           experiment_all_asp_variants=1,
                           experiment_asp_rp_dp_sp=1, experiment_asp_rp_d_sp=1)
    for name in ("experiment_asp_aero_flow", "experiment_asp_aero_flow_no_convex",
                 "experiment_asp_aero_flow_nr_nd", "experiment_asp_aero_flow_nr_d",
                 "experiment_asp_aero_flow_r_nd", "experiment_casa",
                 "experiment_route_delay", "experiment_route", "experiment_delay",
                 "experiment_mip"):
        setattr(args, name, 0)
    built = [s["key"] for s in module.build_system_config(here, Path("/tmp"), "verify", args)]
    expected = all27_systems() + list(NAMED2_SYSTEMS)
    if built == expected:
        print(f"[ok] system keys match the caller ({len(built)} systems)")
        return 0
    print("[MISMATCH] all27_systems() disagrees with build_system_config()", file=sys.stderr)
    print(f"  caller:   {built}", file=sys.stderr)
    print(f"  mirrored: {expected}", file=sys.stderr)
    return 1


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
    """One row per unit of work, grouped by tier so each tier is a contiguous --array range."""
    rows: List[List[str]] = []

    def emit(tier, profile, threads, variants, problem, instance, granularity):
        # --per-run-systems splits a variant set into one row per solver run, which is the shape
        # the per-run job restructuring wants. It needs the caller's --only-system selector; the
        # runner refuses the row rather than silently running all 27 if that flag is absent.
        systems = ["-"]
        if args.per_run_systems:
            systems = list(NAMED2_SYSTEMS) if variants == "named2" else all27_systems()
        for system in systems:
            runs = 1 if system != "-" else RUNS_PER_VARIANT_SET[variants]
            rows.append([
                str(len(rows) + 1), tier, profile, str(threads), variants,
                problem, instance, str(granularity), system, str(runs),
            ])

    def pad_to_chunk_boundary():
        """Start every tier on a fresh array task.

        Tiers are submitted separately and tier C asks for five CPUs rather than two, so an array
        task straddling two tiers would run a four-thread unit inside a two-CPU allocation. With
        CHUNK=1 this never pads; with CHUNK>1 it inserts rows the runner skips.
        """
        while args.chunk > 1 and len(rows) % args.chunk != 0:
            rows.append([str(len(rows) + 1), "PAD", "-", "0", "-", "-", "-", "-", "-", "0"])

    for tier in args.tiers:
        pad_to_chunk_boundary()
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


def summarise(rows, time_limit: int, chunk: int, max_array: int) -> None:
    """Row counts, worst-case core-hours and the submission lines, per tier.

    Core-hours are counted on SOLVER cores (the threads a run actually uses). The ALLOCATION is
    larger -- the queue charges --cpus-per-task, which is 2 for a single-threaded run -- and the
    design document states both.

    CHUNK rows share one array task, so the array range is in chunks and the printed job count
    is the number of array tasks, not the number of rows.
    """
    tiers: Dict[str, Dict[str, float]] = {}
    for row in rows:
        tier, threads, runs = row[1], int(row[3]), int(row[9])
        acc = tiers.setdefault(tier, {"rows": 0, "runs": 0, "core_h": 0.0, "threads": threads,
                                      "first": int(row[0]), "last": int(row[0])})
        acc["rows"] += 1
        acc["runs"] += runs
        acc["core_h"] += runs * threads * time_limit / 3600.0
        acc["first"] = min(acc["first"], int(row[0]))
        acc["last"] = max(acc["last"], int(row[0]))

    cpus = {"P": 2, "A": 2, "B": 2, "C": 5}
    label = {"P": "preflight", "A": "main grid", "B": "variant breadth", "C": "thread probe"}

    def hours_per_task(acc):
        """Worst-case wall time of ONE array task, which is what the -t request has to cover."""
        runs_per_row = acc["runs"] / max(1, acc["rows"])
        return runs_per_row * chunk * time_limit / 3600.0

    print()
    print(f"{'tier':<6}{'rows':>7}{'tasks':>7}{'runs':>8}{'worst core-h':>14}"
          f"{'worst h/task':>14}   array range")
    print("-" * 88)
    total_rows = total_runs = total_tasks = 0
    total_core_h = 0.0
    for tier in ("P", "A", "B", "C"):
        if tier not in tiers:
            continue
        acc = tiers[tier]
        first_task = (acc["first"] - 1) // chunk + 1
        last_task = (acc["last"] - 1) // chunk + 1
        tasks = last_task - first_task + 1
        print(f"{tier:<6}{acc['rows']:>7}{tasks:>7}{acc['runs']:>8}{acc['core_h']:>14,.0f}"
              f"{hours_per_task(acc):>14,.1f}   {first_task}-{last_task}  ({label[tier]})")
        total_rows += acc["rows"]
        total_tasks += tasks
        total_runs += acc["runs"]
        total_core_h += acc["core_h"]
    print("-" * 88)
    print(f"{'ALL':<6}{total_rows:>7}{total_tasks:>7}{total_runs:>8}{total_core_h:>14,.0f}")
    print(f"\nWorst case assumes every run uses the full {time_limit} s. The small family is "
          f"built so exact\nmethods CLOSE instances, so the real cost is well below this -- see "
          f"the design document\nfor the expected-case arithmetic.")
    if total_tasks > max_array:
        print(f"\n[note] {total_tasks} array tasks against --max-array-size={max_array}, so the "
              f"submission lines\n       below split each tier into WAVES that move ROW_OFFSET. "
              f"Check the real cap with\n       `scontrol show config | grep MaxArraySize`.")

    print("\nSubmit one tier at a time (nothing is submitted by this script):")
    for tier in ("P", "A", "B", "C"):
        if tier not in tiers:
            continue
        acc = tiers[tier]
        # Round the -t request up to the next whole hour, with an hour of slack for grounding,
        # process start-up and writing the result matrices.
        hours = max(1, int(hours_per_task(acc) + 0.5) + 1)
        tasks = (acc["last"] - acc["first"] + 1 + chunk - 1) // chunk
        print(f"  # tier {tier} -- {label[tier]}")
        # A tier longer than MaxArraySize is submitted as several WAVES rather than as fewer,
        # longer tasks: ROW_OFFSET moves the window instead of CHUNK making each job bigger,
        # which is the whole point of the per-run job shape.
        wave_start = acc["first"] - 1
        while wave_start < acc["last"]:
            wave_tasks = min(max_array, tasks)
            print(f"  sbatch -t {hours:02d}:00:00 --cpus-per-task={cpus[tier]} "
                  f"--export=ALL,CHUNK={chunk},ROW_OFFSET={wave_start} "
                  f"--array=1-{wave_tasks}%40 run_asp_ablation.slurm")
            wave_start += wave_tasks * chunk
            tasks -= wave_tasks


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
    parser.add_argument("--chunk", type=int, default=1,
                        help="Manifest rows per array task (default 1: one job per unit). Raise "
                             "it if the cluster's MaxArraySize is smaller than the row count; "
                             "the runner reads the same value from $CHUNK.")
    parser.add_argument("--max-array-size", type=int, default=1000,
                        help="Largest job array this cluster accepts (default 1000). A tier with "
                             "more tasks than this is submitted as several waves.")
    parser.add_argument("--per-run-systems", action="store_true",
                        help="Emit one row per SOLVER RUN instead of one per variant set. "
                             "Requires start_benchmark_caller.py to support --only-system, which "
                             "the per-run job restructuring is adding. This is what turns the "
                             "27-variant tier B from one 13.5 h job into 27 short ones.")
    parser.add_argument("--verify-system-keys", action="store_true",
                        help="Check the mirrored 27 system keys against build_system_config() "
                             "in start_benchmark_caller.py, then exit.")

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

    if args.verify_system_keys:
        return verify_system_keys()
    if args.chunk < 1:
        raise SystemExit("[ERROR] --chunk must be at least 1")

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
    print(f"wrote {args.out} ({len(rows)} rows, {args.chunk} row(s) per array task)")

    summarise(rows, args.time_limit, args.chunk, args.max_array_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
