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
Every tier runs single-threaded clasp search: the author wants the ablation to measure the
search configuration alone, so there is no thread tier.

    P  preflight   One instance, four profiles, the two tier-A systems, short limit. Proves the
                   flags reach clingo and that --solver-stats lands in the recorded JSON before
                   anything large is launched.
    A  main grid   4 profiles x every small instance, running the two exact configurations the
                   large family also runs (05_ASP_rp_dp_sp, 05_ASP_rp_d_sp). This tier produces
                   the recommendation.
    B  breadth     4 profiles x TEN exact-ASP variants (variant set "breadth10") on a small
                   stratified subsample. Answers "does the tier-A recommendation generalise across
                   the variant grid?" without paying for all 27:

                       rp_dp_sp                          the tier-A centre
                       rp_d_sp   rp_nd_sp                ground delay varied (d, nd)
                       rp_dp_ns  rp_dp_s                 sectorisation varied (ns, s)
                       nr_dp_sp  r_dp_sp                 rerouting varied (nr, r)
                       nr_nd_ns  r_d_s                   the two corners: nothing enabled, all full
                       r_d_ns                            fully loaded without sectorisation

                   One factor at a time around the centre (rp, dp, sp), plus the two corners and a
                   fully loaded setup without sectorisation, so every level of every axis appears.
                   It also carries the built-in control: the encoding's only live #heuristic
                   directive grounds to nothing unless dynamic sectorisation is PARTIAL, so
                   `domain` must behave exactly like `default` (and usc-domain like usc) on every
                   variant that is not _sp -- five of the ten here. If it does not, something in the
                   pipeline is wrong; analyze_asp_ablation.py checks it on whichever non-_sp cells
                   exist.

WHY tier B NEEDS ONE ROW PER SOLVER RUN
The caller's --experiment-* flags cannot express the ten: the per-variant flags
(--experiment-asp-r-d-s and friends) are parsed and then never read in build_system_config(), so
the exact-ASP systems are selectable only as "all 27" or as the two named singles. Tier B therefore
enables all 27 exactly as the "all27" variant set does and picks ONE of them per row with the
caller's --only-system. Without --per-run-systems a tier-B row could only mean "all 27", so this
script refuses to build tier B without it, and run_asp_ablation.slurm refuses a breadth10 row that
names no system.

Usage:
    ./build_ablation_manifest.py --per-run-systems        # reads ../05_instances/problems.tsv
    ./build_ablation_manifest.py --tiers P                # preflight rows only
    ./build_ablation_manifest.py --assume-grid --per-run-systems
                                                          # size the campaign before the data
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
#: ground-delay x rerouting x dynamic-sectorisation cross; "breadth10" is tier B's ten (see the
#: module docstring), which exists only as one row per run.
RUNS_PER_VARIANT_SET = {"named2": 2, "all27": 27, "breadth10": 10}

#: Tiers this script knows, in submission order.
ALL_TIERS = ("P", "A", "B")

#: Tier B's variants, as the <rerouting>_<delay>_<sectorisation> suffix of the system keys
#: all27_systems() builds. Order is the order of the rows.
BREADTH10_VARIANTS = (
    "rp_dp_sp",                  # centre: the tier-A configuration
    "rp_d_sp", "rp_nd_sp",       # ground delay varied
    "rp_dp_ns", "rp_dp_s",       # dynamic sectorisation varied
    "nr_dp_sp", "r_dp_sp",       # rerouting varied
    "nr_nd_ns", "r_d_s",         # the two corners
    "r_d_ns",                    # fully loaded, no sectorisation
)

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


def breadth10_systems() -> List[str]:
    """Tier B's ten system keys, taken from all27_systems() so the index is the caller's own."""
    by_variant = {key.split("_ASP_", 1)[1]: key for key in all27_systems()}
    missing = [v for v in BREADTH10_VARIANTS if v not in by_variant]
    if missing or len(set(BREADTH10_VARIANTS)) != 10:
        raise ValueError(f"BREADTH10_VARIANTS is not ten distinct all-27 variants: {missing}")
    return [by_variant[v] for v in BREADTH10_VARIANTS]


def systems_of(variants: str) -> List[str]:
    """The system keys one variant set stands for."""
    if variants == "named2":
        return list(NAMED2_SYSTEMS)
    if variants == "all27":
        return all27_systems()
    if variants == "breadth10":
        return breadth10_systems()
    raise ValueError(f"unknown variant set {variants!r}")


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
        # Tier B's ten are picked out of that same list with --only-system, so each has to be a
        # key the caller builds under the "all27" flags.
        breadth = breadth10_systems()
        absent = [key for key in breadth if key not in built]
        if absent:
            print(f"[MISMATCH] breadth10 keys the caller does not build: {absent}",
                  file=sys.stderr)
            return 1
        print(f"[ok] breadth10 (tier B) keys are all built by the caller: {', '.join(breadth)}")
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
            systems = systems_of(variants)
        elif variants == "breadth10":
            # Guarded in main() too. A breadth10 row without a system would read as "all 27".
            raise SystemExit("[ERROR] the breadth10 variant set needs --per-run-systems")
        for system in systems:
            runs = 1 if system != "-" else RUNS_PER_VARIANT_SET[variants]
            rows.append([
                str(len(rows) + 1), tier, profile, str(threads), variants,
                problem, instance, str(granularity), system, str(runs),
            ])

    def pad_to_chunk_boundary():
        """Start every tier on a fresh array task.

        Tiers are submitted separately, each with its own -t, so an array task straddling two
        tiers would run under the other tier's walltime request. With CHUNK=1 this never pads;
        with CHUNK>1 it inserts rows the runner skips.
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
                        emit("B", profile, 1, "breadth10", name, inst, granularity)

    return rows


def max_array_tasks(max_array_size: int) -> int:
    """The most tasks one `--array=1-N` may hold: MaxArraySize bounds the INDEX, so N <= size - 1.

    At MaxArraySize=50000 the cluster rejected --array=1-50000 as an invalid job array
    specification. Same rule as build_worklist.max_array_tasks().
    """
    return max(1, max_array_size - 1)


def plan_waves(first_row: int, tasks: int, chunk: int, max_array_size: int):
    """(ROW_OFFSET, n_tasks) per wave of one tier, each submitted as --array=1-<n_tasks>.

    first_row is the tier's first manifest row (1-based). Task t of a wave runs rows
    ROW_OFFSET + (t-1)*CHUNK + 1 .. ROW_OFFSET + t*CHUNK, as run_asp_ablation.slurm computes them.
    """
    per_wave = max_array_tasks(max_array_size)
    waves, done = [], 0
    while done < tasks:
        n = min(per_wave, tasks - done)
        waves.append((first_row - 1 + done * chunk, n))
        done += n
    return waves


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

    cpus = {"P": 2, "A": 2, "B": 2}
    label = {"P": "preflight", "A": "main grid", "B": "variant breadth"}

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
    for tier in ALL_TIERS:
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
    if any((acc["last"] - acc["first"] + chunk) // chunk > max_array_tasks(max_array)
           for acc in tiers.values()):
        print(f"\n[note] a tier has more array tasks than MaxArraySize={max_array} allows in one "
              f"array (at most\n       {max_array_tasks(max_array)}: the highest index is "
              f"MaxArraySize - 1), so the submission lines below\n       split it into WAVES "
              f"that move ROW_OFFSET. Check the real cap with\n"
              f"       `scontrol show config | grep MaxArraySize`.")

    print("\nSubmit one tier at a time (nothing is submitted by this script):")
    for tier in ALL_TIERS:
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
        for row_offset, wave_tasks in plan_waves(acc["first"], tasks, chunk, max_array):
            print(f"  sbatch -t {hours:02d}:00:00 --cpus-per-task={cpus[tier]} "
                  f"--export=ALL,CHUNK={chunk},ROW_OFFSET={row_offset} "
                  f"--array=1-{wave_tasks}%40 run_asp_ablation.slurm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance-root", type=Path, default=Path("../05_instances"),
                        help="Where the expansion script put the instances (default ../05_instances)")
    parser.add_argument("--out", type=Path, default=Path("ablation_tasks.tsv"),
                        help="Manifest to write (default ablation_tasks.tsv)")
    parser.add_argument("--tiers", type=str, default="P,A,B",
                        help="Comma-separated tiers to emit, in order. Default P,A,B. Tier B "
                             "needs --per-run-systems.")
    parser.add_argument("--time-limit", type=int, default=1800,
                        help="Per-run limit used ONLY for the cost estimate printed here "
                             "(default 1800, the value the papers use)")
    parser.add_argument("--chunk", type=int, default=1,
                        help="Manifest rows per array task (default 1: one job per unit). Raise "
                             "it if the cluster's MaxArraySize is smaller than the row count; "
                             "the runner reads the same value from $CHUNK.")
    parser.add_argument("--max-array-size", type=int, default=1001,
                        help="The cluster's MaxArraySize, as `scontrol show config` reports it "
                             "(default 1001, SLURM's own default). The highest array index SLURM "
                             "accepts is one less, so a tier with more tasks than that is "
                             "submitted as several waves.")
    parser.add_argument("--per-run-systems", action="store_true",
                        help="Emit one row per SOLVER RUN instead of one per variant set. "
                             "Requires start_benchmark_caller.py to support --only-system, which "
                             "the per-run job restructuring added. REQUIRED for tier B, whose "
                             "ten variants no --experiment-* flag can select.")
    parser.add_argument("--verify-system-keys", action="store_true",
                        help="Check the mirrored 27 system keys against build_system_config() "
                             "in start_benchmark_caller.py, then exit.")

    parser.add_argument("--tier-b-sizes", type=str, default="10,20,30,40",
                        help="Flight counts in the tier-B subsample (default 10,20,30,40)")
    parser.add_argument("--tier-b-seed", type=int, default=11904657,
                        help="Single seed for tier B (default 11904657, the generator's own)")
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
        if tier not in ALL_TIERS:
            raise SystemExit(f"[ERROR] unknown tier {tier!r}; expected one of "
                             f"{', '.join(ALL_TIERS)} (the 4-thread tier C was dropped: the "
                             f"ablation measures single-threaded search only)")
    if "B" in args.tiers and not args.per_run_systems:
        raise SystemExit("[ERROR] tier B runs ten of the 27 exact-ASP variants, which only "
                         "--only-system can select: pass --per-run-systems (or leave B out of "
                         "--tiers). Without it a tier-B row could only mean all 27.")
    args.tier_b_sizes = [int(x) for x in args.tier_b_sizes.split(",") if x.strip()]
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
