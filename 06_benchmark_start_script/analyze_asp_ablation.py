#!/usr/bin/env python3
"""Turn the ablation campaign's output into the recommendation it exists to produce.

    ./analyze_asp_ablation.py                          # newest output/*_ABLATION folder
    ./analyze_asp_ablation.py --folder 20260920_ABLATION
    ./analyze_asp_ablation.py --check                  # pipeline self-checks only

It answers, per profile: how many instances were CLOSED (optimum proven), how long that took
where it happened, and for the rest how far the reported incumbent is from the best bound
available. It writes ablation_runs.csv, ablation_profiles.csv and ablation_summary.md into the
campaign folder and prints the recommendation.

WHAT IT READS, AND WHY NOT THE OBVIOUS THING
Per run it reads `individual_outputs/<INSTANCE>_<SYSTEM>.json`, which holds every JSON line the
solver printed. The per-metric CSVs next to it are not used: `sol_value_to_rows()` in
start_benchmark_caller.py discovers its header while walking instances, so a metric that is
absent on the first instance and present on a later one yields rows of differing length. The
JSON has no such quirk, and it is the only place the --solver-stats fields survive.

COMPARING PROFILES AT A TIMEOUT IS NOT COMPARING NUMBERS
The objective is lexicographic, six levels deep:

    @10 overload   @9 arrival delay   @8 sector number   @7 sector diff   @6 reroute   @5 reconfig

Core-guided search (usc) drives the top levels down and can leave the lower ones at their initial
values; branch-and-bound improves all six together. Comparing two timed-out runs by any single
number therefore compares different things. Every comparison here is LEXICOGRAPHIC on the six-
level vector, and a gap is reported as the highest-priority level at which the incumbent differs
from the reference, with the size of that difference.

WHERE THE REFERENCE COMES FROM, in this order:

  1. Some profile CLOSED that (variant, instance): its cost vector is a PROVEN optimum, so every
     other profile's gap against it is exact. The small family exists so exact methods can close
     instances, so this is the common case and the one the recommendation rests on.
  2. Nobody closed it, but a run recorded SOLVER-LOWER-BOUND: the lexicographic maximum of those
     bounds. Note the limitation below -- today this case is rare.
  3. Nobody closed it and no bound was recorded: the best incumbent any profile reported, which
     is an UPPER bound. Distances measured against it are reported as "no bound" and are a
     ranking, not an optimality gap. They are never mixed into the gap statistics.

THE LIMITATION, STATED PLAINLY. SOLVER-LOWER-BOUND is read from ctl.statistics after
ctl.solve() returns, and the benchmark's time limit arrives as an external SIGKILL, so a
timed-out run prints no summary line and case 2 almost never fires. Until a run can be stopped
from INSIDE the process, gaps on instances that nobody closes are case 3. The report says which
case each number came from rather than blurring them together.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Objective levels, highest priority first, with the clingo weak-constraint priority they carry
#: in 02_ASP/encoding.lp. The order IS the lexicographic comparison.
COST_KEYS: Tuple[Tuple[str, int], ...] = (
    ("OVERLOAD", 10), ("ARRIVAL-DELAY", 9), ("SECTOR-NUMBER", 8),
    ("SECTOR-DIFF", 7), ("REROUTE", 6), ("RECONFIG", 5),
)

#: Outcome codes the caller writes into the last JSON line.
CODES = {"T": "timeout", "M": "memout", "E": "error", "P": "unparsable", "": "completed"}

#: <index>_ASP_<rerouting>_<grounddelay>_<sectorisation>. The index differs between the 27-way
#: expansion and the two named singles for the SAME configuration (18_ASP_rp_dp_sp is
#: 05_ASP_rp_dp_sp), so the index is dropped and the configuration is what identifies a variant.
SYSTEM_RE = re.compile(r"^\d+_ASP_(?P<variant>[a-z]+_[a-z]+_[a-z]+)$")

#: <TIER>_<PROFILE>_t<THREADS>, the directory run_asp_ablation.slurm creates per unit.
UNIT_RE = re.compile(r"^(?P<tier>[A-Z])_(?P<profile>[a-z-]+)_t(?P<threads>\d+)$")


class Run:
    """One (profile, threads, variant, instance) solver run."""

    __slots__ = ("tier", "profile", "threads", "variant", "problem", "instance",
                 "size", "seed", "outcome", "cost", "lower", "closed", "wall_s",
                 "models", "grounding_s", "solver_s")

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    @property
    def has_incumbent(self) -> bool:
        return self.cost is not None

    @property
    def key(self) -> Tuple[str, str, str]:
        """What a run is compared ACROSS profiles on."""
        return (self.variant, self.problem, self.instance)


def parse_instance_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        size, seed = name.split("_SEED", 1)
        return int(size), int(seed)
    except ValueError:
        return None, None


def cost_vector(record: Dict) -> Optional[List[int]]:
    """The six-level cost of a reported model, or None if no model was ever reported.

    SOLVER-COST (per model) and SOLVER-COSTS (from the summary) are clingo's own vectors and are
    preferred when present, because they are what clasp optimised. The six objective keys are the
    fallback and are always there on a line that carries a model.
    """
    for key in ("SOLVER-COSTS", "SOLVER-COST"):
        value = record.get(key)
        if isinstance(value, list) and len(value) == len(COST_KEYS):
            return [int(v) for v in value]
    if all(key in record for key, _ in COST_KEYS):
        return [int(record[key]) for key, _ in COST_KEYS]
    return None


def lower_bound(record: Dict) -> Optional[List[int]]:
    value = record.get("SOLVER-LOWER-BOUND")
    if isinstance(value, list) and len(value) == len(COST_KEYS):
        return [int(v) for v in value]
    return None


def read_wall_times(unit_dir: Path) -> Dict[Tuple[str, str], float]:
    """(instance, system) -> wall seconds, from execution_time.csv.

    This file is a plain matrix -- instances down, systems across -- and does not go through
    sol_value_to_rows(), so it is safe to read. A failure code ('T', 'M', 'E', 'P') sits where a
    number would be; those become None and the run's outcome carries the reason.
    """
    path = unit_dir / "execution_time.csv"
    times: Dict[Tuple[str, str], float] = {}
    if not path.exists():
        return times
    with path.open() as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return times
    header = rows[0][1:]
    for row in rows[1:]:
        if not row:
            continue
        for system, cell in zip(header, row[1:]):
            try:
                times[(row[0], system)] = float(cell)
            except (TypeError, ValueError):
                continue
    return times


def collect(folder: Path) -> List[Run]:
    """Every run in the campaign folder, one Run per individual_outputs JSON file."""
    runs: List[Run] = []
    for unit_dir in sorted(folder.glob("**/individual_outputs")):
        parent = unit_dir.parent
        # <folder>/<TIER>_<PROFILE>_t<N>/[<SYSTEM>/]<PROBLEM>/<INSTANCE>/individual_outputs
        parts = parent.relative_to(folder).parts
        unit = None
        for part in parts:
            match = UNIT_RE.match(part)
            if match:
                unit = match
                break
        if unit is None:
            print(f"[warn] cannot tell which unit {parent} belongs to; skipped", file=sys.stderr)
            continue
        problem, instance = parts[-2], parts[-1]
        wall = read_wall_times(parent)

        for json_path in sorted(unit_dir.glob("*.json")):
            system = json_path.stem[len(instance) + 1:] if json_path.stem.startswith(
                instance + "_") else json_path.stem
            variant_match = SYSTEM_RE.match(system)
            if not variant_match:
                continue                       # not an exact-ASP system; nothing to ablate
            try:
                lines = json.loads(json_path.read_text()).get("object", [])
            except (OSError, ValueError) as exc:
                print(f"[warn] unreadable {json_path}: {exc}", file=sys.stderr)
                continue
            if not lines:
                continue
            last = lines[-1] if isinstance(lines[-1], dict) else {}

            # The best model actually reported, not merely the last line: the last line of a
            # SIGKILLed run can be a bare {"ERROR": "T"} with no model in it.
            best_cost = None
            best_lower = None
            models = 0
            for line in lines:
                if not isinstance(line, dict):
                    continue
                candidate = cost_vector(line)
                if candidate is not None:
                    models += 1
                    if best_cost is None or candidate < best_cost:
                        best_cost = candidate
                candidate_lower = lower_bound(line)
                if candidate_lower is not None:
                    if best_lower is None or candidate_lower > best_lower:
                        best_lower = candidate_lower

            outcome = CODES.get(last.get("ERROR"), "unknown")
            # SOLVER-EXHAUSTED is clingo's own "the search space was closed". COMPUTATION-FINISHED
            # is fed from the same SolveResult.exhausted since 5ad2d85, and is the fallback for a
            # run recorded without --solver-stats.
            closed = bool(last.get("SOLVER-EXHAUSTED",
                                   last.get("COMPUTATION-FINISHED", False)))
            if outcome != "completed":
                closed = False                  # a killed run proved nothing, whatever it printed

            size, seed = parse_instance_name(instance)
            runs.append(Run(
                tier=unit.group("tier"), profile=unit.group("profile"),
                threads=int(unit.group("threads")),
                variant=variant_match.group("variant"),
                problem=problem, instance=instance, size=size, seed=seed,
                outcome=outcome, cost=best_cost, lower=best_lower, closed=closed,
                wall_s=wall.get((instance, system)),
                models=models,
                grounding_s=last.get("GROUNDING-TIME"),
                solver_s=last.get("TOTAL-TIME-TO-THIS-POINT"),
            ))
    return runs


def references(runs: Sequence[Run]) -> Dict[Tuple[str, str, str], Dict]:
    """A reference cost per (variant, problem, instance), with where it came from.

    Also flags the one thing that would be a genuine defect rather than noise: two profiles
    PROVING different optima for the same instance. Both are exact methods on the same program,
    so that cannot happen unless something is wrong.
    """
    refs: Dict[Tuple[str, str, str], Dict] = {}
    proven: Dict[Tuple[str, str, str], Dict[str, List[int]]] = defaultdict(dict)

    for run in runs:
        if run.closed and run.cost is not None:
            # Keyed by TIER too: the same cell is proven in tier A as 05_ASP_rp_dp_sp and in
            # tier B as 18_ASP_rp_dp_sp, and those are the cross-tier consistency check. Keying
            # on the profile alone would let one silently overwrite the other and hide exactly
            # the disagreement this is looking for.
            proven[run.key][f"{run.tier}/{run.profile}/t{run.threads}"] = run.cost

    for key, by_profile in proven.items():
        distinct = {tuple(cost) for cost in by_profile.values()}
        refs[key] = {
            "cost": list(min(distinct)),
            "source": "proven optimum",
            "conflict": sorted(by_profile.items()) if len(distinct) > 1 else None,
        }

    for run in runs:
        if run.key in refs:
            continue
        entry = refs.setdefault(run.key, {"cost": None, "source": "none", "conflict": None})
        if run.lower is not None and (entry["source"] != "lower bound"
                                      or run.lower > entry["cost"]):
            entry["cost"], entry["source"] = run.lower, "lower bound"
        elif entry["source"] == "none" and run.cost is not None:
            entry["cost"], entry["source"] = run.cost, "best incumbent (no bound)"
        elif entry["source"] == "best incumbent (no bound)" and run.cost is not None \
                and run.cost < entry["cost"]:
            entry["cost"] = run.cost
    return refs


def gap(cost: Optional[Sequence[int]], reference: Optional[Sequence[int]]):
    """(level, absolute, relative) at the highest-priority level where cost exceeds reference.

    None where there is nothing to compare, and (None, 0, 0.0) where the two agree on every
    level -- which for a proven reference means the run found the optimum without proving it.
    """
    if cost is None or reference is None:
        return None
    for (name, priority), value, bound in zip(COST_KEYS, cost, reference):
        if value != bound:
            absolute = value - bound
            relative = absolute / abs(bound) if bound else float("inf") if absolute else 0.0
            return (f"@{priority} {name}", absolute, relative)
    return (None, 0, 0.0)


def par2(runs: Sequence[Run], limit: float) -> Optional[float]:
    """Penalised average runtime: an unclosed run counts as twice the limit.

    The standard way to score a solver over a mixed set of closed and unclosed instances, and the
    only one here that does not silently average over different instance sets.
    """
    values = [run.wall_s if (run.closed and run.wall_s is not None) else 2 * limit
              for run in runs]
    return statistics.mean(values) if values else None


def summarise(runs: Sequence[Run], refs, limit: float, tier: Optional[str] = None):
    """Per-(profile, threads) aggregates, plus the paired comparison against `default`."""
    selected = [r for r in runs if tier is None or r.tier == tier]
    arms = sorted({(r.profile, r.threads) for r in selected})

    # Instances every arm attempted, so "closed" counts are over the same set.
    attempted = defaultdict(set)
    for run in selected:
        attempted[(run.profile, run.threads)].add(run.key)
    common = set.intersection(*attempted.values()) if attempted else set()

    closed_by_arm = {}
    for arm in arms:
        closed_by_arm[arm] = {r.key for r in selected
                              if (r.profile, r.threads) == arm and r.closed and r.key in common}
    closed_by_all = set.intersection(*closed_by_arm.values()) if closed_by_arm else set()

    # Best incumbent per cell, once, rather than rescanning the pool for every run.
    best_cost_by_key: Dict[Tuple[str, str, str], List[int]] = {}
    for run in selected:
        if run.cost is None:
            continue
        current = best_cost_by_key.get(run.key)
        if current is None or run.cost < current:
            best_cost_by_key[run.key] = run.cost

    rows = []
    for arm in arms:
        mine = [r for r in selected if (r.profile, r.threads) == arm and r.key in common]
        closed = [r for r in mine if r.closed]
        everywhere = [r for r in mine if r.key in closed_by_all and r.wall_s is not None]
        gaps, from_zero = [], 0
        for run in mine:
            if run.closed:
                continue
            ref = refs.get(run.key, {})
            if ref.get("source") not in ("proven optimum", "lower bound"):
                continue                       # case 3 is a ranking, never a gap
            measured = gap(run.cost, ref.get("cost"))
            if measured is None or measured[0] is None:
                continue
            if measured[2] == float("inf"):
                # The reference is 0 at that level and the incumbent is not, so there is no
                # relative gap to quote -- the run simply did not reach a level the optimum
                # reaches. That is the single most telling failure mode here (bb sitting at
                # overload 39 where usc reaches 0), so it is counted rather than averaged away.
                from_zero += 1
            else:
                gaps.append(measured[2])
        rows.append({
            "profile": arm[0],
            "threads": arm[1],
            "runs": len(mine),
            "closed": len(closed),
            "closed_pct": 100.0 * len(closed) / len(mine) if mine else 0.0,
            "no_incumbent": sum(1 for r in mine if not r.has_incumbent),
            "median_time_closed_s": statistics.median(
                [r.wall_s for r in closed if r.wall_s is not None]) if closed else None,
            "median_time_common_s": statistics.median(
                [r.wall_s for r in everywhere]) if everywhere else None,
            "par2_s": par2(mine, limit),
            "n_gaps": len(gaps),
            "median_gap": statistics.median(gaps) if gaps else None,
            "gap_from_zero": from_zero,
            "lex_best": sum(1 for r in mine if best_cost_by_key.get(r.key) == r.cost
                            and r.cost is not None),
        })
    return rows, common, closed_by_all


def recommend(rows) -> Tuple[str, List[str]]:
    """The pre-registered decision rule, applied to the tier-A aggregates.

    Fixed in advance so the recommendation is a reading of the data and not a search through it:
      1. most instances CLOSED;
      2. ties broken by PAR2 wall clock;
      3. ties broken by how often the arm's incumbent is lexicographically best.
    A separate WARNING is raised, not a demotion, when the winner returns no incumbent at all on
    runs it does not close: a configuration that answers "nothing" on a timeout is a poor default
    for anyone who needs an answer within a budget, however many instances it proves.
    """
    if not rows:
        return "no data", []
    ranked = sorted(rows, key=lambda r: (-r["closed"],
                                         r["par2_s"] if r["par2_s"] is not None else float("inf"),
                                         -r["lex_best"]))
    winner = ranked[0]
    notes = []
    name = winner["profile"] + (f" at {winner['threads']} threads" if winner["threads"] > 1 else "")
    text = (f"{name}: closed {winner['closed']}/{winner['runs']} "
            f"({winner['closed_pct']:.0f}%), PAR2 {winner['par2_s']:.0f}s")
    if winner["no_incumbent"]:
        notes.append(
            f"{winner['profile']} returned NO incumbent at all on {winner['no_incumbent']} run(s). "
            f"Anyone who needs an answer within the budget, rather than a proof, should read the "
            f"per-profile no_incumbent column before adopting it.")
    baseline = next((r for r in rows if r["profile"] == "default" and r["threads"] == 1), None)
    if baseline and winner is not baseline:
        notes.append(
            f"default closed {baseline['closed']}/{baseline['runs']} with PAR2 "
            f"{baseline['par2_s']:.0f}s, so the recommendation is a change from the shipped "
            f"default, which stays `default` unless the author decides otherwise.")
    return text, notes


def self_checks(runs: Sequence[Run], refs) -> List[str]:
    """Things that must hold if the pipeline did what it was told. Violations are defects.

    1. No two arms may PROVE different optima for the same cell.
    2. `domain` must behave exactly like `default` on every variant whose sectorisation is not
       PARTIAL. 02_ASP/encoding.lp derives initial_sector/1 only under
       regulation_restricted_dynamic_sector_allocation, and the encoding's one live #heuristic
       directive has initial_sector(SEC) in its body, so on _ns and _s variants the directive
       grounds to nothing and --heuristic=Domain has nothing to honour. The same holds for
       usc-domain against usc. A difference in the CLOSED set there means the two arms did not
       run what this script thinks they ran.
    """
    problems = []
    for key, ref in refs.items():
        if ref.get("conflict"):
            problems.append(
                f"CRITICAL: {key} has two proven optima: {ref['conflict']}. Two exact runs on "
                f"the same program cannot both be right.")

    by_arm = defaultdict(dict)
    for run in runs:
        if run.threads == 1:
            by_arm[run.profile][run.key] = run
    for plain, with_domain in (("default", "domain"), ("usc", "usc-domain")):
        shared = set(by_arm.get(plain, {})) & set(by_arm.get(with_domain, {}))
        inert = [k for k in shared if not k[0].endswith("_sp")]
        differing = [k for k in inert
                     if by_arm[plain][k].closed != by_arm[with_domain][k].closed]
        if differing:
            problems.append(
                f"{with_domain} differs from {plain} on {len(differing)} non-_sp cell(s), where "
                f"the #heuristic directive cannot ground. Check that the profile reached clingo "
                f"and that the right variant ran. Examples: {differing[:3]}")
        elif inert:
            problems.append(
                f"[ok] {with_domain} matches {plain} on all {len(inert)} non-_sp cells, as the "
                f"encoding requires.")
    return problems


def write_outputs(folder: Path, runs, refs, rows, limit) -> None:
    with (folder / "ablation_runs.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tier", "profile", "threads", "variant", "problem", "instance",
                         "flights", "seed", "outcome", "closed", "wall_s", "models",
                         "cost", "lower_bound", "reference", "reference_source",
                         "gap_level", "gap_absolute", "gap_relative"])
        for run in runs:
            ref = refs.get(run.key, {})
            measured = gap(run.cost, ref.get("cost")) if not run.closed else None
            writer.writerow([
                run.tier, run.profile, run.threads, run.variant, run.problem, run.instance,
                run.size, run.seed, run.outcome, int(bool(run.closed)),
                f"{run.wall_s:.3f}" if run.wall_s is not None else "",
                run.models,
                " ".join(map(str, run.cost)) if run.cost else "",
                " ".join(map(str, run.lower)) if run.lower else "",
                " ".join(map(str, ref.get("cost"))) if ref.get("cost") else "",
                ref.get("source", ""),
                measured[0] if measured and measured[0] else "",
                measured[1] if measured else "",
                f"{measured[2]:.4f}" if measured and measured[2] != float("inf") else "",
            ])

    with (folder / "ablation_profiles.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["profile"])
        writer.writeheader()
        writer.writerows(rows)

    text, notes = recommend(rows)
    lines = [
        "# ASP solver-option ablation -- results", "",
        f"Per-run limit {limit:.0f}s. Closed = clingo's own SOLVER-EXHAUSTED, i.e. the optimum "
        f"was PROVEN,\nnot merely reached. Gaps are lexicographic over the six objective levels "
        f"and are only\ncomputed against a proven optimum or a recorded lower bound.", "",
        "| profile | threads | runs | closed | closed % | no incumbent | median s (closed) | "
        "median s (closed by all) | PAR2 s | gaps | median gap | missed level | lex-best |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        def fmt(value, spec=".1f"):
            return "-" if value is None else format(value, spec)
        lines.append(
            f"| {row['profile']} | {row['threads']} | {row['runs']} | {row['closed']} | "
            f"{row['closed_pct']:.0f}% | {row['no_incumbent']} | {fmt(row['median_time_closed_s'])} | "
            f"{fmt(row['median_time_common_s'])} | {fmt(row['par2_s'], '.0f')} | {row['n_gaps']} | "
            f"{fmt(row['median_gap'], '.3f')} | {row['gap_from_zero']} | {row['lex_best']} |")
    lines += ["", "## Recommendation", "", text, ""]
    lines += [f"- {note}" for note in notes]
    (folder / "ablation_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--folder", type=str, default=None,
                        help="Campaign folder under --output-root; default is the newest "
                             "*_ABLATION")
    parser.add_argument("--tier", type=str, default="A",
                        help="Tier the recommendation is read from (default A, the main grid). "
                             "Pass 'all' to aggregate every tier together.")
    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="Per-run limit the campaign used; PAR2 needs it")
    parser.add_argument("--check", action="store_true",
                        help="Run the pipeline self-checks and exit")
    args = parser.parse_args()

    root = args.output_root
    if args.folder:
        folder = root / args.folder
    else:
        candidates = sorted(p for p in root.glob("*_ABLATION") if p.is_dir())
        if not candidates:
            raise SystemExit(f"[ERROR] no *_ABLATION folder under {root}; pass --folder")
        folder = candidates[-1]
    print(f"[folder] {folder}")

    runs = collect(folder)
    if not runs:
        raise SystemExit(f"[ERROR] no exact-ASP runs found under {folder}")
    print(f"[runs]   {len(runs)} solver runs, "
          f"{len({r.key for r in runs})} (variant, instance) cells, "
          f"{len({(r.profile, r.threads) for r in runs})} arms")

    refs = references(runs)
    checks = self_checks(runs, refs)
    for line in checks:
        print(line)
    if args.check:
        return 1 if any(not c.startswith("[ok]") for c in checks) else 0

    tier = None if args.tier == "all" else args.tier
    rows, common, closed_by_all = summarise(runs, refs, args.time_limit, tier)
    if not rows:
        raise SystemExit(f"[ERROR] no runs in tier {args.tier}")
    print(f"[tier {args.tier}] {len(common)} cells attempted by every arm, "
          f"{len(closed_by_all)} closed by all of them")

    write_outputs(folder, runs, refs, rows, args.time_limit)
    text, notes = recommend(rows)
    print("\nRECOMMENDATION: " + text)
    for note in notes:
        print("  note: " + note)
    print(f"\nwrote {folder}/ablation_runs.csv, ablation_profiles.csv, ablation_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
