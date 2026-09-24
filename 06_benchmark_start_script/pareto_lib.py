"""Shared code of the exact (delay, sectors) Pareto-front campaign.

Used by build_pareto_manifest.py (planner), run_pareto_steps.slurm (via the manifest it writes),
merge_pareto_steps.py (merge + certification) and plot_pareto_fronts.py (figure). The design is
pareto_research/CAMPAIGN_DRAFT.md in the JOAS paper folder; this file holds the parts every one of
those scripts must agree on:

  * the campaign: ten variants in three groups, the flight counts per group, the memory classes;
  * the label of a front, <region>_<flights>_<seed>__<variant>[__<metric>] (the metric suffix only
    when it is not the campaign's `signed`);
  * where a step's result lives: output/<FOLDER>/steps/<label>/<round>_<step>.jsonl, with <step>
    DF (delay-first end), SF (sectors-first end) or K<k> (sectors minimised under delay <= k);
  * reading a step file, and the certification of Section 3.4 of the draft.

CERTIFICATION
The quantity certified is f(K) = the least number of sectors at the least overload o* of the
unbounded program, subject to total arrival delay <= K. f is non-increasing in K. Two staircases
bound it:

  U(K)  fewest sectors among all feasible points at overload o* with delay <= K, found in ANY step
        (every model a step reported, proven or not; a model of a bounded step is feasible for the
        unbounded program too).
  L(K)  the largest lower bound that some certificate gives at K. A certificate (Kc, v) says
        f(K) >= v for every K <= Kc; v may be infinite (no solution at overload o*). They come from
        clingo's lexicographic lower bound, which every step records at its end:
          - sectors-first step with bound Kc (Kc = +inf when unbounded) and lower bound (lo, ls, ld):
              lo > o*  ->  (Kc, inf)          no solution with delay <= Kc reaches o*
              lo = o*  ->  (Kc, ls)
            and when the step is EXHAUSTED with optimum (o*, s, d): also (d - 1, s + 1), since the
            solver minimised delay among the least-sector solutions under the bound;
          - a sectors-first step that is exhausted without any model (the bound is unsatisfiable):
              (Kc, inf);
          - delay-first (unbounded) step with lower bound (lo, ld, ls), lo = o*:
              (ld - 1, inf)                   no o*-solution has delay below ld
            and when it is EXHAUSTED with optimum (o*, D_min, S): also (D_min, S);
          - the `floored` metric: (-1, inf), since no delay is negative.
        The extra certificates of an exhausted step are exactly what the old plot_fronts.py rules
        used; for a step stopped at its deadline only the bound on its own top open level is used.
f(K) is known wherever L(K) = U(K).

Front points are the corners of U. Between two neighbouring corners (d0, s0), (d1, s1) the stretch
is CERTIFIED when L(d1 - 1) >= s0 and L(d1) >= s1. The LEFT end is certified when
L(d_first - 1) is infinite and L(d_first) >= s_first; the RIGHT end when L(+inf) >= s_last and
L(d_last - 1) > s_last -- only an unbounded sectors-first step gives L(+inf), which keeps the rule of
plot_fronts.py that a bounded step repeating the last point says nothing about larger delays.

Classes (draft Section 2.3): exact = both ends and every stretch certified; partial = at least one
certified stretch; endpoints = the left (delay-first) end certified, no stretch; none = the rest.
o* itself must be known: the largest overload lower bound of an unbounded step equals the least
overload of any point found. Otherwise the front is `none`.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INF = math.inf

# ---------------------------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------------------------

#: The arrival-delay metric the V2 campaign and the papers use (start_benchmark_caller.py passes
#: common/arrival_delay.py's default, `signed`, on every command since c09f48d).
CAMPAIGN_METRIC = "signed"
#: The clasp seed start_benchmark_caller.py passes to every 02_ASP run (build_command's default),
#: so a delay-first end here is the same run as the usc27 one.
SOLVER_SEED = 11904657
#: 02_ASP's default --max-time; the campaign solves at this horizon only (--fixed-horizon).
HORIZON = 24
#: Every step, every round (Alexander, 2026-09-24).
STEP_TIME_LIMIT = 1800
#: The known-hard steps of the pilot's P2 part are run once at this limit (draft Section 3.3).
P2_TIME_LIMIT = 3600

REROUTING = {"nr": 0, "rp": 1, "r": 2}
GROUND_DELAY = {"nd": 0, "dp": 1, "d": 2}
SECTORISATION = {"ns": 0, "sp": 1, "s": 2}

#: Table 1 of the draft. Group -> variants; flight counts with a full front and with ends only.
GROUPS = {
    "A": {"variants": ("nr_dp_sp", "nr_d_sp", "rp_nd_sp", "rp_dp_sp", "rp_d_sp"),
          "full": (20, 30, 40, 60), "ends": (80, 100), "mem": "8G"},
    "B": {"variants": ("r_d_sp", "r_dp_sp", "r_nd_sp"),
          "full": (20, 30, 40), "ends": (), "mem": "40G"},
    "C": {"variants": ("rp_dp_s", "rp_d_s"),
          "full": (20, 30), "ends": (), "mem": "8G"},
}
#: Group B's check pair runs at 20 flights only (draft Section 2.2).
FULL_SIZES_OVERRIDE = {"r_dp_sp": (20,), "r_nd_sp": (20,)}
VARIANTS = tuple(v for g in GROUPS.values() for v in g["variants"])
#: Memory classes: --mem of the SLURM job, and the address-space cap the runner puts on 02_ASP.
MEM_CLASSES = {"8G": {"sbatch_mem": "8G", "as_gib": 7}, "40G": {"sbatch_mem": "40G", "as_gib": 35}}

#: Round G: at most this many K values strictly inside (D_min, D_S) per front.
GRID_CAP = 24
#: Round G is skipped for a (group, flights) cell whose round-E pass rate is below this.
PASS_RATE_MIN = 0.10
#: Round F: at most this many K values per front per round, and this many core-hours in total.
FILL_CAP = 12
FILL_BUDGET_CORE_H = 1000.0

#: The pilot (draft Section 3.3). Problem directories under the pilot instance root, as the
#: submission sketch rsyncs them.
PILOT_V1_EU = "V1-MAJOR-EUROPE-10x10"
PILOT_V1_CE = "V1-CENTRAL-EUROPE-5x5"   # the BSc student's instance (V1 generator)
PILOT_FRONTS = (
    # (problem, instance, variant, metric)
    *[(PILOT_V1_EU, f"0000020_SEED{seed}", variant, metric)
      for seed in (150699, 11904657) for variant in ("rp_d_sp", "rp_dp_sp")
      for metric in ("floored", "signed")],
    *[(PILOT_V1_CE, "0000020_SEED42", variant, metric)
      for variant in ("rp_d_sp", "r_d_sp") for metric in ("floored", "signed")],
)
#: P2: the known-hard middle steps of the laptop sweeps (eu42dp, eu13dp), once each at 3600 s.
PILOT_P2 = (
    *[(PILOT_V1_EU, "0000020_SEED42", "rp_dp_sp", "floored", k) for k in (4, 8, 12, 16)],
    *[(PILOT_V1_EU, "0000020_SEED13", "rp_dp_sp", "floored", k) for k in (6, 12, 14)],
)
#: The laptop fronts the pilot must reproduce under `floored` (all points proven there).
PILOT_EXPECTED = {
    f"{PILOT_V1_EU}_20_150699__rp_d_sp__floored":
        [(3, 743), (4, 734), (5, 733), (6, 732), (7, 731), (9, 730), (11, 729)],
    f"{PILOT_V1_EU}_20_150699__rp_dp_sp__floored":
        [(3, 743), (4, 734), (5, 733), (6, 732), (7, 731), (9, 730), (11, 729)],
    f"{PILOT_V1_EU}_20_11904657__rp_d_sp__floored": [(0, 741), (1, 731), (2, 730), (3, 729)],
    f"{PILOT_V1_EU}_20_11904657__rp_dp_sp__floored": [(0, 741), (1, 731), (2, 730), (3, 729)],
    f"{PILOT_V1_CE}_20_42__rp_d_sp__floored":
        [(1, 307), (2, 304), (3, 303), (6, 302), (9, 301), (12, 300)],
    f"{PILOT_V1_CE}_20_42__r_d_sp__floored":
        [(1, 307), (2, 304), (3, 303), (5, 302), (6, 301), (8, 300)],
}


def group_of(variant: str) -> str:
    for name, group in GROUPS.items():
        if variant in group["variants"]:
            return name
    raise ValueError(f"variant {variant!r} is not one of the campaign's ten")


def mem_class_of(variant: str) -> str:
    return GROUPS[group_of(variant)]["mem"]


def regulation_flags(variant: str) -> Tuple[int, int, int]:
    """(ground delay, rerouting, dynamic sectorisation) for main.py, from <r>_<d>_<s>."""
    r, d, s = variant.split("_")
    return GROUND_DELAY[d], REROUTING[r], SECTORISATION[s]


def full_sizes(variant: str) -> Tuple[int, ...]:
    return FULL_SIZES_OVERRIDE.get(variant, GROUPS[group_of(variant)]["full"])


def end_sizes(variant: str) -> Tuple[int, ...]:
    return GROUPS[group_of(variant)]["ends"]


# ---------------------------------------------------------------------------------------------
# Labels and paths
# ---------------------------------------------------------------------------------------------

_V2_PROBLEM = re.compile(r"^\d+-\d+-(?P<region>.+?)(?:-V2)?$")


def region_of(problem: str) -> str:
    """30-1-CENTRAL-EUROPE-5x5-V2 -> CENTRAL-EUROPE-5x5; a pilot directory name is kept as is."""
    match = _V2_PROBLEM.match(problem)
    return match.group("region") if match else problem


def parse_instance(name: str) -> Tuple[int, int]:
    size, seed = name.split("_SEED", 1)
    return int(size), int(seed)


def make_label(problem: str, instance: str, variant: str, metric: str = CAMPAIGN_METRIC) -> str:
    flights, seed = parse_instance(instance)
    label = f"{region_of(problem)}_{flights}_{seed}__{variant}"
    if metric != CAMPAIGN_METRIC:
        label += f"__{metric}"
    return label


_LABEL = re.compile(r"^(?P<region>.+)_(?P<flights>\d+)_(?P<seed>\d+)__(?P<variant>[a-z]+_[a-z]+_[a-z]+)"
                    r"(?:__(?P<metric>signed|floored|absolute))?$")


def parse_label(label: str) -> Dict[str, object]:
    match = _LABEL.match(label)
    if not match:
        raise ValueError(f"not a front label: {label!r}")
    return {"region": match.group("region"), "flights": int(match.group("flights")),
            "seed": int(match.group("seed")), "variant": match.group("variant"),
            "metric": match.group("metric") or CAMPAIGN_METRIC}


def step_key(order: str, bound: Optional[int]) -> str:
    if bound is not None:
        return f"K{int(bound)}"
    return "DF" if order == "delay-first" else "SF"


def steps_dir(output_root: Path, folder: str) -> Path:
    return Path(output_root) / folder / "steps"


def step_path(output_root: Path, folder: str, label: str, round_name: str, key: str) -> Path:
    return steps_dir(output_root, folder) / label / f"{round_name}_{key}.jsonl"


# ---------------------------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------------------------

#: Columns of a Pareto manifest. run_pareto_steps.slurm reads them positionally.
MANIFEST_COLUMNS = ("row_id", "round", "mem_class", "label", "problem", "instance", "variant",
                    "metric", "time_granularity", "order", "bound", "step", "time_limit")


@dataclass
class Row:
    round: str
    mem_class: str
    label: str
    problem: str
    instance: str
    variant: str
    metric: str
    time_granularity: str
    order: str
    bound: Optional[int]
    time_limit: int

    @property
    def step(self) -> str:
        return step_key(self.order, self.bound)

    def cells(self, row_id: int) -> List[str]:
        return [str(row_id), self.round, self.mem_class, self.label, self.problem, self.instance,
                self.variant, self.metric, str(self.time_granularity), self.order,
                "none" if self.bound is None else str(self.bound), self.step,
                str(self.time_limit)]


def write_manifest(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for i, row in enumerate(rows, start=1):
            fh.write("\t".join(row.cells(i)) + "\n")
    tmp.replace(path)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def ranges(numbers: Iterable[int]) -> str:
    """1,2,3,7,9,10 -> '1-3,7,9-10', the form sbatch --array takes."""
    out, start, prev = [], None, None
    for n in sorted(numbers):
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


# ---------------------------------------------------------------------------------------------
# Step files
# ---------------------------------------------------------------------------------------------

#: The six objective levels, in the SECTORS-FIRST order of the five-column front CSV:
#: overload, sectors, delay, sector_diff, reroute, reconfig.
LEVELS = ("o", "s", "d", "sd", "rr", "rc")
#: Encoding priority of each level, per objective order.
PRIORITY_OF = {
    "sectors-first": {"o": 10, "s": 9, "d": 8, "sd": 7, "rr": 6, "rc": 5},
    "delay-first": {"o": 10, "d": 9, "s": 8, "sd": 7, "rr": 6, "rc": 5},
}
DEFAULT_PRIORITIES = [10, 9, 8, 7, 6, 5]


def place(vector, priorities, order: str) -> Optional[Dict[str, int]]:
    """A clingo cost/bound vector (one entry per priority PRESENT, highest first) on the six levels.

    A level absent from the ground program costs 0 in every model. Returns None for anything that
    is not a list of finite numbers of a length the priorities explain.
    """
    if not isinstance(vector, list) or not vector:
        return None
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    if not (isinstance(priorities, list) and len(priorities) == len(values)):
        if len(values) != 6:
            return None
        priorities = DEFAULT_PRIORITIES
    by_priority = dict(zip(priorities, values))
    return {level: int(by_priority.get(prio, 0)) for level, prio in PRIORITY_OF[order].items()}


@dataclass
class Step:
    """One solver run of a front: an end (DF, SF) or a bounded step (K<k)."""
    label: str
    round: str
    key: str
    order: str
    bound: Optional[int]
    path: Path
    final: Optional[Dict] = None               # the result line (the one --solve-deadline prints)
    models: List[Dict[str, int]] = field(default_factory=list)    # every reported model's costs
    model_times: List[float] = field(default_factory=list)
    incumbent: Optional[Dict[str, int]] = None
    lower: Optional[Dict[str, int]] = None
    exhausted: bool = False
    stopped: Optional[bool] = None
    prov: Dict[str, str] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def unbounded(self) -> bool:
        return self.bound is None

    @property
    def has_model(self) -> bool:
        return self.incumbent is not None

    @property
    def status(self) -> str:
        """The clingo-style status the five-column front CSV carries."""
        if self.final is None:
            return "KILLED"            # no result line: killed at the limit, out of memory, crashed
        if self.exhausted:
            return "OPTIMUM FOUND" if self.has_model else "UNSATISFIABLE"
        return "SATISFIABLE" if self.has_model else "UNKNOWN"

    @property
    def wall_s(self) -> Optional[float]:
        try:
            return float(self.prov.get("wall_s"))
        except (TypeError, ValueError):
            return None


def _read_prov(path: Path) -> Dict[str, str]:
    prov: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                prov[key.strip()] = value.strip()
    return prov


_STEP_FILE = re.compile(r"^(?P<round>[A-Z][A-Z0-9]*)_(?P<key>DF|SF|K-?\d+)\.jsonl$")


def load_step(path: Path, priorities_hint: Optional[Dict[str, List[int]]] = None) -> Optional[Step]:
    """Read one step file. None if the name is not a step file."""
    path = Path(path)
    match = _STEP_FILE.match(path.name)
    if not match:
        return None
    key = match.group("key")
    order = "delay-first" if key == "DF" else "sectors-first"
    bound = int(key[1:]) if key.startswith("K") else None
    step = Step(label=path.parent.name, round=match.group("round"), key=key, order=order,
                bound=bound, path=path, prov=_read_prov(path.with_suffix(".prov")))

    lines = []
    for raw in path.read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            lines.append(json.loads(raw))
        except ValueError:
            continue
    finals = [line for line in lines if "SOLVER-STOPPED-AT-DEADLINE" in line]
    step.final = finals[-1] if finals else None
    priorities = (step.final or {}).get("SOLVER-COST-PRIORITIES")
    if not isinstance(priorities, list) and priorities_hint:
        priorities = priorities_hint.get(order)

    # Which options were in force. An imported delay-first end (usc27) predates the PARETO-* keys.
    if step.final is not None and "PARETO-OBJECTIVE-ORDER" in step.final:
        if step.final["PARETO-OBJECTIVE-ORDER"] != order:
            step.problems.append(f"line says order {step.final['PARETO-OBJECTIVE-ORDER']}, "
                                 f"file says {order}")
        if step.final.get("PARETO-DELAY-BOUND") != bound:
            step.problems.append(f"line says bound {step.final.get('PARETO-DELAY-BOUND')}, "
                                 f"file says {bound}")
    if step.final is not None and step.final.get("SOLVER-HORIZON-FINAL") is False:
        step.problems.append("SOLVER-HORIZON-FINAL is false: not a fixed-horizon result")

    for line in lines:
        if "OVERLOAD" not in line:
            continue                                   # the no-model statistics line
        cost = place(line.get("SOLVER-COST"), priorities, order)
        if cost is None and line is step.final:
            cost = place(line.get("SOLVER-COSTS"), priorities, order)
        if cost is not None:
            step.models.append(cost)
            step.model_times.append(float(line.get("TOTAL-TIME-TO-THIS-POINT") or 0.0))

    if step.final is not None:
        step.exhausted = bool(step.final.get("COMPUTATION-FINISHED", False))
        step.stopped = step.final.get("SOLVER-STOPPED-AT-DEADLINE")
        if "OVERLOAD" in step.final:
            step.incumbent = (place(step.final.get("SOLVER-COSTS"), priorities, order)
                              or place(step.final.get("SOLVER-COST"), priorities, order))
        step.lower = place(step.final.get("SOLVER-LOWER-BOUND"), priorities, order)
        if step.exhausted and step.incumbent is not None:
            step.lower = dict(step.incumbent)          # a proven optimum is its own bound
    elif step.models:
        step.incumbent = min(step.models, key=lambda c: [c[level] for level in
                                                          _order_levels(order)])
    return step


def _order_levels(order: str) -> List[str]:
    return sorted(LEVELS, key=lambda level: -PRIORITY_OF[order][level])


def load_front_steps(front_dir: Path) -> List[Step]:
    """Every step file of one front, the priorities of a killed step taken from its siblings."""
    paths = sorted(Path(front_dir).glob("*.jsonl"))
    hint: Dict[str, List[int]] = {}
    for path in paths:
        step = load_step(path)
        if step and step.final and isinstance(step.final.get("SOLVER-COST-PRIORITIES"), list):
            hint.setdefault(step.order, step.final["SOLVER-COST-PRIORITIES"])
    steps = [load_step(path, hint) for path in paths]
    return [s for s in steps if s is not None]


def load_all(output_root: Path, folder: str) -> Dict[str, List[Step]]:
    root = steps_dir(output_root, folder)
    fronts: Dict[str, List[Step]] = {}
    if not root.is_dir():
        return fronts
    for front_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        fronts[front_dir.name] = load_front_steps(front_dir)
    return fronts


# ---------------------------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------------------------

@dataclass
class Front:
    label: str
    metric: str
    o_star: Optional[int] = None
    o_known: bool = False
    corners: List[Tuple[int, int]] = field(default_factory=list)   # U's corners, delay ascending
    certificates: List[Tuple[float, float]] = field(default_factory=list)
    point_proven: List[bool] = field(default_factory=list)          # on the front, proven
    stretch_ok: List[bool] = field(default_factory=list)
    left_ok: bool = False
    right_ok: bool = False
    cls: str = "none"
    df: Optional[Step] = None
    sf: Optional[Step] = None
    notes: List[str] = field(default_factory=list)

    def L(self, k: float) -> float:
        """Largest certified lower bound on f(k); 0 when nothing is known."""
        return max([v for kc, v in self.certificates if k <= kc], default=0)

    def U(self, k: float) -> float:
        return min([s for d, s in self.corners if d <= k], default=INF)

    @property
    def certified_share(self) -> Optional[float]:
        """Share of the integer delays in [first corner, last corner] at which f is known."""
        if not self.corners:
            return None
        lo, hi = self.corners[0][0], self.corners[-1][0]
        known = sum(1 for k in range(lo, hi + 1) if self.L(k) >= self.U(k))
        return known / (hi - lo + 1)

    def lower_staircase(self) -> List[Tuple[int, float]]:
        """(k, L(k)) at every k in [first corner - 1, last corner] where L changes."""
        if not self.corners:
            return []
        lo, hi = self.corners[0][0] - 1, self.corners[-1][0]
        out, last = [], None
        for k in range(lo, hi + 1):
            value = self.L(k)
            if value != last:
                out.append((k, value))
                last = value
        return out


def _ends(steps: Sequence[Step]) -> Tuple[Optional[Step], Optional[Step]]:
    """The delay-first and sectors-first end, preferring a proven one, then the latest round."""
    def pick(key):
        cands = [s for s in steps if s.key == key and s.final is not None]
        cands.sort(key=lambda s: (s.exhausted and s.has_model, s.round))
        return cands[-1] if cands else next((s for s in steps if s.key == key), None)
    return pick("DF"), pick("SF")


def certify(label: str, steps: Sequence[Step], metric: Optional[str] = None) -> Front:
    metric = metric or parse_label(label)["metric"]
    front = Front(label=label, metric=metric)
    front.df, front.sf = _ends(steps)
    usable = [s for s in steps if not s.problems]
    for s in steps:
        for problem in s.problems:
            front.notes.append(f"{s.path.name}: {problem} (step ignored)")

    points = [cost for s in usable for cost in s.models]
    if not points:
        front.notes.append("no feasible point in any step")
        return front
    o_ub = min(p["o"] for p in points)
    o_lb = max([s.lower["o"] for s in usable if s.unbounded and s.lower is not None], default=None)
    front.o_star = o_ub
    front.o_known = o_lb is not None and o_lb == o_ub
    if not front.o_known:
        front.notes.append(f"least overload not proven (points reach {o_ub}, "
                           f"unbounded lower bound {o_lb})")
        o_star = o_ub
    else:
        o_star = o_ub

    at_o = sorted({(p["d"], p["s"]) for p in points if p["o"] == o_star})
    corners, best = [], INF
    for d, s in at_o:                                   # delay ascending: keep strict improvements
        if s < best:
            corners.append((d, s))
            best = s
    front.corners = corners

    certs: List[Tuple[float, float]] = []
    if front.o_known:
        if metric == "floored":
            certs.append((-1, INF))
        for s in usable:
            kc = INF if s.bound is None else s.bound
            if s.order == "sectors-first":
                if s.final is not None and s.exhausted and not s.has_model:
                    certs.append((kc, INF))
                    continue
                if s.lower is None:
                    continue
                if s.lower["o"] > o_star:
                    certs.append((kc, INF))
                elif s.lower["o"] == o_star:
                    certs.append((kc, s.lower["s"]))
                    if s.exhausted and s.incumbent is not None and s.incumbent["o"] == o_star:
                        certs.append((min(kc, s.incumbent["d"] - 1), s.incumbent["s"] + 1))
            else:  # delay-first, always unbounded
                if s.lower is None or s.lower["o"] != o_star:
                    continue
                certs.append((s.lower["d"] - 1, INF))
                if s.exhausted and s.incumbent is not None and s.incumbent["o"] == o_star:
                    certs.append((s.incumbent["d"], s.incumbent["s"]))
    front.certificates = certs

    L = front.L
    front.point_proven = [L(d) >= s and L(d - 1) > s for d, s in corners]
    front.stretch_ok = [L(d1 - 1) >= s0 and L(d1) >= s1
                        for (d0, s0), (d1, s1) in zip(corners, corners[1:])]
    if corners:
        d_first, s_first = corners[0]
        d_last, s_last = corners[-1]
        front.left_ok = L(d_first - 1) == INF and L(d_first) >= s_first
        front.right_ok = L(INF) >= s_last and L(d_last - 1) > s_last
    if front.o_known and corners:
        if front.left_ok and front.right_ok and all(front.stretch_ok):
            front.cls = "exact"
        elif any(front.stretch_ok):
            front.cls = "partial"
        elif front.left_ok:
            front.cls = "endpoints"
    if front.o_known and front.o_star and front.o_star > 0:
        front.notes.append(f"least overload is {front.o_star} > 0 at the fixed horizon "
                           f"{HORIZON}: V2 would have extended it")
    return front


def end_summary(front: Front) -> Dict[str, object]:
    """What round E established, as the planner needs it."""
    df, sf = front.df, front.sf
    df_proven = bool(df and df.exhausted and df.has_model and not df.problems)
    sf_proven = bool(sf and sf.exhausted and sf.has_model and not sf.problems)
    out = {"df_proven": df_proven, "sf_proven": sf_proven,
           "d_min": df.incumbent["d"] if df_proven else None,
           "s_at_d_min": df.incumbent["s"] if df_proven else None,
           "d_s": sf.incumbent["d"] if sf_proven else None,
           "s_min": sf.incumbent["s"] if sf_proven else None}
    if df_proven and sf_proven and df.incumbent["o"] != sf.incumbent["o"]:
        # Both ends are optima of programs with the same top level; this cannot happen unless
        # something is wrong, and a grid planned from it would be meaningless.
        out["df_proven"] = out["sf_proven"] = False
        front.notes.append(f"proven ends disagree on the overload: DF {df.incumbent['o']}, "
                           f"SF {sf.incumbent['o']}")
    return out


# ---------------------------------------------------------------------------------------------
# The five-column front CSV (bsc_student_stuff/20260923/pareto/plot_fronts.py reads this shape)
# ---------------------------------------------------------------------------------------------

def front_csv_rows(label: str, steps: Sequence[Step]) -> List[str]:
    """label,K|none,status,time,c1 c2 c3 c4 c5 c6 with the costs in sectors-first order.

    A proven delay-first end becomes a proven row at K = D_min: at overload o* no solution has less
    delay, so minimising sectors under K = D_min returns the same point. An unproven delay-first
    end is written at K = its own delay, as the incumbent it is.
    """
    rows = []
    for s in sorted(steps, key=_step_sort_key):
        cost = s.incumbent
        if s.key == "DF":
            k = str(cost["d"]) if cost is not None else "none"
        else:
            k = "none" if s.bound is None else str(s.bound)
        time = s.wall_s
        if time is None and s.final is not None:
            time = s.final.get("SOLVER-SEARCH-ENDED-S") or 0.0
        costs = " ".join(str(cost[level]) for level in LEVELS) if cost else "none"
        rows.append(f"{label},{k},{s.status},{(time or 0.0):.3f},{costs}")
    return rows


def _step_sort_key(s: Step):
    order = {"SF": 0, "DF": 2}.get(s.key, 1)
    return (order, -(s.bound if s.bound is not None else 0), s.round)
