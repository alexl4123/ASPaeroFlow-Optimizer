"""One ASPaeroFlow iteration's sub-problem, read back from its ASP instance and solved again.

Every iteration of the heuristic builds a self-contained ASP instance (facts only) for the first
overloaded (time, sector) pair: the candidate paths of at most a few flights, the sector
configurations that may replace the current one, and the residual capacities. The encoding then
picks one path per flight and one configuration under five lexicographic weak constraints (see
01_ASPaeroFlow/encoding.lp):

    priority 10  overload     overload inside the instance + overload elsewhere under the configuration
    priority  9  delay        arrival delay of the flights in the instance
    priority  8  sectors      number of sectors at the hotspot time under the configuration
    priority  7  changed      number of flights whose path is not path 0
    priority  6  config       index of the configuration (prefers the current one, index 0)

Because the instance is self-contained, a contrastive question about the iteration ("why was flight
F rerouted?") is answered by solving the same instance again with one more constraint and comparing
the two cost vectors. No matrices and no second optimizer process are needed.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import clingo

#: Priority levels of encoding.lp, highest first, with the name the explanations use for each.
LEVELS: Tuple[Tuple[int, str], ...] = (
    (10, "overload"),
    (9, "delay"),
    (8, "sectors"),
    (7, "changed"),
    (6, "config"),
)

def amount(level: str, n: int) -> str:
    """'3 more units of overload', '1 more time period of arrival delay', ... for a level and a count."""
    n = abs(int(n))
    s = "" if n == 1 else "s"
    return {
        "overload": f"{n} more unit{s} of overload",
        "delay": f"{n} more time period{s} of arrival delay",
        "sectors": f"{n} more sector{s}",
        "changed": f"{n} more changed flight{s}",
        "config": "a sector configuration further from the current one",
    }[level]


LEVEL_TEXT = {
    "overload": "network overload",
    "delay": "arrival delay",
    "sectors": "number of sectors",
    "changed": "number of changed flights",
    "config": "deviation from the current sector configuration",
}


@dataclass
class CandidatePath:
    """One candidate path of one flight: a route and a departure delay, as the instance encodes it."""

    flight: int
    index: int
    start: int                      # time the flight's operations start (after the delay)
    arrival: Optional[int]          # actual arrival time step of this candidate
    trajectory: Dict[int, int]      # absolute time step -> navpoint
    route: Tuple[int, ...]          # navpoints in order, without repeats

    def delay_against(self, planned_arrival: Optional[int]) -> Optional[int]:
        if planned_arrival is None or self.arrival is None:
            return None
        return self.arrival - planned_arrival


@dataclass
class SectorConfig:
    """One sector configuration the sub-problem may choose."""

    index: int
    number_sectors: Optional[int]   # sectors at the hotspot time under this configuration
    outside_overload: Optional[int]  # overload outside the instance under this configuration
    assignment: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (navpoint, time) -> sector


@dataclass
class Subproblem:
    """The facts of one iteration's instance, organised for explanation."""

    instance: str
    paths: Dict[int, Dict[int, CandidatePath]]   # flight -> path index -> path
    planned_arrival: Dict[int, int]
    configs: Dict[int, SectorConfig]
    decision_flights: List[int]                  # flights with a choice of path
    parent: Dict[int, int] = field(default_factory=dict)  # later leg -> the decision flight it follows

    @classmethod
    def from_instance(cls, instance: str) -> "Subproblem":
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], instance)
        ctl.ground([("base", [])])

        starts: Dict[Tuple[int, int], int] = {}
        arrivals: Dict[Tuple[int, int], int] = {}
        hops: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        singles: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        path_ids: Dict[int, List[int]] = {}
        planned: Dict[int, int] = {}
        configs: Dict[int, SectorConfig] = {}

        def cfg(c: int) -> SectorConfig:
            if c not in configs:
                configs[c] = SectorConfig(index=c, number_sectors=None, outside_overload=None)
            return configs[c]

        for atom in ctl.symbolic_atoms:
            s = atom.symbol
            a = s.arguments
            if s.name == "paths" and len(a) == 2:
                path_ids.setdefault(a[0].number, []).append(a[1].number)
            elif s.name == "actual_flight_operations_start_time" and len(a) == 3:
                starts[(a[0].number, a[2].number)] = a[1].number
            elif s.name == "actual_arrival_time" and len(a) == 3:
                arrivals[(a[0].number, a[2].number)] = a[1].number
            elif s.name == "next_pos" and len(a) == 6:
                hops.setdefault((a[0].number, a[1].number), []).append(
                    (a[2].number, a[3].number, a[4].number, a[5].number))
            elif s.name == "single_pos" and len(a) == 4:
                singles.setdefault((a[0].number, a[1].number), []).append((a[2].number, a[3].number))
            elif s.name == "planned_arrival_time" and len(a) == 2:
                planned[a[0].number] = a[1].number
            elif s.name == "config" and len(a) in (1, 2):
                c = cfg(a[0].number)
                if len(a) == 2:
                    c.outside_overload = a[1].number
            elif s.name == "config_number_sectors" and len(a) == 2:
                cfg(a[0].number).number_sectors = a[1].number
            elif s.name == "possible_assignment" and len(a) == 4:
                cfg(a[3].number).assignment[(a[0].number, a[2].number)] = a[1].number

        paths: Dict[int, Dict[int, CandidatePath]] = {}
        keys = set(starts) | set(hops) | set(singles)
        for flight, index in keys:
            start = starts.get((flight, index), 0)
            trajectory: Dict[int, int] = {}
            for nav, t0 in singles.get((flight, index), []):
                trajectory[t0 + start] = nav
            for nav0, t0, nav1, t1 in hops.get((flight, index), []):
                trajectory[t0 + start] = nav0
                trajectory[t1 + start] = nav1
            route: List[int] = []
            for t in sorted(trajectory):
                if not route or route[-1] != trajectory[t]:
                    route.append(trajectory[t])
            paths.setdefault(flight, {})[index] = CandidatePath(
                flight=flight, index=index, start=start, arrival=arrivals.get((flight, index)),
                trajectory=trajectory, route=tuple(route))

        # Later legs of the same aircraft follow the path index of the leg in the hotspot:
        #   chosen_path(LATER,P) :- chosen_path(FLIGHT,P).
        parent: Dict[int, int] = {}
        for later, _p, flight in re.findall(r"chosen_path\((\d+),(\d+)\)\s*:-\s*chosen_path\((\d+),\2\)", instance):
            parent[int(later)] = int(flight)

        return cls(instance=instance, paths=paths, planned_arrival=planned, configs=configs,
                   decision_flights=sorted(path_ids), parent=parent)

    def decision_flight_of(self, flight: int) -> int:
        return self.parent.get(flight, flight)

    # ------------------------------------------------------------------ descriptions

    def unchanged_path(self, flight: int, current: Optional[Dict[int, int]]) -> Optional[int]:
        """Index of the candidate path that equals the flight's current trajectory, if any."""
        if current is None:
            return None
        for index, path in sorted(self.paths.get(flight, {}).items()):
            if path.trajectory == current:
                return index
        return None

    def config_sectors_at(self, config: int, t: int) -> Dict[int, List[int]]:
        """sector -> navpoints at time t under a configuration (only navpoints in the instance)."""
        out: Dict[int, List[int]] = {}
        for (nav, time_), sec in self.configs[config].assignment.items():
            if time_ == t:
                out.setdefault(sec, []).append(nav)
        return {sec: sorted(navs) for sec, navs in sorted(out.items())}


@dataclass
class Solution:
    """An optimal answer of a sub-problem, possibly under extra constraints."""

    satisfiable: bool
    optimal: bool
    costs: Dict[str, int]                       # level name -> cost, computed from the atoms
    clingo_cost: List[int]
    chosen_config: Optional[int]
    chosen_paths: Dict[int, int]
    overloads: List[Tuple[int, int, int]]       # (sector, time, amount) inside the instance
    delays: Dict[int, int]                      # flight -> arrival delay (as scored)
    seconds: float

    def vector(self) -> Tuple[int, ...]:
        return tuple(self.costs.get(name, 0) for _, name in LEVELS)


def solve(encoding: str, instance: str, extra: str = "", seed: int = 11904657,
          options: Optional[List[str]] = None, time_limit: Optional[float] = None) -> Solution:
    """One-shot: solve encoding + instance + extra to optimality and read every level's cost off the atoms."""
    start = time.time()
    ctl = clingo.Control([f"--seed={seed}", "--warn=none"] + list(options or []))
    last: Dict[str, object] = {}

    def on_model(model: clingo.Model) -> None:
        last["atoms"] = list(model.symbols(atoms=True))
        last["cost"] = list(model.cost)
        last["optimal"] = model.optimality_proven

    ctl.add("base", [], encoding + "\n" + instance + "\n" + extra)
    ctl.ground([("base", [])])
    with ctl.solve(on_model=on_model, async_=True) as handle:
        if time_limit is not None and not handle.wait(time_limit):
            handle.cancel()
        result = handle.get()
    seconds = time.time() - start
    if "atoms" not in last:
        return Solution(False, bool(result.exhausted), {}, [], None, {}, [], {}, seconds)
    return _solution_from(last["atoms"], last["cost"], bool(last["optimal"]) or bool(result.exhausted), seconds)


def compare(factual: Solution, foil: Solution) -> Optional[str]:
    """The highest-priority level at which the two solutions differ (None if they tie everywhere)."""
    for _, name in LEVELS:
        if factual.costs.get(name, 0) != foil.costs.get(name, 0):
            return name
    return None


def _solution_from(atoms, cost, optimal: bool, seconds: float) -> Solution:
    chosen_config = None
    chosen_paths: Dict[int, int] = {}
    overloads: List[Tuple[int, int, int]] = []
    delays: Dict[int, int] = {}
    reroutes = 0
    config_outside: Dict[int, int] = {}
    config_sectors: Dict[int, int] = {}
    for s in atoms:
        a = s.arguments
        if s.name == "chosen_config":
            chosen_config = a[0].number
        elif s.name == "chosen_path":
            chosen_paths[a[0].number] = a[1].number
        elif s.name == "overload" and len(a) == 3:
            overloads.append((a[0].number, a[1].number, a[2].number))
        elif s.name == "arrival_delay" and len(a) == 2:
            delays[a[0].number] = a[1].number
        elif s.name == "reroute":
            reroutes += 1
        elif s.name == "config" and len(a) == 2:
            config_outside[a[0].number] = a[1].number
        elif s.name == "config_number_sectors" and len(a) == 2:
            config_sectors[a[0].number] = a[1].number
    costs = {
        "overload": sum(o for _, _, o in overloads) + config_outside.get(chosen_config, 0),
        "delay": sum(delays.values()),
        "sectors": config_sectors.get(chosen_config, 0),
        "changed": reroutes,
        "config": chosen_config or 0,
    }
    return Solution(True, optimal, costs, [int(c) for c in cost], chosen_config, chosen_paths,
                    sorted(overloads), delays, seconds)


class SubproblemSolver:
    """encoding + instance grounded once; candidates are banned and lock groups switched on per query.

    Bans (`xai_no_path`, `xai_no_config`) are external atoms of the base program. A lock group is a
    set of bans added later as its own program part behind one external atom `xai_group(n)`, which
    the query passes as an assumption, so that an infeasible combination of locks comes back as an
    unsatisfiable core naming the groups that conflict.
    """

    XAI_PROGRAM = """
#external xai_no_path(F,P) : paths(F,P).
:- xai_no_path(F,P), chosen_path(F,P).
#external xai_no_config(C) : config(C,_).
#external xai_no_config(C) : config(C).
:- xai_no_config(C), chosen_config(C).
"""

    def __init__(self, encoding: str, sub: Subproblem, seed: int = 11904657,
                 options: Optional[List[str]] = None):
        self.sub = sub
        self.ctl = clingo.Control([f"--seed={seed}", "--warn=none"] + list(options or []))
        self.ctl.add("base", [], encoding + "\n" + sub.instance + "\n" + self.XAI_PROGRAM)
        self.ctl.ground([("base", [])])
        self._true: set = set()
        self._groups: Dict[str, clingo.Symbol] = {}

    def add_group(self, name: str, paths=(), configs=(), bodies=()) -> None:
        """A named set of bans that queries can switch on together (e.g. one user lock).

        `bodies` are extra constraint bodies, e.g. "chosen_config(4), chosen_path(1,74)" to ban one
        combination of choices."""
        if name in self._groups:
            return
        n = len(self._groups)
        atom = clingo.Function("xai_group", [clingo.Number(n)])
        rules = [f"#external xai_group({n})."]
        rules += [f":- xai_group({n}), chosen_path({f},{p})." for f, p in sorted(paths)]
        rules += [f":- xai_group({n}), chosen_config({c})." for c in sorted(configs)]
        rules += [f":- xai_group({n}), {body}." for body in bodies]
        part = f"xai_group_{n}"
        self.ctl.add(part, [], "\n".join(rules))
        self.ctl.ground([(part, [])])
        self._groups[name] = atom

    def solve(self, ban_paths=(), ban_configs=(), groups=(), time_limit: Optional[float] = None):
        """Optimal answer under the bans and the switched-on groups. Returns (Solution, core group names)."""
        wanted = {clingo.Function("xai_no_path", [clingo.Number(f), clingo.Number(p)]) for f, p in ban_paths}
        wanted |= {clingo.Function("xai_no_config", [clingo.Number(c)]) for c in ban_configs}
        for atom in self._true - wanted:
            self.ctl.assign_external(atom, False)
        for atom in wanted - self._true:
            self.ctl.assign_external(atom, True)
        self._true = wanted
        by_atom = {self._groups[g]: g for g in groups}
        for atom in self._groups.values():
            # Switched-on groups are left open and assumed true, so that a conflict shows in the core.
            self.ctl.assign_external(atom, None if atom in by_atom else False)
        assumptions = [(atom, True) for atom in by_atom]

        start = time.time()
        last: Dict[str, object] = {}
        core: List[str] = []

        def on_model(model: clingo.Model) -> None:
            last["atoms"] = list(model.symbols(atoms=True))
            last["cost"] = list(model.cost)
            last["optimal"] = model.optimality_proven

        def on_core(literals) -> None:
            lits = set(literals)
            for atom, name in by_atom.items():
                lit = self.ctl.symbolic_atoms[atom].literal if atom in self.ctl.symbolic_atoms else None
                if lit is not None and lit in lits:
                    core.append(name)

        if time_limit is None:
            result = self.ctl.solve(assumptions=assumptions, on_model=on_model, on_core=on_core)
        else:
            # (the core callback is not delivered in asynchronous mode, so locks are best solved without a limit)
            with self.ctl.solve(assumptions=assumptions, on_model=on_model, on_core=on_core, async_=True) as handle:
                if not handle.wait(time_limit):
                    handle.cancel()
                result = handle.get()
        seconds = time.time() - start
        if "atoms" not in last:
            return Solution(False, bool(result.exhausted), {}, [], None, {}, [], {}, seconds), sorted(core)
        optimal = bool(last["optimal"]) or bool(result.exhausted)
        return _solution_from(last["atoms"], last["cost"], optimal, seconds), []
