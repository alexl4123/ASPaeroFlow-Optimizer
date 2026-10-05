"""Contrastive explanations of one ASPaeroFlow iteration, computed on its sub-problem.

Every answer compares the recorded choice of the iteration (the "factual") with the best answer of
the same sub-problem under one extra requirement (the "foil"): keep a flight as it was, delay it
without rerouting it, keep the current sectors, and so on. The two cost vectors are compared level
by level in the encoding's priority order; the first level at which they differ decides, and the
levels below it are what the decision cost. A user lock ("what if flight 18 keeps its route?") is a
foil the user writes; locks that cannot hold together come back as an unsatisfiable core.

Scope: everything here is relative to one decomposition step, i.e. to the candidate paths and the
sector configurations generated for that step's hotspot. Nothing is claimed about the global plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .subproblem import LEVELS, LEVEL_TEXT, amount, CandidatePath, Solution, Subproblem, SubproblemSolver, compare
from .trace import TraceReader

PathSet = Set[Tuple[int, int]]

SCOPE = ("Scope: this compares answers of one ASPaeroFlow step only - the candidate routes, delays and "
         "sector configurations generated for this step's hotspot. Plans outside these candidates are not "
         "considered, and the step's costs are local: network overload, then the arrival delay of the "
         "flights in the step, then the number of sectors at the hotspot time, then the number of changed "
         "flights, then the index of the sector configuration.")


# ---------------------------------------------------------------------------- user locks

@dataclass
class Lock:
    """A requirement a user puts on one step. `kind` is one of:

    keep F          flight F keeps its current trajectory
    path F P        flight F takes candidate path P
    no_delay F      flight F departs no later than now
    max_delay F D   flight F (and its later legs) arrives at most D steps after its planned arrival
    avoid N [F]     no flight (or only flight F) passes navpoint N
    keep_sectors    the hotspot's sectors stay as they are (configuration 0)
    """

    kind: str
    args: Tuple[int, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Lock":
        parts = text.replace(":", " ").replace(",", " ").split()
        return cls(parts[0], tuple(int(x) for x in parts[1:]))

    def __str__(self) -> str:
        return " ".join([self.kind] + [str(a) for a in self.args])


# ---------------------------------------------------------------------------- the explainer

@dataclass
class Contrast:
    label: str
    feasible: bool
    solution: Optional[Solution]
    deciding: Optional[str]
    text: str
    core: List[str] = field(default_factory=list)


class IterationExplainer:
    """Answers questions about one iteration of a trace."""

    def __init__(self, trace: TraceReader, iteration: int, k: int = 0):
        self.trace = trace
        self.iteration = iteration
        self.record = trace.iterations[iteration]
        self.sub_record = self.record["subproblems"][k]
        self.sub = Subproblem.from_instance(trace.instance(iteration, k))
        self.solver = SubproblemSolver(trace.encoding, self.sub, seed=int(trace.run.get("seed", 11904657)),
                                       options=trace.run.get("solver_options") or None)
        self.current: Dict[int, Dict[int, int]] = {}
        for f in self.sub.paths:
            cur = trace.current_trajectory(iteration, f, k)
            if cur is not None:
                self.current[f] = cur
        self.chosen_config = int(self.sub_record["chosen_config"])
        self.chosen_paths = {int(f): int(p) for f, p in self.sub_record["chosen_paths"].items()}
        self.factual = self._solve_factual()

    # ------------------------------------------------------------------ helpers

    def _all_paths(self, flight: int) -> List[int]:
        return sorted(self.sub.paths.get(flight, {}))

    def _ban_all_but(self, flight: int, allowed: Iterable[int]) -> PathSet:
        allowed = set(allowed)
        return {(flight, p) for p in self._all_paths(flight) if p not in allowed}

    def _solve_factual(self) -> Solution:
        bans: PathSet = set()
        for f in self.sub.decision_flights:
            if f in self.chosen_paths:
                bans |= self._ban_all_but(f, [self.chosen_paths[f]])
        configs = [c for c in self.sub.configs if c != self.chosen_config]
        sol, _ = self.solver.solve(ban_paths=bans, ban_configs=configs)
        return sol

    def unchanged(self, flight: int) -> Optional[int]:
        return self.sub.unchanged_path(flight, self.current.get(flight))

    def describe_path(self, flight: int, index: int) -> Dict[str, Any]:
        """A candidate path in terms of the flight's current trajectory."""
        path: CandidatePath = self.sub.paths[flight][index]
        cur = self.current.get(flight)
        out: Dict[str, Any] = {"flight": flight, "path": index, "route": list(path.route),
                               "start": min(path.trajectory), "arrival": max(path.trajectory)}
        planned = self.sub.planned_arrival.get(flight)
        if planned is not None:
            out["arrival_delay"] = max(path.trajectory) - planned
        if cur:
            cur_route: List[int] = []
            for t in sorted(cur):
                if not cur_route or cur_route[-1] != cur[t]:
                    cur_route.append(cur[t])
            out["departure_shift"] = min(path.trajectory) - min(cur)
            out["arrival_shift"] = max(path.trajectory) - max(cur)
            out["rerouted"] = list(path.route) != cur_route
            out["current_route"] = cur_route
            out["unchanged"] = path.trajectory == cur
        return out

    def path_text(self, flight: int, index: int) -> str:
        d = self.describe_path(flight, index)
        if d.get("unchanged"):
            return f"flight {flight} keeps its trajectory"
        parts = []
        shift = d.get("departure_shift", 0)
        if shift > 0:
            parts.append(f"departs {shift} step{'s' if shift != 1 else ''} later")
        elif shift < 0:
            parts.append(f"departs {-shift} step{'s' if shift != -1 else ''} earlier")
        if d.get("rerouted"):
            parts.append("flies " + "-".join(map(str, d["route"])) + " instead of " + "-".join(map(str, d["current_route"])))
        elif shift != 0:
            parts.append("on its current route")
        arr = d.get("arrival_shift", 0)
        if arr != shift and arr != 0:
            parts.append(f"arrives {abs(arr)} step{'s' if abs(arr) != 1 else ''} {'later' if arr > 0 else 'earlier'}")
        return f"flight {flight} " + ", ".join(parts) if parts else f"flight {flight} keeps its trajectory"

    def config_text(self, config: int) -> str:
        sc = self.sub.configs.get(config)
        if config == 0:
            return "the hotspot's sectors stay as they are"
        n = sc.number_sectors if sc else None
        n0 = self.sub.configs[0].number_sectors if 0 in self.sub.configs else None
        if n is not None and n0 is not None and n != n0:
            return f"sector configuration {config} ({n} sectors at the hotspot time instead of {n0})"
        return f"sector configuration {config}"

    def _ladder(self, a: Solution, b: Solution) -> List[Dict[str, Any]]:
        deciding = compare(a, b)
        rows, seen = [], False
        for _, name in LEVELS:
            va, vb = a.costs.get(name, 0), b.costs.get(name, 0)
            rows.append({"level": name, "text": LEVEL_TEXT[name], "chosen": va, "alternative": vb,
                         "difference": vb - va, "decides": name == deciding, "below": seen})
            if name == deciding:
                seen = True
        return rows

    def _differences(self, foil: Solution, skip_flights=(), skip_config: bool = False) -> List[str]:
        """What else the foil changes compared with the factual (other flights, the configuration)."""
        out = []
        for f in self.sub.decision_flights:
            pf, pa = self.chosen_paths.get(f), foil.chosen_paths.get(f)
            if pa is not None and pa != pf and f not in skip_flights:
                out.append(self.path_text(f, pa))
        if not skip_config and foil.chosen_config is not None and foil.chosen_config != self.chosen_config:
            out.append(self.config_text(foil.chosen_config))
        return out

    def _contrast_text(self, label: str, foil: Solution) -> Tuple[Optional[str], str]:
        deciding = compare(self.factual, foil)
        if deciding is None:
            return None, (f"{label.capitalize()} is exactly as good on every criterion. The step's choice "
                          f"between the two is a tie broken by the solver, not a reason.")
        fv, av = self.factual.costs[deciding], foil.costs[deciding]
        if av < fv:
            return deciding, (f"{label.capitalize()} would be better on {LEVEL_TEXT[deciding]} ({av} instead of "
                              f"{fv}); the recorded choice is not optimal for this step (time limit or a defect).")
        text = (f"{label.capitalize()} would mean {amount(deciding, av - fv)} "
                f"({LEVEL_TEXT[deciding]} {av} instead of {fv}).")
        prices = []
        below = False
        for _, name in LEVELS:
            if name == deciding:
                below = True
                continue
            if below and self.factual.costs[name] > foil.costs[name]:
                prices.append(amount(name, self.factual.costs[name] - foil.costs[name]))
        if prices:
            text += " The chosen answer accepts " + " and ".join(prices) + " for that."
        return deciding, text

    def _contrast(self, label: str, ban_paths: PathSet = frozenset(), ban_configs: Iterable[int] = (),
                  skip_flights=(), skip_config: bool = False) -> Contrast:
        foil, core = self.solver.solve(ban_paths=ban_paths, ban_configs=ban_configs)
        if not foil.satisfiable:
            return Contrast(label, False, None, None,
                            f"{label.capitalize()} is impossible in this step: no combination of the candidates satisfies it.")
        deciding, text = self._contrast_text(label, foil)
        others = self._differences(foil, skip_flights=skip_flights, skip_config=skip_config)
        if others:
            text += " In that answer " + "; ".join(others) + "."
        return Contrast(label, True, foil, deciding, text)

    def _payload(self, question: str, answer: str, contrasts: List[Contrast], extra: Optional[Dict] = None) -> Dict[str, Any]:
        out = {
            "iteration": self.iteration,
            "question": question,
            "answer": answer,
            "factual": {"costs": self.factual.costs, "chosen_config": self.chosen_config,
                        "chosen_paths": self.chosen_paths},
            "contrasts": [{
                "label": c.label, "feasible": c.feasible, "deciding_level": c.deciding, "text": c.text,
                "costs": c.solution.costs if c.solution else None,
                "ladder": self._ladder(self.factual, c.solution) if c.solution else None,
                "core": c.core,
            } for c in contrasts],
            "scope": SCOPE,
        }
        if extra:
            out.update(extra)
        return out

    # ------------------------------------------------------------------ questions

    def why_hotspot(self) -> Dict[str, Any]:
        h = self.record["hotspot"]
        p = self.record.get("parameters", {})
        flights = self.sub.decision_flights
        later = sorted(self.sub.parent)
        text = (f"Sector {h['sector']} at time step {h['time']} had {h.get('demand', '?')} flights for a capacity of "
                f"{h.get('capacity', '?')}. ASPaeroFlow always works on the earliest overloaded time step (and there on "
                f"the overloaded sector with the lowest number). Of the flights in that sector it takes the "
                f"{p.get('max_aircraft', len(flights))} with the shortest flight time: "
                f"{', '.join(map(str, flights))}" + (f", together with the later legs of the same aircraft "
                f"({', '.join(map(str, later))})" if later else "") + ". This is a fixed rule, not an optimisation.")
        if p.get("failed_attempts_before"):
            text += (f" Before this step, {p['failed_attempts_before']} attempt(s) at this hotspot had failed, so "
                     f"the delay window was widened {p.get('additional_time_increase', 0)} time(s).")
        return {"iteration": self.iteration, "question": "Why this hotspot and these flights?", "answer": text,
                "hotspot": h, "decision_flights": flights, "later_legs": self.sub.parent}

    def why_flight(self, flight: int) -> Dict[str, Any]:
        f = self.sub.decision_flight_of(flight)
        if f not in self.sub.decision_flights:
            return {"iteration": self.iteration, "question": f"Why flight {flight}?",
                    "answer": f"Flight {flight} was not part of this step.", "contrasts": [], "scope": SCOPE}
        pf = self.chosen_paths[f]
        pu = self.unchanged(f)
        chosen = self.describe_path(f, pf)
        contrasts: List[Contrast] = []
        if pu is not None and pf == pu:
            question = f"Why was flight {f} not changed?"
            answer = f"In this step {self.path_text(f, pf)}."
            contrasts.append(self._contrast(f"changing flight {f}", ban_paths={(f, pu)}))
        else:
            question = f"Why was flight {f} changed?"
            answer = f"In this step {self.path_text(f, pf)}."
            if pu is not None:
                contrasts.append(self._contrast(f"keeping flight {f} as it was", ban_paths=self._ban_all_but(f, [pu]),
                                                skip_flights=[f]))
            if chosen.get("rerouted"):
                same_route = [p for p in self._all_paths(f) if not self.describe_path(f, p).get("rerouted")]
                if same_route:
                    contrasts.append(self._contrast(f"delaying flight {f} on its current route instead",
                                                    ban_paths=self._ban_all_but(f, same_route)))
            if chosen.get("departure_shift", 0) > 0:
                no_delay = [p for p in self._all_paths(f) if self.describe_path(f, p).get("departure_shift", 0) <= 0]
                if no_delay:
                    contrasts.append(self._contrast(f"rerouting flight {f} without delaying it",
                                                    ban_paths=self._ban_all_but(f, no_delay)))
        if flight != f:
            answer = f"Flight {flight} is a later leg of the aircraft that flies flight {f}; it moves with it. " + answer
        return self._payload(question, answer, contrasts, {"chosen_path": chosen})

    def why_sectors(self) -> Dict[str, Any]:
        c = self.chosen_config
        if c == 0:
            question = "Why were the sectors not changed?"
            answer = "In this step " + self.config_text(0) + "."
            contrasts = [self._contrast("reconfiguring the hotspot's sectors", ban_configs=[0])]
            # (the configuration the foil picks is worth naming here)
        else:
            question = "Why were the sectors changed?"
            answer = "In this step ASPaeroFlow chose " + self.config_text(c) + "."
            contrasts = [self._contrast("keeping the hotspot's sectors as they are",
                                        ban_configs=[x for x in self.sub.configs if x != 0], skip_config=True)]
        changes = self.record.get("sector_changes") or {}
        return self._payload(question, answer, contrasts, {"sector_changes": changes})

    def tie_check(self) -> Dict[str, Any]:
        """Is the recorded answer the only best one? (Ties are broken by the solver's search order.)"""
        body = ", ".join([f"chosen_config({self.chosen_config})"] +
                         [f"chosen_path({f},{p})" for f, p in sorted(self.chosen_paths.items())
                          if f in self.sub.decision_flights])
        self.solver.add_group("not_factual", bodies=[body])
        other, _ = self.solver.solve(groups=["not_factual"])
        tie = other.satisfiable and compare(self.factual, other) is None
        out = {"tie": tie, "factual_costs": self.factual.costs}
        if tie:
            out["equally_good"] = self._differences(other)
        return out

    def alternatives(self) -> Dict[str, Any]:
        """Best answer for every route of every decision flight and for every configuration, ranked."""
        rows = []
        for f in self.sub.decision_flights:
            routes: Dict[Tuple[int, ...], List[int]] = {}
            for p in self._all_paths(f):
                routes.setdefault(tuple(self.sub.paths[f][p].route), []).append(p)
            for route, idx in routes.items():
                sol, _ = self.solver.solve(ban_paths=self._ban_all_but(f, idx))
                rows.append({"kind": "route", "flight": f, "route": list(route),
                             "chosen": self.chosen_paths.get(f) in idx,
                             "best_path": sol.chosen_paths.get(f) if sol.satisfiable else None,
                             "costs": sol.costs if sol.satisfiable else None,
                             "deciding_level": compare(self.factual, sol) if sol.satisfiable else None})
        for c in sorted(self.sub.configs):
            sol, _ = self.solver.solve(ban_configs=[x for x in self.sub.configs if x != c])
            rows.append({"kind": "config", "config": c, "number_sectors": self.sub.configs[c].number_sectors,
                         "chosen": c == self.chosen_config,
                         "costs": sol.costs if sol.satisfiable else None,
                         "deciding_level": compare(self.factual, sol) if sol.satisfiable else None})
        key = lambda r: tuple(r["costs"][n] for _, n in LEVELS) if r["costs"] else (float("inf"),)
        rows.sort(key=key)
        return {"iteration": self.iteration, "factual": self.factual.costs, "alternatives": rows, "scope": SCOPE}

    # ------------------------------------------------------------------ what-if with user locks

    def _lock_bans(self, lock: Lock) -> Tuple[PathSet, Set[int]]:
        k, a = lock.kind, lock.args
        paths: PathSet = set()
        configs: Set[int] = set()
        if k == "keep_sectors":
            configs = {c for c in self.sub.configs if c != 0}
        elif k in ("keep", "path", "no_delay", "max_delay"):
            f = self.sub.decision_flight_of(a[0])
            if k == "keep":
                pu = self.unchanged(f)
                paths = self._ban_all_but(f, [pu] if pu is not None else [])
            elif k == "path":
                paths = self._ban_all_but(f, [a[1]])
            elif k == "no_delay":
                paths = {(f, p) for p in self._all_paths(f) if self.describe_path(f, p).get("departure_shift", 0) > 0}
            else:
                limit = a[1]
                members = [f] + [l for l, par in self.sub.parent.items() if par == f]
                for p in self._all_paths(f):
                    for m in members:
                        cp = self.sub.paths.get(m, {}).get(p)
                        planned = self.sub.planned_arrival.get(m)
                        if cp and planned is not None and cp.trajectory and max(cp.trajectory) - planned > limit:
                            paths.add((f, p))
                            break
        elif k == "avoid":
            nav = a[0]
            only = self.sub.decision_flight_of(a[1]) if len(a) > 1 else None
            for f in self.sub.decision_flights:
                if only is not None and f != only:
                    continue
                members = [f] + [l for l, par in self.sub.parent.items() if par == f]
                for p in self._all_paths(f):
                    if any(nav in self.sub.paths.get(m, {}).get(p, CandidatePath(m, p, 0, None, {}, ())).route
                           for m in members):
                        paths.add((f, p))
        else:
            raise ValueError(f"unknown lock kind: {k}")
        return paths, configs

    def what_if(self, locks: List[Lock]) -> Dict[str, Any]:
        names = []
        for lock in locks:
            paths, configs = self._lock_bans(lock)
            self.solver.add_group(str(lock), paths=paths, configs=configs)
            names.append(str(lock))
        sol, core = self.solver.solve(groups=names)
        question = "What if " + " and ".join(names) + "?"
        if not sol.satisfiable:
            # Shrink the core to a minimal one: drop every lock without which it stays unsatisfiable.
            core = list(core) or list(names)
            for name in list(core):
                rest = [c for c in core if c != name]
                if rest and not self.solver.solve(groups=rest)[0].satisfiable:
                    core = rest
            if len(core) == 1:
                answer = (f"'{core[0]}' cannot hold in this step: it excludes every candidate of the flight "
                          f"(or configuration) it constrains.")
            else:
                answer = ("These requirements cannot hold together in this step: " +
                          " and ".join(f"'{c}'" for c in core) + " exclude every combination of the candidates.")
            return {"iteration": self.iteration, "question": question, "answer": answer, "feasible": False,
                    "core": core, "scope": SCOPE}
        deciding, text = self._contrast_text("this answer", sol)
        changes = self._differences(sol)
        answer = ("With these requirements the step's best answer: " + ("; ".join(changes) if changes else
                  "is the same choice as recorded") + ". " + text)
        return {"iteration": self.iteration, "question": question, "answer": answer, "feasible": True,
                "costs": sol.costs, "chosen_paths": sol.chosen_paths, "chosen_config": sol.chosen_config,
                "ladder": self._ladder(self.factual, sol), "scope": SCOPE}
