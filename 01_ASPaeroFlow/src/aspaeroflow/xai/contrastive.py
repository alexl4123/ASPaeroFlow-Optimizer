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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import clingo

from .reasons import explain_step, step_context
from .subproblem import LEVELS, LEVEL_TEXT, amount, CandidatePath, Solution, Subproblem, SubproblemSolver, compare
from .trace import TraceReader

PathSet = Set[Tuple[int, int]]

SCOPE = ("Scope: this compares answers of one ASPaeroFlow step only - the candidate routes, delays and "
         "sector configurations generated for this step's hotspot. Plans outside these candidates are not "
         "considered, and the step's costs are local: total overload, then the arrival delay of the "
         "flights in the step, then the number of open sectors at the hotspot time, then the number of flights "
         "sent to the solver off the filed route or the earliest offered departure, then the index of the sector "
         "configuration.")


# ---------------------------------------------------------------------------- user locks

def _periods(n: int) -> str:
    """A duration in the interface's words ("step N" is the optimizer step, a duration is "N time periods")."""
    return "1 time period" if n == 1 else f"{n} time periods"


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
        # foils of the keep contrasts (keep_contrasts), solved once or read from keep_contrasts.jsonl (load_keep), so
        # that a row line and the dialog card about the same flight describe the same answer
        self._foils: Dict[Tuple, Solution] = {}
        self._memo: Dict[Tuple, Solution] = {}         # answers of the other cards, by their bans (_solve_bans)

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

    # ------------------------------------------------------------------ keep contrasts (row reasons, K12)

    def unchanged_paths(self, flight: int) -> Optional[List[int]]:
        """Every candidate path of a solver flight whose trajectory equals its current one (None: none, or unknown)."""
        cur = self.current.get(flight)
        if cur is None:
            return None
        found = [p for p in self._all_paths(flight) if self.sub.paths[flight][p].trajectory == cur]
        return found or None

    def _trajectory(self, flight: int, index: Optional[int]) -> Optional[Dict[int, int]]:
        if index is None or index not in self.sub.paths.get(flight, {}):
            return None
        return self.sub.paths[flight][index].trajectory

    def chosen_current(self, flight: int) -> bool:
        """The recorded answer gives the flight its current trajectory."""
        cur = self.current.get(flight)
        return cur is not None and self._trajectory(flight, self.chosen_paths.get(flight)) == cur

    def _solve_bans(self, ban_paths: PathSet = frozenset(), ban_configs: Iterable[int] = ()) -> Solution:
        """Best answer under these bans, solved once per explainer: a card and a what-if with the same bans then
        describe the same answer, also when several answers are equally good."""
        key = (frozenset(ban_paths), frozenset(ban_configs))
        if key not in self._memo:
            self._memo[key], _ = self.solver.solve(ban_paths=ban_paths, ban_configs=ban_configs)
        return self._memo[key]

    def _foil(self, key: Tuple, ban_paths: PathSet = frozenset(), ban_configs: Iterable[int] = ()) -> Solution:
        if key not in self._foils:
            self._foils[key], _ = self.solver.solve(ban_paths=ban_paths, ban_configs=ban_configs)
        return self._foils[key]

    def keep_flight(self, flight: int) -> Optional[Solution]:
        """Best answer of the step with the solver flight on its current trajectory (None: not offered or unknown)."""
        allowed = self.unchanged_paths(flight)
        if not allowed:
            return None
        return self._foil(("keep", flight), ban_paths=self._ban_all_but(flight, allowed))

    def keep_both(self, flight: int, other: int) -> Optional[Solution]:
        a, b = self.unchanged_paths(flight), self.unchanged_paths(other)
        if not a or not b:
            return None
        return self._foil(("keep_both", flight, other),
                          ban_paths=self._ban_all_but(flight, a) | self._ban_all_but(other, b))

    def alt_flight(self, flight: int) -> Optional[Solution]:
        """Best answer of the step with every path banned whose trajectory is the chosen one."""
        chosen = self._trajectory(flight, self.chosen_paths.get(flight))
        if chosen is None:
            return None
        same = {(flight, p) for p in self._all_paths(flight) if self.sub.paths[flight][p].trajectory == chosen}
        return self._foil(("alt", flight), ban_paths=same)

    def keep_layout(self) -> Optional[Solution]:
        """Best answer of the step with the current layout of the hotspot sector (configuration 0)."""
        if self.chosen_config == 0:
            return None
        return self._foil(("keep_layout",), ban_configs=[c for c in self.sub.configs if c != 0])

    def alt_layout(self) -> Optional[Solution]:
        if self.chosen_config == 0:
            return None
        return self._foil(("alt_layout",), ban_configs=[self.chosen_config])

    @staticmethod
    def _solution_entry(sol: Solution) -> Dict[str, Any]:
        return {"feasible": sol.satisfiable, "optimal": sol.optimal,
                "costs": dict(sol.costs) if sol.satisfiable else None, "clingo_cost": list(sol.clingo_cost),
                "chosen_paths": {str(f): p for f, p in sorted(sol.chosen_paths.items())} if sol.satisfiable else None,
                "chosen_config": sol.chosen_config}

    @staticmethod
    def _solution_of(entry: Dict[str, Any]) -> Solution:
        feasible = bool(entry.get("feasible"))
        return Solution(feasible, bool(entry.get("optimal")), dict(entry.get("costs") or {}),
                        list(entry.get("clingo_cost") or []), entry.get("chosen_config"),
                        {int(f): int(p) for f, p in (entry.get("chosen_paths") or {}).items()}, [], {}, 0.0)

    def _moved(self, foil: Solution, flight: int) -> Tuple[List[int], List[int]]:
        """(moves, others) of a keep-F foil: the solver flights other than F that keep their current trajectory in the
        recorded answer and leave it in the foil, and every solver flight other than F whose trajectory in the foil
        differs from the recorded answer."""
        moves, others = [], []
        for g in self.sub.decision_flights:
            if g == flight:
                continue
            in_foil = self._trajectory(g, foil.chosen_paths.get(g))
            if in_foil != self._trajectory(g, self.chosen_paths.get(g)):
                others.append(g)
                if self.chosen_current(g):
                    moves.append(g)
        return moves, others

    def keep_contrasts(self) -> Dict[str, Any]:
        """Raw numbers of the keep contrasts of this step (no text), one line of keep_contrasts.jsonl (xai/keep.py).

        For every changed solver flight F: keep F on its current trajectory, another trajectory than the chosen one
        (alt), and, when keeping F is exactly as good and moves exactly one other solver flight G, keep F and G
        together. For a chosen layout other than the current one: keep the current layout, and another layout."""
        start = time.time()
        changed = {int(f) for f in (self.record.get("flight_changes") or {})}
        flights: Dict[str, Any] = {}
        for f in self.sub.decision_flights:
            item: Dict[str, Any] = {"current_recorded": f in self.current, "chosen_current": self.chosen_current(f),
                                    "changed": f in changed, "unchanged_paths": self.unchanged_paths(f),
                                    "keep": None, "keep_both": None, "alt": None}
            if f in changed:
                keep = self.keep_flight(f)
                if keep is not None:
                    item["keep"] = self._solution_entry(keep)
                    if keep.satisfiable:
                        moves, others = self._moved(keep, f)
                        item["keep"]["moves"] = moves
                        item["keep"]["others"] = others
                        item["keep"]["layout_differs"] = keep.chosen_config != self.chosen_config
                        if compare(self.factual, keep) is None and len(moves) == 1:
                            both = self.keep_both(f, moves[0])
                            if both is not None:
                                item["keep_both"] = self._solution_entry(both) | {"flight": moves[0]}
                alt = self.alt_flight(f)
                if alt is not None:
                    item["alt"] = self._solution_entry(alt)
            flights[str(f)] = item
        layout: Dict[str, Any] = {"chosen_config": self.chosen_config, "keep": None, "alt": None}
        if self.chosen_config != 0:
            layout["keep"] = self._solution_entry(self.keep_layout())
            layout["alt"] = self._solution_entry(self.alt_layout())
        return {"v": 1, "iteration": self.iteration, "subproblems": len(self.record.get("subproblems") or []),
                "factual": self._solution_entry(self.factual),
                "recorded_cost": [int(c) for c in self.sub_record.get("clingo_cost") or []],
                "flights": flights, "layout": layout, "seconds": round(time.time() - start, 3),
                "clingo": clingo.__version__}

    def load_keep(self, entry: Dict[str, Any]) -> None:
        """Use the foils of a stored keep_contrasts line (same iteration) instead of solving them again."""
        if int(entry.get("iteration", -1)) != self.iteration:
            raise ValueError(f"keep contrasts of iteration {entry.get('iteration')} given to iteration {self.iteration}")
        for f, item in (entry.get("flights") or {}).items():
            f = int(f)
            if item.get("keep"):
                self._foils.setdefault(("keep", f), self._solution_of(item["keep"]))
            if item.get("alt"):
                self._foils.setdefault(("alt", f), self._solution_of(item["alt"]))
            if item.get("keep_both"):
                g = int(item["keep_both"]["flight"])
                self._foils.setdefault(("keep_both", f, g), self._solution_of(item["keep_both"]))
        layout = entry.get("layout") or {}
        if layout.get("keep"):
            self._foils.setdefault(("keep_layout",), self._solution_of(layout["keep"]))
        if layout.get("alt"):
            self._foils.setdefault(("alt_layout",), self._solution_of(layout["alt"]))

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

    def change_clause(self, flight: int, index: int) -> str:
        """What a candidate path does to the flight compared with its trajectory before the step, without the
        flight's name: "keeps its trajectory", "departs 1 time period later, flies 51-22-15-53 instead of ..."."""
        d = self.describe_path(flight, index)
        if "departure_shift" not in d:      # trajectory before the step not recorded: say only what the path is
            return "flies " + "-".join(map(str, d["route"])) + f" from time {d['start']}"
        if d.get("unchanged"):
            return "keeps its trajectory"
        parts = []
        shift = d.get("departure_shift", 0)
        if shift > 0:
            parts.append(f"departs {_periods(shift)} later")
        elif shift < 0:
            parts.append(f"departs {_periods(-shift)} earlier")
        if d.get("rerouted"):
            parts.append("flies " + "-".join(map(str, d["route"])) + " instead of " + "-".join(map(str, d["current_route"])))
        elif shift != 0:
            parts.append("on its current route")
        arr = d.get("arrival_shift", 0)
        if arr != shift and arr != 0:
            parts.append(f"arrives {_periods(abs(arr))} {'later' if arr > 0 else 'earlier'}")
        return ", ".join(parts) if parts else "keeps its trajectory"

    def path_text(self, flight: int, index: int) -> str:
        return f"flight {flight} " + self.change_clause(flight, index)

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

    def _contrast_text(self, label: str, foil: Solution,
                       chosen_name: str = "the chosen answer") -> Tuple[Optional[str], str]:
        """The verdict of one comparison: `label` names the foil, `chosen_name` the recorded answer."""
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
            text += f" {chosen_name[0].upper()}{chosen_name[1:]} accepts " + " and ".join(prices) + " for that."
        return deciding, text

    def _contrast(self, label: str, ban_paths: PathSet = frozenset(), ban_configs: Iterable[int] = (),
                  skip_flights=(), skip_config: bool = False, foil: Optional[Solution] = None) -> Contrast:
        if foil is None:
            foil = self._solve_bans(ban_paths, ban_configs)
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

    #: Step header lines (xai/reasons.py) that answer "why this hotspot and these flights".
    HOTSPOT_KINDS = ("hotspot", "retry", "rule", "capacity", "subproblem", "legs", "delays", "changed", "unchanged")

    def why_hotspot(self) -> Dict[str, Any]:
        """The step header's own sentences (one source for both): spot, rule, flights, delays, result."""
        h = self.record["hotspot"]
        flights = self.sub.decision_flights
        try:
            previous, rejected = step_context(self.trace.iterations, self.iteration)
            step = explain_step(self.record, previous=previous, run=self.trace.run, trace_folder=self.trace.folder,
                                rejected_before=rejected)
            lines = step["lines"]
            if step["errors"]:                   # record_error or render_error: its one line says why
                text = " ".join(line["text"] for line in lines)
            else:
                text = " ".join(line["text"] for line in lines
                                if line["kind"] in self.HOTSPOT_KINDS or line["kind"].startswith(
                                    ("hotspot_", "retry_", "capacity_", "subproblem_", "delays_", "changed_", "legs_")))
        except Exception as exc:  # the explanation endpoint must answer
            text = f"The explanation text of this step could not be produced ({type(exc).__name__})."
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
            keep = self.keep_flight(f)          # the same answer as the row line of this flight (keep_contrasts)
            if keep is not None:
                contrasts.append(self._contrast(f"keeping flight {f} as it was", skip_flights=[f], foil=keep))
            if chosen.get("rerouted"):
                same_route = [p for p in self._all_paths(f) if not self.describe_path(f, p).get("rerouted")]
                if same_route:
                    contrasts.append(self._contrast(f"delaying flight {f} on its current route instead",
                                                    ban_paths=self._ban_all_but(f, same_route)))
            if chosen.get("departure_shift", 0) > 0:
                # the foil allows every path of F without a later departure, the unchanged one included
                no_delay = [p for p in self._all_paths(f) if self.describe_path(f, p).get("departure_shift", 0) <= 0]
                if no_delay:
                    contrasts.append(self._contrast(f"not delaying flight {f}",
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
            contrasts = [self._contrast("keeping the hotspot's sectors as they are", skip_config=True,
                                        foil=self.keep_layout())]
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

    def _flight_of_lock(self, lock: Lock) -> int:
        """The solver flight a flight lock constrains (a later leg names the flight it follows); ValueError for a
        flight that is not part of the step, whose lock would ban nothing."""
        if not lock.args:
            raise ValueError(f"lock '{lock}' needs a flight")
        f = self.sub.decision_flight_of(lock.args[0])
        if f not in self.sub.decision_flights:
            raise ValueError(f"flight {lock.args[0]} is not part of step {self.iteration}")
        return f

    def _filed_before(self, flight: int) -> bool:
        """The flight's trajectory before this step is its filed one: no kept step before this one changed it (the
        run starts from the filed plan). False when a record does not list its changed flights."""
        if "flight_changes" not in self.record:
            return False
        for n, record in self.trace.iterations.items():
            if n >= self.iteration or not record.get("accepted"):
                continue
            changes = record.get("flight_changes")
            if changes is None:
                return False
            if str(flight) in {str(f) for f in changes}:
                return False
        return True

    def lock_text(self, lock: Lock) -> str:
        """The requirement in the interface's words (lower case; the menu capitalises it)."""
        k, a = lock.kind, lock.args
        if k == "keep_sectors":
            return self.config_text(0)
        if k == "avoid":
            if len(a) > 1:
                return f"flight {self._flight_of_lock(Lock('keep', a[1:]))} does not pass navpoint {a[0]}"
            return f"no flight of the step passes navpoint {a[0]}"
        f = self._flight_of_lock(lock)
        if k == "keep":
            return f"flight {f} keeps its trajectory from before the step"
        if k == "no_delay":
            if self._filed_before(f):
                return f"flight {f} departs on time, as in the filed plan"
            return f"flight {f} departs no later than before the step"
        if k == "path":
            return f"flight {f} takes candidate path {a[1]}"
        if k == "max_delay":
            return f"flight {f} and its later flights arrive at most {_periods(a[1])} after their planned arrival"
        raise ValueError(f"unknown lock kind: {k}")

    def lock_met(self, lock: Lock) -> bool:
        """The recorded answer of the step already satisfies the lock (no ban hits it)."""
        paths, configs = self._lock_bans(lock)
        return (not any((f, p) in paths for f, p in self.chosen_paths.items())
                and self.chosen_config not in configs)

    def what_if_menu(self) -> Dict[str, Any]:
        """The requirements a user can pick for this step, each with its state: open (it changes the answer of the
        step), met (the recorded answer already meets it) or not_offered (no candidate of the step meets it). A
        solver flight whose trajectory before the step is not recorded gets no entries (neither can be checked)."""
        entries: List[Dict[str, Any]] = []

        def entry(lock: Lock, flight: Optional[int], offered: bool) -> None:
            state = "not_offered" if not offered else ("met" if self.lock_met(lock) else "open")
            entries.append({"lock": str(lock), "kind": lock.kind, "flight": flight, "text": self.lock_text(lock),
                            "state": state})

        for f in self.sub.decision_flights:
            if f not in self.current:
                continue
            entry(Lock("keep", (f,)), f, bool(self.unchanged_paths(f)))
            entry(Lock("no_delay", (f,)), f,
                  any(self.describe_path(f, p).get("departure_shift", 0) <= 0 for p in self._all_paths(f)))
        entry(Lock("keep_sectors"), None, 0 in self.sub.configs)
        return {"iteration": self.iteration, "menu": entries}

    def _lock_bans(self, lock: Lock) -> Tuple[PathSet, Set[int]]:
        k, a = lock.kind, lock.args
        paths: PathSet = set()
        configs: Set[int] = set()
        if k == "keep_sectors":
            configs = {c for c in self.sub.configs if c != 0}
        elif k in ("keep", "path", "no_delay", "max_delay"):
            f = self._flight_of_lock(lock)
            if k == "keep":
                # every candidate whose trajectory is the one before the step stays allowed (the keep card's bans)
                paths = self._ban_all_but(f, self.unchanged_paths(f) or [])
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
            only = self._flight_of_lock(Lock("keep", a[1:])) if len(a) > 1 else None
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

    def _same_card(self, lock: Lock) -> Optional[Tuple[str, Solution]]:
        """The card of this step's explanation that compares exactly this one lock with the recorded answer (same
        bans), with that card's answer; None when the step has no such card."""
        if lock.kind == "keep_sectors":
            if self.chosen_config == 0:
                return None
            return "Keeping the hotspot's sectors as they are", self.keep_layout()
        if lock.kind not in ("keep", "no_delay"):
            return None
        f = self._flight_of_lock(lock)
        pf, pu = self.chosen_paths.get(f), self.unchanged(f)
        if pf is None or (pu is not None and pf == pu):     # why_flight: no changed-flight cards
            return None
        if lock.kind == "keep":
            keep = self.keep_flight(f)
            return None if keep is None else (f"Keeping flight {f} as it was", keep)
        if self.describe_path(f, pf).get("departure_shift", 0) <= 0:
            return None
        allowed = [p for p in self._all_paths(f) if self.describe_path(f, p).get("departure_shift", 0) <= 0]
        if not allowed:
            return None
        return f"Not delaying flight {f}", self._solve_bans(self._ban_all_but(f, allowed))

    def _members(self) -> List[Tuple[int, Optional[int]]]:
        """(flight, the solver flight it follows or None) for every flight of the sub-problem, solver flights first."""
        out: List[Tuple[int, Optional[int]]] = []
        for f in self.sub.decision_flights:
            out.append((f, None))
            out += [(l, f) for l, par in sorted(self.sub.parent.items()) if par == f]
        return out

    def _answer_of(self, paths: Dict[int, int], config: Optional[int]) -> Dict[str, Any]:
        """One answer of the step: every flight of the sub-problem with its path (a later flight follows the path
        index of its solver flight), what that does to it, and the sector configuration."""
        flights = []
        for m, parent in self._members():
            index = paths.get(parent if parent is not None else m)
            if index is None or index not in self.sub.paths.get(m, {}):
                continue
            path = self.sub.paths[m][index]
            flights.append({"flight": m, "parent": parent, "path": index, "text": self.change_clause(m, index),
                            "route": list(path.route),
                            "trajectory": {str(t): n for t, n in sorted(path.trajectory.items())},
                            "changed": path.trajectory != self.current.get(m)})
        return {"flights": flights, "config": config,
                "sectors": self.config_text(config) if config is not None else None}

    def _table(self, recorded: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        """Rows of the flights whose trajectory differs between the two answers, the sector configuration row when
        the configurations differ, and the clauses that are the same in both answers and differ from before the
        step."""
        mine = {e["flight"]: e for e in recorded["flights"]}
        rows, same = [], []
        for e in other["flights"]:
            r = mine.get(e["flight"])
            if r is None:
                continue
            if r["trajectory"] != e["trajectory"]:
                rows.append({"flight": e["flight"], "parent": e["parent"], "recorded": r["text"], "what_if": e["text"]})
            elif r["changed"]:
                same.append(f"flight {e['flight']} {e['text']}")
        config_row = None
        if recorded["config"] != other["config"]:
            config_row = {"time": self.record["hotspot"].get("time"), "recorded": recorded["sectors"],
                          "what_if": other["sectors"]}
        elif recorded["config"] not in (None, 0):
            same.append(recorded["sectors"])
        return {"rows": rows, "config": config_row, "same": same}

    def what_if(self, locks: List[Lock]) -> Dict[str, Any]:
        """The step's sub-problem solved again under the user's locks and compared with the recorded answer of the
        step. Nothing else is recomputed: later steps and the final plan are not touched. Locks the recorded answer
        already meets are not solved when all are met; otherwise all locks are solved together."""
        if not locks:
            raise ValueError("no lock given")
        step = f"step {self.iteration}"
        many = len(locks) > 1
        noun = "these requirements" if many else "this requirement"
        texts = [self.lock_text(lock) for lock in locks]              # raises for a flight outside the step
        requirements = [{"lock": str(lock), "text": text, "met": self.lock_met(lock)}
                        for lock, text in zip(locks, texts)]
        question = f"What if, in {step}, " + " and ".join(texts) + "?"
        out: Dict[str, Any] = {"iteration": self.iteration, "question": question, "requirements": requirements,
                               "already_met": False, "feasible": True, "same_as": None, "scope": SCOPE}
        if all(r["met"] for r in requirements):
            if not many and locks[0].kind == "avoid" and not self._lock_bans(locks[0])[0]:
                answer = (f"No candidate route of {step} passes navpoint {locks[0].args[0]}; {step}'s choice already "
                          f"meets this requirement.")
            else:
                answer = f"{step.capitalize()}'s choice already meets {noun}. Nothing is solved again."
            return out | {"already_met": True, "answer": answer}

        same = self._same_card(locks[0]) if not many else None
        if same is not None:
            out["same_as"], sol = same
            core: List[str] = []
        else:
            names = []
            for lock in locks:
                paths, configs = self._lock_bans(lock)
                self.solver.add_group(str(lock), paths=paths, configs=configs)
                names.append(str(lock))
            sol, core = self.solver.solve(groups=names)
            if not sol.satisfiable:
                # Shrink the core to a minimal one: drop every lock without which it stays unsatisfiable.
                core = list(core) or list(names)
                for name in list(core):
                    rest = [c for c in core if c != name]
                    if rest and not self.solver.solve(groups=rest)[0].satisfiable:
                        core = rest
                by_name = dict(zip(names, texts))
                core_texts = [by_name[c] for c in core]
                if len(core) == 1:
                    answer = (f"No combination of {step}'s candidate routes, delays and sector options meets the "
                              f"requirement: {core_texts[0]}.")
                else:
                    answer = (f"No combination of {step}'s candidate routes, delays and sector options meets these "
                              f"requirements together: " + " and ".join(core_texts) + ".")
                return out | {"feasible": False, "core": core, "core_texts": core_texts, "answer": answer}
        if not sol.satisfiable:          # the card's own answer: no candidate meets the lock
            text = self.lock_text(locks[0])
            return out | {"feasible": False, "core": [str(locks[0])], "core_texts": [text],
                          "answer": (f"No combination of {step}'s candidate routes, delays and sector options meets "
                                     f"the requirement: {text}.")}
        label = "the answer with your requirements" if many else "the answer with your requirement"
        deciding, verdict = self._contrast_text(label, sol, chosen_name=f"{step}'s choice")
        recorded = self._answer_of(self.chosen_paths, self.chosen_config)
        other = self._answer_of(sol.chosen_paths, sol.chosen_config)
        changes = self._differences(sol)
        answer = verdict + (" In that answer " + "; ".join(changes) + "." if changes else "")
        return out | {"verdict": verdict, "deciding_level": deciding, "answer": answer,
                      "answers": {"recorded": recorded, "what_if": other}, "table": self._table(recorded, other),
                      "costs": sol.costs, "chosen_paths": sol.chosen_paths, "chosen_config": sol.chosen_config,
                      "ladder": self._ladder(self.factual, sol)}
