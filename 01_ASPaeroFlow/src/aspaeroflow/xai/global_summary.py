"""A whole run summarised from its local explanations (first step towards global explanations).

Per step: what the step did (delay, reroute, sector split, or a combination), which priority level
decided against doing nothing (all decision flights kept and the sectors unchanged), whether the
step was tied, and the change of the objective values. Over the run: how often each kind of
action and each deciding level occurred, the share of tied steps, and where each flight's final
arrival delay came from (the steps and hotspots that moved it).

Fidelity: everything here is computed from the recorded steps, so it is exact for this run; it
says nothing about other instances (that is the job of a surrogate model fitted over many runs,
see notes/XAI_GLOBAL_FIDELITY.md).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from .contrastive import IterationExplainer
from .subproblem import LEVELS, compare
from .trace import TraceReader


def step_summary(ex: IterationExplainer) -> Dict[str, Any]:
    record = ex.record
    actions = set()
    flights = {}
    for f in ex.sub.decision_flights:
        d = ex.describe_path(f, ex.chosen_paths[f])
        kind = []
        if d.get("departure_shift", 0) > 0:
            kind.append("delay")
        if d.get("rerouted"):
            kind.append("reroute")
        flights[f] = kind or ["unchanged"]
        actions.update(kind)
    if ex.chosen_config != 0:
        actions.add("split")

    # the step against doing nothing at all: keep every decision flight, keep the sectors
    bans = set()
    keep_possible = True
    for f in ex.sub.decision_flights:
        pu = ex.unchanged(f)
        if pu is None:
            keep_possible = False
            continue
        bans |= ex._ban_all_but(f, [pu])
    foil, _ = ex.solver.solve(ban_paths=bans, ban_configs=[c for c in ex.sub.configs if c != 0])
    deciding = compare(ex.factual, foil) if foil.satisfiable else None
    margin = (foil.costs[deciding] - ex.factual.costs[deciding]) if deciding else 0

    # was each lever the step used necessary? the same step without it: worse (at which level) or as good
    levers = {}
    without = {
        "split": (set(), [c for c in ex.sub.configs if c != 0]) if ex.chosen_config != 0 else None,
        "delay": ({(f, p) for f in ex.sub.decision_flights for p in ex._all_paths(f)
                   if ex.describe_path(f, p).get("departure_shift", 0) > 0}, []) if "delay" in actions else None,
        "reroute": ({(f, p) for f in ex.sub.decision_flights for p in ex._all_paths(f)
                     if ex.describe_path(f, p).get("rerouted")}, []) if "reroute" in actions else None,
    }
    for lever, ban in without.items():
        if ban is None:
            continue
        alt, _ = ex.solver.solve(ban_paths=ban[0], ban_configs=ban[1])
        if not alt.satisfiable:
            levers[lever] = {"needed": True, "level": "no answer without it", "margin": None}
        else:
            level = compare(ex.factual, alt)
            levers[lever] = {"needed": level is not None, "level": level,
                             "margin": (alt.costs[level] - ex.factual.costs[level]) if level else 0}

    obj = record["objectives"]
    return {
        "iteration": record["iteration"], "accepted": record["accepted"],
        "hotspot": {k: record["hotspot"].get(k) for k in ("sector", "time", "demand", "capacity")},
        "action": "+".join(sorted(actions)) or "nothing",
        "flights": flights,
        # always "overload" for an accepted step (acceptance requires less overload); kept as a check
        "deciding_level_vs_nothing": deciding if foil.satisfiable else ("keeping impossible" if not keep_possible else "infeasible"),
        "levers": levers,
        "margin": margin,
        "tie": ex.tie_check()["tie"],
        "overload_removed": (obj.get("OVERLOAD-BEFORE") or 0) - (obj.get("OVERLOAD") or 0),
        "objectives": {k: obj.get(k) for k in ("OVERLOAD", "ARRIVAL-DELAY", "SECTOR-NUMBER", "REROUTE")},
    }


def delay_provenance(trace: TraceReader) -> Dict[int, List[Dict[str, Any]]]:
    """flight -> the accepted steps that moved its arrival (time shift and the step's hotspot)."""
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in trace:
        if not record["accepted"]:
            continue
        for fid, change in (record.get("flight_changes") or {}).items():
            old = {int(t): n for t, n in change["old_flight"].items()}
            new = {int(t): n for t, n in change["new_flight"].items()}
            if not old or not new:
                continue
            out[int(fid)].append({"iteration": record["iteration"], "arrival_shift": max(new) - max(old),
                                  "hotspot": (record["hotspot"]["sector"], record["hotspot"]["time"])})
    return dict(out)


def summarize(trace: TraceReader) -> Dict[str, Any]:
    steps = [step_summary(IterationExplainer(trace, n)) for n in sorted(trace.iterations)]
    accepted = [s for s in steps if s["accepted"]]
    provenance = delay_provenance(trace)
    by_hotspot = Counter()
    for moves in provenance.values():
        for m in moves:
            by_hotspot[m["hotspot"]] += m["arrival_shift"]
    return {
        "steps": steps,
        "actions": dict(Counter(s["action"] for s in accepted)),
        "deciding_levels": {name: sum(1 for s in accepted if s["deciding_level_vs_nothing"] == name) for _, name in LEVELS},
        "levers": {lever: {"used": sum(1 for s in accepted if lever in s["levers"]),
                           "needed": sum(1 for s in accepted if s["levers"].get(lever, {}).get("needed")),
                           "decided_by": dict(Counter(s["levers"][lever]["level"] for s in accepted
                                                      if s["levers"].get(lever, {}).get("needed")))}
                   for lever in ("delay", "reroute", "split")},
        "tied_steps": sum(1 for s in accepted if s["tie"]),
        "accepted_steps": len(accepted), "rejected_steps": len(steps) - len(accepted),
        "delay_by_hotspot": [{"sector": s, "time": t, "arrival_shift": v} for (s, t), v in by_hotspot.most_common()],
        "flights_moved": len(provenance),
        "flights_moved_more_than_once": sum(1 for m in provenance.values() if len(m) > 1),
    }
