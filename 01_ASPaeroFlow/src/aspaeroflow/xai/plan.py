"""The plan of a recorded run as data (K01): the filed plan, the final plan, what moved each flight, the steps and
the run's key figures, written as plan.json and as the same content in ASP facts (plan.lp).

    python -m src.aspaeroflow.xai.plan <trace folder> [--data-dir DIR] [--out DIR]   (from 01_ASPaeroFlow)

The final plan is the filed plan (flights.csv) with every accepted step's new trajectories applied in step order.
Times are step numbers of the day; one step lasts 60 / timestep_granularity minutes (the edge cost uses
3600 / granularity seconds per step). The checks are reported, never repaired: a failed check means the run or its
reconstruction violates the model.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..arrival_delay_bootstrap import DEFAULT_ARRIVAL_DELAY_METRIC, apply
from .trace import TraceReader

Trajectory = Dict[int, int]
MAX_LISTED = 20


def _rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def resolve_data_dir(folder: Path, run: Dict[str, Any], data_dir: Optional[Path] = None) -> Path:
    """The instance folder: the argument; else run.json's data_dir if absolute; else a folder of that name inside or
    next to the trace folder, or the path relative to the trace folder; the working directory only last (a recorded
    relative path is relative to wherever the run was started)."""
    candidates: List[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    recorded = run.get("data_dir")
    if recorded:
        p = Path(recorded)
        candidates += [p] if p.is_absolute() else [folder / p.name, folder.parent / p.name, folder / p, p]
    for c in candidates:
        if (c / "flights.csv").is_file():
            return c
    raise FileNotFoundError(f"no instance with flights.csv for trace {folder} (tried {[str(c) for c in candidates]})")


def route_of(trajectory: Trajectory) -> List[int]:
    """Navpoints in time order, consecutive repeats removed."""
    out: List[int] = []
    for t in sorted(trajectory):
        if not out or out[-1] != trajectory[t]:
            out.append(trajectory[t])
    return out


def clock(step: int, minutes_per_step: float, steps_per_day: int) -> str:
    day, rest = divmod(step, steps_per_day)
    minutes = int(round(rest * minutes_per_step))
    text = f"{minutes // 60:02d}:{minutes % 60:02d}"
    return f"+{day}d {text}" if day else text


def _ints(d: Dict[Any, Any]) -> Trajectory:
    return {int(t): int(n) for t, n in d.items()}


def _check(name: str, violations: List[Any]) -> Dict[str, Any]:
    return {"name": name, "ok": not violations, "count": len(violations), "violations": violations[:MAX_LISTED]}


def build_plan(folder: Path, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    return _build(folder, data_dir)[0]


def _build(folder: Path, data_dir: Optional[Path]) -> Tuple[Dict[str, Any], Dict[int, Trajectory], Dict[int, Trajectory]]:
    """The plan and the filed and final trajectories (those go to plan.lp only: in plan.json they would be large)."""
    folder = Path(folder)
    trace = TraceReader(folder)
    run = json.loads((folder / "run.json").read_text())
    instance = resolve_data_dir(folder, run, data_dir)
    granularity = int(run.get("timestep_granularity") or 1)
    mps = 60 / granularity
    per_day = 24 * granularity
    metric = run.get("arrival_delay_metric") or DEFAULT_ARRIVAL_DELAY_METRIC

    filed: Dict[int, Trajectory] = defaultdict(dict)
    for r in _rows(instance / "flights.csv"):
        filed[int(float(r["Flight_ID"]))][int(float(r["Time"]))] = int(float(r["Position"]))
    aircraft: Dict[int, int] = {}
    if (instance / "airplane_flight_assignment.csv").is_file():
        aircraft = {int(float(r["Flight_ID"])): int(float(r["Airplane_ID"]))
                    for r in _rows(instance / "airplane_flight_assignment.csv")}

    current: Dict[int, Trajectory] = {f: dict(t) for f, t in filed.items()}
    moved: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    steps: List[Dict[str, Any]] = []
    chain: List[Dict[str, Any]] = []
    last_objectives = run.get("initial_objectives") or {}
    for record in trace:
        if not record.get("accepted"):
            continue
        it = record["iteration"]
        hotspot = record.get("hotspot") or {}
        in_hotspot = {int(f["id"]) for f in hotspot.get("flights") or []} if hotspot.get("flights") is not None else None
        changes = record.get("flight_changes") or {}
        for fid in sorted(changes, key=int):
            f = int(fid)
            old, new = _ints(changes[fid]["old_flight"]), _ints(changes[fid]["new_flight"])
            if old != current.get(f, {}):
                chain.append({"iteration": it, "flight": f})
            if old and new:
                moved[f].append({"iteration": it, "departure_shift": min(new) - min(old),
                                 "arrival_shift": max(new) - max(old), "rerouted": route_of(new) != route_of(old),
                                 "cause": None if in_hotspot is None else ("hotspot" if f in in_hotspot else "rotation"),
                                 "hotspot": {"sector": hotspot.get("sector"), "time": hotspot.get("time")}})
            current[f] = new
        sc = record.get("sector_changes") or {}
        split = None
        if sc and sc.get("post_sector_config") is not None:
            split = {"sector": hotspot.get("sector"), "time": hotspot.get("time"),
                     "parts": len(sc.get("post_sector_config") or {})}
        steps.append({"iteration": it, "hotspot": {k: hotspot.get(k) for k in ("sector", "time", "demand", "capacity",
                                                                                 "overload")}
                      | {"clock": clock(hotspot["time"], mps, per_day) if hotspot.get("time") is not None else None},
                      "flights_moved": sorted(int(f) for f in changes), "sector_split": split,
                      "objectives": record.get("objectives") or {}})
        last_objectives = record.get("objectives") or last_objectives

    def times(t: Trajectory) -> Dict[str, Any]:
        dep, arr = min(t), max(t)
        return {"departure": dep, "arrival": arr, "departure_clock": clock(dep, mps, per_day),
                "arrival_clock": clock(arr, mps, per_day), "route": route_of(t)}

    flights: List[Dict[str, Any]] = []
    for f in sorted(filed):
        a, b = filed[f], current[f]
        dep_delay, arr_delay = min(b) - min(a), int(apply(max(b) - max(a), metric))
        extra = (max(b) - min(b)) - (max(a) - min(a))
        mv = moved.get(f, [])
        largest = max(mv, key=lambda m: (m["arrival_shift"], -m["iteration"]))["iteration"] if mv else None
        flights.append({"flight": f, "aircraft": aircraft.get(f), "origin": a[min(a)], "destination": a[max(a)],
                        "filed": times(a), "final": times(b),
                        "departure_delay": {"steps": dep_delay, "minutes": dep_delay * mps},
                        "arrival_delay": {"steps": arr_delay, "minutes": arr_delay * mps},
                        "rerouted": route_of(a) != route_of(b), "changed": a != b,
                        "extra_flight_time": {"steps": extra, "minutes": extra * mps},
                        "moved_by": mv, "largest_hotspot": largest})

    # rotation continuity: per aircraft, legs in filed departure order; the next leg departs from the airport the
    # previous one arrived at, strictly after that arrival (the optimizer couples legs by time only, strictly)
    legs: Dict[int, List[int]] = defaultdict(list)
    for f in sorted(filed, key=lambda f: (min(filed[f]), f)):
        if f in aircraft:
            legs[aircraft[f]].append(f)
    turnarounds, rotation = [], []
    for ac, fs in sorted(legs.items()):
        for p, n in zip(fs, fs[1:]):
            P, N = current[p], current[n]
            gap = min(N) - max(P)
            turnarounds.append({"aircraft": ac, "previous": p, "next": n, "gap": gap})
            if P[max(P)] != N[min(N)] or gap <= 0:
                rotation.append({"aircraft": ac, "previous": p, "next": n, "arrives_at": P[max(P)],
                                 "departs_from": N[min(N)], "gap": gap})

    by = {x["flight"]: x for x in flights}
    earlier = [f for f in by if by[f]["departure_delay"]["steps"] < 0]
    delays = sorted(flights, key=lambda x: (-x["arrival_delay"]["steps"], x["flight"]))
    obj0 = run.get("initial_objectives") or {}
    kpi_names = {"overload": "OVERLOAD", "arrival_delay": "ARRIVAL-DELAY", "sectors": "SECTOR-NUMBER",
                 "changed_flights": "REROUTE"}
    kpis = {name: {"filed": obj0.get(key), "final": last_objectives.get(key)} for name, key in kpi_names.items()}
    total_delay = sum(x["arrival_delay"]["steps"] for x in flights)
    kpis |= {"flights_delayed": sum(1 for x in flights if x["arrival_delay"]["steps"] > 0),
             "flights_rerouted": sum(1 for x in flights if x["rerouted"]),
             "total_arrival_delay_minutes": total_delay * mps,
             "largest_arrival_delay_minutes": max((x["arrival_delay"]["steps"] for x in flights), default=0) * mps,
             "largest_delays": [x["flight"] for x in delays[:5] if x["arrival_delay"]["steps"] > 0]}
    changed = sum(1 for x in flights if x["changed"])
    kpi_check = []
    if last_objectives.get("ARRIVAL-DELAY") is not None and last_objectives["ARRIVAL-DELAY"] != total_delay:
        kpi_check.append({"kpi": "ARRIVAL-DELAY", "recorded": last_objectives["ARRIVAL-DELAY"], "plan": total_delay})
    if last_objectives.get("REROUTE") is not None and last_objectives["REROUTE"] != changed:
        kpi_check.append({"kpi": "REROUTE", "recorded": last_objectives["REROUTE"], "plan": changed})

    plan = {"run": {"trace": str(folder), "instance": str(instance), "timestep_granularity": granularity,
                    "minutes_per_step": mps, "steps_per_day": per_day, "arrival_delay_metric": metric,
                    "flight_duration_rule": run.get("flight_duration_rule"), "number_flights": len(filed),
                    "kept_steps": len(steps)},
            "kpis": kpis, "flights": flights, "steps": steps, "turnarounds": turnarounds,
            "checks": [_check("chain", chain), _check("rotation", rotation),
                       _check("no_earlier_departure", earlier), _check("kpis", kpi_check)]}
    return plan, dict(filed), current


PLAN_LP_HEADER = """\
% Plan of a recorded ASPaeroFlow run (xai/plan.py, K01). Times are step numbers; one step lasts M minutes.
% plan_minutes_per_step(M).
% plan_flight(F,Aircraft,Origin,Destination).        aircraft 'none' when the instance has no assignment
% plan_filed(F,Departure,Arrival). plan_final(F,Departure,Arrival).
% plan_filed_at(F,T,N). plan_final_at(F,T,N).         flight F is at navpoint N at step T
% plan_departure_delay(F,D). plan_arrival_delay(F,D).  D in steps (arrival: the run's delay metric); only D != 0
% plan_rerouted(F).                                    the final route (navpoints in order) differs from the filed one
% plan_changed(F).                                     any change of the trajectory (the optimizer's REROUTE counts these)
% plan_moved(F,I,DepartureShift,ArrivalShift).        step I moved flight F
% plan_moved_cause(F,I,Cause).                         hotspot (F was a flight of the hotspot) or rotation
% plan_largest_hotspot(F,I).                           the step that added most to F's arrival delay
% plan_step(I,Sector,Time,Demand,Capacity,Overload).   kept step I and its hotspot
% plan_sector_split(I,Sector,Time,Parts).
% plan_turnaround(Previous,Next,Gap).                 consecutive legs of one aircraft, gap in steps (final plan)
% plan_kpi(Name,Filed,Final).                          overload, arrival_delay, sectors, changed_flights
% plan_check_ok(Name). plan_check_violations(Name,N).  chain, rotation, no_earlier_departure, kpis
"""


def plan_facts(plan: Dict[str, Any], filed: Optional[Dict[int, Trajectory]] = None,
               final: Optional[Dict[int, Trajectory]] = None) -> str:
    out = [PLAN_LP_HEADER, f"plan_minutes_per_step({int(plan['run']['minutes_per_step'])})."]
    flights = plan["flights"]
    for x in flights:
        f = x["flight"]
        ac = x["aircraft"] if x["aircraft"] is not None else "none"
        out.append(f"plan_flight({f},{ac},{x['origin']},{x['destination']}).")
        out.append(f"plan_filed({f},{x['filed']['departure']},{x['filed']['arrival']}).")
        out.append(f"plan_final({f},{x['final']['departure']},{x['final']['arrival']}).")
        for name, key in (("plan_departure_delay", "departure_delay"), ("plan_arrival_delay", "arrival_delay")):
            if x[key]["steps"]:
                out.append(f"{name}({f},{x[key]['steps']}).")
        if x["rerouted"]:
            out.append(f"plan_rerouted({f}).")
        if x["changed"]:
            out.append(f"plan_changed({f}).")
        for m in x["moved_by"]:
            out.append(f"plan_moved({f},{m['iteration']},{m['departure_shift']},{m['arrival_shift']}).")
            if m["cause"]:
                out.append(f"plan_moved_cause({f},{m['iteration']},{m['cause']}).")
        if x["largest_hotspot"] is not None:
            out.append(f"plan_largest_hotspot({f},{x['largest_hotspot']}).")
    for name, traj in (("plan_filed_at", filed or {}), ("plan_final_at", final or {})):
        for f in sorted(traj):
            out += [f"{name}({f},{t},{n})." for t, n in sorted(traj[f].items())]
    return "\n".join(out + _step_facts(plan)) + "\n"


def _step_facts(plan: Dict[str, Any]) -> List[str]:
    out = []
    for s in plan["steps"]:
        h = s["hotspot"]
        vals = [h.get(k) for k in ("sector", "time", "demand", "capacity", "overload")]
        if all(v is not None for v in vals):
            out.append(f"plan_step({s['iteration']},{','.join(map(str, vals))}).")
        if s["sector_split"]:
            sp = s["sector_split"]
            out.append(f"plan_sector_split({s['iteration']},{sp['sector']},{sp['time']},{sp['parts']}).")
    for t in plan["turnarounds"]:
        out.append(f"plan_turnaround({t['previous']},{t['next']},{t['gap']}).")
    for name, k in plan["kpis"].items():
        if isinstance(k, dict) and k["filed"] is not None and k["final"] is not None:
            out.append(f"plan_kpi({name},{k['filed']},{k['final']}).")
    for c in plan["checks"]:
        out.append(f"plan_check_ok({c['name']})." if c["ok"] else f"plan_check_violations({c['name']},{c['count']}).")
    return out


def write_plan(folder: Path, data_dir: Optional[Path] = None, out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    folder = Path(folder)
    plan, filed, final = _build(folder, data_dir)
    out = Path(out_dir) if out_dir else folder
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "plan.lp").write_text(plan_facts(plan, filed, final))
    return out / "plan.json", out / "plan.lp"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    j, lp = write_plan(Path(a.trace), Path(a.data_dir) if a.data_dir else None, Path(a.out) if a.out else None)
    plan = json.loads(j.read_text())
    bad = [c["name"] for c in plan["checks"] if not c["ok"]]
    print(f"wrote {j} and {lp}; checks: {'all ok' if not bad else 'FAILED ' + ', '.join(bad)}")


if __name__ == "__main__":
    main()
