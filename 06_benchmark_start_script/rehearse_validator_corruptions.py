#!/usr/bin/env python3
"""Show that validate_solutions.py catches deliberately broken solutions.

Takes ONE instance of a real problem folder whose runs validate as VALID, copies it once per
corruption into <work-dir>/<case>/, breaks the copy in exactly one way, runs validate_solutions.py
on the copy, and checks that the verdict is the expected one:

    case                    what is broken                                         expected
    baseline                nothing                                                VALID
    edge_break              an en-route waypoint replaced by a non-adjacent vertex INVALID not_an_edge
    edge_time               a waypoint reached one timestep late                   INVALID edge_time
    departs_early           a flight shifted one timestep before its filed start   INVALID departs_before_filed
    endpoint                a flight lands at another airport                      INVALID endpoints
    flight_missing          a flight's row emptied                                 INVALID flight_missing
    flown_twice             a flight's trajectory repeated after it landed         INVALID continues_after_destination
    rotation_overlap        a leg delayed until after the next leg departs         INVALID rotation_overlap
    rotation_no_turnaround  a leg delayed until the next leg's departure timestep  INVALID rotation_no_turnaround
    truncated_file          converted_navpoint_matrix.csv.gz cut in half           INVALID matrix_unreadable
    row_dropped             one row removed from converted_instance_matrix         INVALID matrix_unreadable (shape vs manifest)
    claim_delay             ARRIVAL-DELAY on the last result line + 1             MISMATCH claim:ARRIVAL-DELAY
    claim_overload          OVERLOAD on the last result line + 2                  MISMATCH claim:OVERLOAD
    hidden_overload         every flight back on its filed plan, claims kept       MISMATCH claim:OVERLOAD
    sector_row              one cell of converted_instance_matrix changed          MISMATCH sector_rows
    capacity_cell           one cell of capacity_time_matrix changed               MISMATCH capacity_matrix
    sector_representative   a sector's own vertex moved to a neighbouring sector   INVALID sector_representative
    airport_merged          an en-route vertex put into an airport's sector        INVALID airport_not_atomic
    sector_disconnected     an en-route vertex put into a non-adjacent sector      INVALID sector_disconnected

The flight-level cases break the --static-system run (a system that never reconfigures, so a
shifted trajectory keeps its sector sequence); the allocation cases break the --dynamic-system run.
Flights, legs and vertices are picked from the data, so this runs on any problem folder -- the
laptop rehearsal and a real campaign folder alike. Nothing outside <work-dir> is written.

    ./rehearse_validator_corruptions.py --problem-dir output/20260918_V2/output_<PROBLEM> \\
        --instance-root ../05_instances --work-dir /tmp/corrupt [--instance NAME]

Exit 0 when every case got its expected verdict, 1 otherwise.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_solutions as vs                                       # noqa: E402

MATS = ("converted_navpoint_matrix", "converted_instance_matrix",
        "navaid_sector_time_assignment", "capacity_time_matrix")


def read(run_dir: Path, name: str) -> np.ndarray:
    path = vs.find_matrix_file(run_dir, name)
    return vs.read_matrix(path).astype(np.int64)


def write(run_dir: Path, name: str, array: np.ndarray) -> None:
    """Write as the solvers do (np.savetxt %d), in the format that is already there."""
    path = vs.find_matrix_file(run_dir, name)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            np.savetxt(fh, array, fmt="%d", delimiter=",")
    elif path.suffix == ".npz":
        np.savez_compressed(path, **{name: array})
    else:
        np.savetxt(path, array, fmt="%d", delimiter=",")


class Case:
    def __init__(self, name: str, system: str, expected_status: str, expected_check: str,
                 apply: Callable[[Path, Path, vs.Instance], str]):
        self.name, self.system = name, system
        self.expected_status, self.expected_check = expected_status, expected_check
        self.apply = apply


# ---- helpers on trajectories ------------------------------------------------------------------

def trajectory(nav: np.ndarray, f: int) -> List[Tuple[int, int]]:
    cols = np.flatnonzero(nav[f] != -1)
    return [(int(t), int(nav[f, t])) for t in cols]


def shift_flight(nav: np.ndarray, sec: np.ndarray, f: int, k: int) -> None:
    """Move flight f's whole row k timesteps (nav and sector rows alike)."""
    for m in (nav, sec):
        row = m[f].copy()
        cols = np.flatnonzero(row != -1)
        m[f, :] = -1
        m[f, cols + k] = row[cols]


def adjacency(inst: vs.Instance) -> Dict[int, set]:
    adj: Dict[int, set] = {v: set() for v in range(inst.n_vertices)}
    for a, b in zip(inst.edge_u, inst.edge_v):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


# ---- the corruptions ---------------------------------------------------------------------------

def c_nothing(run_dir, problem_dir, inst):
    return "unchanged"


def c_edge_break(run_dir, problem_dir, inst):
    nav, adj = read(run_dir, MATS[0]), adjacency(inst)
    for f in range(inst.n_flights):
        tr = trajectory(nav, f)
        if len(tr) < 3:
            continue
        (_, v0), (t1, v1), (_, v2) = tr[0], tr[1], tr[2]
        on_path = {v for _, v in tr}
        for x in range(inst.n_vertices):
            if (not inst.is_airport[x] and x not in on_path and x not in adj[v0]
                    and x not in adj[v2]):
                nav[f, t1] = x
                write(run_dir, MATS[0], nav)
                return f"flight {f}: waypoint {v1}@{t1} -> {x} (no edge {v0}-{x})"
    raise RuntimeError("no flight with three waypoints")


def c_edge_time(run_dir, problem_dir, inst):
    nav = read(run_dir, MATS[0])
    for f in range(inst.n_flights):
        tr = trajectory(nav, f)
        if len(tr) >= 3 and tr[1][0] + 1 < tr[2][0]:
            t1, v1 = tr[1]
            nav[f, t1], nav[f, t1 + 1] = -1, v1
            write(run_dir, MATS[0], nav)
            return f"flight {f}: {v1} reached at {t1 + 1} instead of {t1}"
    # every edge takes one timestep (coarse T_gran): hold the flight one extra timestep on its
    # first edge instead, i.e. move everything after the departure one timestep later
    for f in range(inst.n_flights):
        tr = trajectory(nav, f)
        if len(tr) >= 2 and tr[-1][0] + 1 < nav.shape[1]:
            for t, _ in tr[1:]:
                nav[f, t] = -1
            for t, v in tr[1:]:
                nav[f, t + 1] = v
            write(run_dir, MATS[0], nav)
            return f"flight {f}: first edge {tr[0][1]}->{tr[1][1]} takes {tr[1][0] + 1 - tr[0][0]} timesteps"
    raise RuntimeError("no flight with a stretchable edge")


def c_departs_early(run_dir, problem_dir, inst):
    nav, sec = read(run_dir, MATS[0]), read(run_dir, MATS[1])
    prev_of = dict(zip(inst.rotation_next.tolist(), inst.rotation_prev.tolist()))
    for f in range(inst.n_flights):
        tr = trajectory(nav, f)
        if not tr or tr[0][0] != inst.filed_dep[f] or tr[0][0] < 1:
            continue
        p = prev_of.get(f)
        if p is not None and trajectory(nav, p)[-1][0] >= tr[0][0] - 2:
            continue
        shift_flight(nav, sec, f, -1)
        write(run_dir, MATS[0], nav)
        write(run_dir, MATS[1], sec)
        return f"flight {f} departs {tr[0][0] - 1}, filed {tr[0][0]}"
    raise RuntimeError("no undelayed flight that can be moved earlier")


def c_endpoint(run_dir, problem_dir, inst):
    nav = read(run_dir, MATS[0])
    for f in range(inst.n_flights):
        tr = trajectory(nav, f)
        t_last, v_last = tr[-1]
        other = [int(a) for a in inst.airports if a != v_last and a != tr[0][1]]
        if other:
            nav[f, t_last] = other[0]
            write(run_dir, MATS[0], nav)
            return f"flight {f} lands at {other[0]} instead of {v_last}"
    raise RuntimeError("fewer than three airports")


def c_flight_missing(run_dir, problem_dir, inst):
    nav, sec = read(run_dir, MATS[0]), read(run_dir, MATS[1])
    f = inst.n_flights // 2
    nav[f, :] = -1
    sec[f, :] = -1
    write(run_dir, MATS[0], nav)
    write(run_dir, MATS[1], sec)
    return f"flight {f} removed"


def c_flown_twice(run_dir, problem_dir, inst):
    nav = read(run_dir, MATS[0])
    for f in range(inst.n_flights):
        tr = trajectory(nav, f)
        span = tr[-1][0] - tr[0][0]
        start = tr[-1][0] + 2
        if start + span < nav.shape[1]:
            for t, v in tr:
                nav[f, t - tr[0][0] + start] = v
            write(run_dir, MATS[0], nav)
            return f"flight {f} flown again from t={start}"
    raise RuntimeError("no room to repeat a flight")


def _rotation(run_dir, inst, extra: int):
    nav, sec = read(run_dir, MATS[0]), read(run_dir, MATS[1])
    for p, n in zip(inst.rotation_prev, inst.rotation_next):
        tp, tn = trajectory(nav, int(p)), trajectory(nav, int(n))
        k = tn[0][0] - tp[-1][0] + extra          # lands at dep(next) + extra - 0
        if k > 0 and tp[-1][0] + k < nav.shape[1]:
            shift_flight(nav, sec, int(p), k)
            write(run_dir, MATS[0], nav)
            write(run_dir, MATS[1], sec)
            return (f"aircraft {int(inst.flight_aircraft[p])}: flight {int(p)} delayed {k}, lands "
                    f"{tp[-1][0] + k}; flight {int(n)} departs {tn[0][0]}")
    raise RuntimeError("no aircraft with two legs")


def c_rotation_overlap(run_dir, problem_dir, inst):
    return _rotation(run_dir, inst, 1)


def c_rotation_touch(run_dir, problem_dir, inst):
    return _rotation(run_dir, inst, 0)


def c_truncated(run_dir, problem_dir, inst):
    path = vs.find_matrix_file(run_dir, MATS[0])
    data = path.read_bytes()
    path.write_bytes(data[: max(10, len(data) // 2)])
    return f"{path.name}: {len(data)} -> {max(10, len(data) // 2)} bytes"


def c_row_dropped(run_dir, problem_dir, inst):
    sec = read(run_dir, MATS[1])
    write(run_dir, MATS[1], sec[:-1])
    return f"{MATS[1]}: {sec.shape[0]} -> {sec.shape[0] - 1} rows"


def make_claim_case(metric: str, delta: int):
    def apply(run_dir, problem_dir, inst):
        path = problem_dir / "individual_outputs" / f"{run_dir.name}_{apply.system}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["object"][-1][metric] = int(data["object"][-1][metric]) + delta
        path.write_text(json.dumps(data), encoding="utf-8")
        return f"claimed {metric} {data['object'][-1][metric] - delta} -> {data['object'][-1][metric]}"
    return apply


def c_hidden_overload(run_dir, problem_dir, inst):
    nav, sec, alloc = read(run_dir, MATS[0]), read(run_dir, MATS[1]), read(run_dir, MATS[2])
    width = nav.shape[1]
    nav[:, :] = -1
    sec[:, :] = -1
    f, t, v = inst.filed_f, inst.filed_t, inst.filed_v
    nav[f, t] = v
    # sector rows under the half rule
    for flight in range(inst.n_flights):
        tr = trajectory(nav, flight)
        sec[flight, tr[0][0]] = alloc[tr[0][1], tr[0][0]]
        for (ta, va), (tb, vb) in zip(tr, tr[1:]):
            half = (tb - ta) // 2
            for tt in range(ta + 1, tb + 1):
                sec[flight, tt] = alloc[va if tt - ta <= half else vb, min(tt, width - 1)]
    write(run_dir, MATS[0], nav)
    write(run_dir, MATS[1], sec)
    return "all flights on their filed plans"


def c_sector_row(run_dir, problem_dir, inst):
    sec = read(run_dir, MATS[1])
    rows, cols = np.nonzero(sec != -1)
    f, t = int(rows[len(rows) // 2]), int(cols[len(rows) // 2])
    old = int(sec[f, t])
    new = next(int(s) for s in inst.static_allocation if int(s) != old)
    sec[f, t] = new
    write(run_dir, MATS[1], sec)
    return f"flight {f} t={t}: sector {old} -> {new}"


def c_capacity_cell(run_dir, problem_dir, inst):
    cap = read(run_dir, MATS[3])
    s, t = int(inst.static_allocation[~inst.is_airport][0]), cap.shape[1] // 2
    cap[s, t] += 1
    write(run_dir, MATS[3], cap)
    return f"capacity of sector {s} at t={t} + 1"


def _corruption_column(alloc: np.ndarray) -> int:
    return min(5, alloc.shape[1] - 1)


def c_representative(run_dir, problem_dir, inst):
    alloc, adj = read(run_dir, MATS[2]), adjacency(inst)
    t = _corruption_column(alloc)
    col = alloc[:, t]
    for s in np.unique(col):
        members = np.flatnonzero(col == s)
        if inst.is_airport[s] or len(members) < 2:
            continue
        for x in adj[int(s)]:
            if not inst.is_airport[x] and col[x] != s and col[col[x]] == col[x]:
                alloc[int(s), t] = col[x]
                write(run_dir, MATS[2], alloc)
                return f"t={t}: vertex {int(s)} moved into sector {int(col[x])}; sector {int(s)} keeps {len(members) - 1} members"
    raise RuntimeError("no multi-vertex en-route sector with an en-route neighbour")


def c_airport_merged(run_dir, problem_dir, inst):
    alloc, adj = read(run_dir, MATS[2]), adjacency(inst)
    t = _corruption_column(alloc)
    col = alloc[:, t]
    for v in range(inst.n_vertices):
        if inst.is_airport[v] or col[v] == v:
            continue
        airports = [a for a in adj[v] if inst.is_airport[a]] or [int(inst.airports[0])]
        alloc[v, t] = airports[0]
        write(run_dir, MATS[2], alloc)
        return f"t={t}: en-route vertex {v} put into airport sector {airports[0]}"
    raise RuntimeError("no non-representative en-route vertex")


def c_sector_disconnected(run_dir, problem_dir, inst):
    alloc, adj = read(run_dir, MATS[2]), adjacency(inst)
    t = _corruption_column(alloc)
    col = alloc[:, t]
    enroute_sectors = [int(s) for s in np.unique(col) if not inst.is_airport[s]]
    for v in range(inst.n_vertices):
        if inst.is_airport[v] or col[v] == v:
            continue
        for s in enroute_sectors:
            members = set(np.flatnonzero(col == s).tolist())
            if s != col[v] and not (adj[v] & members):
                alloc[v, t] = s
                write(run_dir, MATS[2], alloc)
                return f"t={t}: vertex {v} put into sector {s}, which it does not touch"
    raise RuntimeError("no vertex/sector pair without an edge")


# ---- driver ------------------------------------------------------------------------------------

def build_cases(static_system: str, dynamic_system: str) -> List[Case]:
    S, D = static_system, dynamic_system
    claim_delay = make_claim_case("ARRIVAL-DELAY", 1)
    claim_overload = make_claim_case("OVERLOAD", 2)
    claim_delay.system = S
    claim_overload.system = S
    return [
        Case("baseline_static", S, vs.VALID, "", c_nothing),
        Case("baseline_dynamic", D, vs.VALID, "", c_nothing),
        Case("edge_break", S, vs.INVALID, "not_an_edge", c_edge_break),
        Case("edge_time", S, vs.INVALID, "edge_time", c_edge_time),
        Case("departs_early", S, vs.INVALID, "departs_before_filed", c_departs_early),
        Case("endpoint", S, vs.INVALID, "endpoints", c_endpoint),
        Case("flight_missing", S, vs.INVALID, "flight_missing", c_flight_missing),
        Case("flown_twice", S, vs.INVALID, "continues_after_destination", c_flown_twice),
        Case("rotation_overlap", S, vs.INVALID, "rotation_overlap", c_rotation_overlap),
        Case("rotation_no_turnaround", S, vs.INVALID, "rotation_no_turnaround", c_rotation_touch),
        Case("truncated_file", S, vs.INVALID, "matrix_unreadable", c_truncated),
        Case("row_dropped", S, vs.INVALID, "matrix_unreadable", c_row_dropped),
        Case("claim_delay", S, vs.MISMATCH, "claim:ARRIVAL-DELAY", claim_delay),
        Case("claim_overload", S, vs.MISMATCH, "claim:OVERLOAD", claim_overload),
        Case("hidden_overload", S, vs.MISMATCH, "claim:OVERLOAD", c_hidden_overload),
        Case("sector_row", D, vs.MISMATCH, "sector_rows", c_sector_row),
        Case("capacity_cell", D, vs.MISMATCH, "capacity_matrix", c_capacity_cell),
        Case("sector_representative", D, vs.INVALID, "sector_representative", c_representative),
        Case("airport_merged", D, vs.INVALID, "airport_not_atomic", c_airport_merged),
        Case("sector_disconnected", D, vs.INVALID, "sector_disconnected", c_sector_disconnected),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem-dir", type=Path, required=True)
    ap.add_argument("--instance-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--instance", default=None, help="default: the first with both systems' matrices")
    ap.add_argument("--static-system", default="02_RerouteDelay")
    ap.add_argument("--dynamic-system", default="01_ASPaeroFlow")
    ap.add_argument("--only", default=None, help="comma-separated case names")
    a = ap.parse_args()

    problem_dir = a.problem_dir.resolve()
    folders = vs.Folders()
    problem = problem_dir.name[len("output_"):]
    tg = vs.problem_granularity(problem, problem_dir, a.instance_root, None)

    def run_dir_of(base: Path, system: str, instance: str) -> Path:
        return base / "solver_outputs" / folders.folder(system) / instance

    instance = a.instance
    if instance is None:
        for cand in sorted(p.name for p in run_dir_of(problem_dir, a.static_system, "x").parent.iterdir()):
            if run_dir_of(problem_dir, a.dynamic_system, cand).is_dir():
                instance = cand
                break
    if instance is None:
        print("[ERROR] no instance with matrices of both systems", file=sys.stderr)
        return 2
    inst = vs.Instance(a.instance_root / problem / instance, tg)
    print(f"problem {problem}  instance {instance}  T_gran {tg}  flights {inst.n_flights}")

    cases = build_cases(a.static_system, a.dynamic_system)
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c.name in keep]

    results = []
    for case in cases:
        base = a.work_dir / case.name / problem_dir.parent.name / problem_dir.name
        if base.exists():
            shutil.rmtree(base)
        (base / "individual_outputs").mkdir(parents=True)
        src_json = problem_dir / "individual_outputs" / f"{instance}_{case.system}.json"
        shutil.copy2(src_json, base / "individual_outputs" / src_json.name)
        src_run = run_dir_of(problem_dir, case.system, instance)
        dst_run = run_dir_of(base, case.system, instance)
        shutil.copytree(src_run, dst_run)
        try:
            what = case.apply(dst_run, base, inst)
        except RuntimeError as exc:
            results.append((case, "SKIPPED", str(exc), False))
            continue
        out = a.work_dir / case.name / "validation"
        if out.exists():
            shutil.rmtree(out)
        proc = subprocess.run([sys.executable, str(Path(vs.__file__).resolve()), "--problem-dir", str(base),
                               "--instance-root", str(a.instance_root), "--out-dir", str(out),
                               "--systems", case.system, "--timestep-granularity", str(tg)],
                              capture_output=True, text=True)
        rows = vs.read_rows(out.glob("*/validation_*.csv"))
        row = next((r for r in rows if r["system"] == case.system), None)
        if row is None:
            results.append((case, "NO ROW", proc.stdout[-300:] + proc.stderr[-300:], False))
            continue
        checks = row["failed_checks"].split(";") if row["failed_checks"] else []
        ok = row["status"] == case.expected_status and (not case.expected_check or case.expected_check in checks)
        results.append((case, f"{row['status']} (exit {proc.returncode})",
                        f"{what}  ->  {row['failed_checks'] or '-'}"
                        + (f"  [{row['first_violation'][:110]}]" if row.get("first_violation") else ""), ok))

    width = max(len(c.name) for c, *_ in results)
    print()
    for case, got, detail, ok in results:
        expect = case.expected_status + (f" {case.expected_check}" if case.expected_check else "")
        print(f"{'CAUGHT' if ok else 'FAILED':<7} {case.name:<{width}}  {case.system:<16} expected {expect:<40} got {got}")
        print(f"        {'':<{width}}  {detail}")
    n_ok = sum(1 for *_, ok in results if ok)
    print(f"\n{n_ok} of {len(results)} cases got their expected verdict")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
