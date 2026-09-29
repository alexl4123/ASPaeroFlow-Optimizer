#!/usr/bin/env python3
"""Independent validator for the solution matrices of a benchmark campaign.

For every run of the selected systems in one merged problem folder

    output/<FOLDER>/output_<PROBLEM>/
        individual_outputs/<INSTANCE>_<SYSTEM>.json   every JSON line the solver printed
        solver_outputs/<RESULTS-DIR>/<INSTANCE>/      the run's result matrices

this script loads the instance (<instance-root>/<PROBLEM>/<INSTANCE>/) and the matrices, checks
the hard constraints of the joint ATFCM model on the solution, recomputes every objective the
solver claimed in its last JSON line, and writes one CSV row per run.

It does NOT import any solver code to evaluate a solution. The instance files are parsed here,
and the edge traversal time, the sector-occupancy rule, the composite sector capacity, the
evaluation window and all six objectives are implemented here, from the definitions in
Beiser et al. (LPNMR 2026, Def. 2-7; ATMOS 2026, Def. 1-13) and in the solvers' reporting code
(01_ASPaeroFlow/src/aspaeroflow/main_loop_components/after_optimization.py, 04_MIP/main.py).
The only import from this repository is start_benchmark_caller.build_system_config, to learn
which solver_outputs/ folder each system writes to.

WHAT A MATRIX IS (01_ASPaeroFlow/main.py _save_results, 04_MIP/main.py, 02_ASP/main.py)

    converted_navpoint_matrix      |F| x |T|  vertex reached by flight f at timestep t, else -1
    converted_instance_matrix      |F| x |T|  sector flight f occupies at t, else -1
    navaid_sector_time_assignment  |V| x |T|  sector (= representative vertex) of v at t
    capacity_time_matrix           |V| x |T|  composite capacity of sector s at t (01, 04 only)

HARD CONSTRAINTS (a violation makes the run INVALID)

    flight_missing / extra_flight_rows   every flight flown exactly once, no phantom rows
    bad_vertex_id / bad_sector_id        ids inside the instance
    endpoints                            starts at the filed origin, ends at the filed destination
    continues_after_destination          the destination is reached once, at the end
    departs_before_filed                 no departure before the filed departure
    not_an_edge                          consecutive waypoints are joined by a graph edge
                                         (a stay at the same AIRPORT is a ground wait and allowed)
    edge_time                            t_{j+1} - t_j = max(1, ceil(d / (v * 0.51444) / (3600/T_gran)))
    not_simple                           no en-route vertex visited twice
    rotation_place                       a leg departs from the airport where the aircraft's
                                         previous leg landed
    rotation_overlap                     a leg departs before the aircraft's previous leg landed
    rotation_no_turnaround               a leg departs in the timestep its previous leg landed
                                         (the model requires dep >= arr + 1)
    sector_representative                a used sector id i contains its own vertex i
    airport_not_atomic                   every airport is a singleton sector
    sector_disconnected                  en-route sectors are connected (ATMOS Def. 11; LPNMR
                                         does not require it -- see --no-connectivity)
    matrix_unreadable / matrix_shape     files readable, shapes consistent (and as manifest.json says)
    granularity_mismatch                 the run was given another T_gran than its problem has

CONSISTENCY (a failure makes the run a MISMATCH)

    claimed != recomputed for OVERLOAD, ARRIVAL-DELAY, SECTOR-NUMBER, SECTOR-DIFF, REROUTE, RECONFIG
    sector_rows        converted_instance_matrix differs from the occupancy implied by the
                       trajectory and the sector allocation
    capacity_matrix    capacity_time_matrix differs from the capacity implied by the allocation

OBJECTIVES, as the solvers compute them (W = the instance's evaluation window)

    OVERLOAD       sum_{s,t} max(0, load(s,t) - cap(s,t)) over the matrix's whole time axis;
                   load from the trajectory under the half rule, cap(s,t) = max of the member
                   vertices' sectors.csv capacities
    ARRIVAL-DELAY  sum_f metric(t_arr(f) - t_arr_filed(f)), metric signed | floored | absolute
    SECTOR-NUMBER  sum over the W window columns of the number of distinct sectors
    SECTOR-DIFF    number of (v, t) with sec(v, t) != sec(v, t-1), on the matrix's own axis
    REROUTE        number of flights whose navpoint row in the first W columns differs from the
                   filed one -- i.e. delayed AND/OR rerouted (the papers' K_r)
    RECONFIG       number of (v, t), t < W, with sec(v, t) != the instance's allocation

STATUSES AND EXIT CODES

    VALID       every check passed, every claimed objective reproduced          exit 0
    NO_MATRIX   no matrix for this run. Expected for TIMEOUT/MEMOUT/ERROR runs  exit 0
                (the solvers write matrices only on a normal exit). A run that
                finished without one is flagged finished_run_without_matrix     exit 1
                A run whose matrix another system overwrote in a shared folder
                (03A_CASA writes to solver_outputs/03_DELAY) is flagged
                matrix_overwritten_by:<system>; counted in the summary          exit 0
    UNVERIFIED  matrix present but lacks the sector allocation (02_ASP with
                dynamic sectorisation does not write it)                        exit 1
    MISMATCH    consistency failure (claim != recomputation)                    exit 1
    INVALID     hard-constraint violation                                       exit 2
    ERROR       the validator could not validate (instance missing or failing
                its own self-check, unexpected exception)                       exit 3

The process exits with the worst code over everything it validated, so a SLURM array shows the
problem folders that need a look directly in `sacct` (ExitCode 1:0, 2:0, 3:0).

USAGE

    ./validate_solutions.py --problem-dir output/20260918_V2/output_<PROBLEM> \\
        --instance-root ../05_instances --out-dir output/VALIDATION_20260918 [--resume]
    ./validate_solutions.py --summarize --out-dir output/VALIDATION_20260918
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

#: The systems whose results are published. --systems all validates whatever is present.
PUBLISHED_SYSTEMS: Tuple[str, ...] = (
    "01_ASPaeroFlow", "0B_Sector_NoReroute_NoDelay", "0C_Sector_NoReroute_Delay",
    "0D_Sector_Reroute_NoDelay", "02_RerouteDelay", "2A_Reroute", "03_DELAY", "03A_CASA",
    "04_MIP", "05_ASP_rp_dp_sp", "05_ASP_rp_d_sp",
)

#: The objectives on a solver's result line, in the order the solvers print them.
METRICS: Tuple[str, ...] = ("OVERLOAD", "ARRIVAL-DELAY", "SECTOR-NUMBER", "SECTOR-DIFF",
                            "REROUTE", "RECONFIG")

#: Metres per second in one knot, as the data generator and common/edge_cost.py use it.
KNOTS_TO_METRES_PER_SECOND = 0.51444

#: Hours in the instance day; every solver runs with --max-time 24.
MAX_TIME_HOURS = 24

ARRIVAL_DELAY_METRICS = ("signed", "floored", "absolute")

VALID, NO_MATRIX, UNVERIFIED, MISMATCH, INVALID, ERROR = (
    "VALID", "NO_MATRIX", "UNVERIFIED", "MISMATCH", "INVALID", "ERROR")

#: Hard constraints of the model. Any count > 0 makes a run INVALID.
HARD_CHECKS: Tuple[str, ...] = (
    "matrix_unreadable", "matrix_shape", "granularity_mismatch", "flight_missing",
    "extra_flight_rows", "bad_vertex_id",
    "endpoints", "continues_after_destination", "departs_before_filed", "not_an_edge",
    "edge_time", "not_simple", "rotation_place", "rotation_overlap", "rotation_no_turnaround",
    "bad_sector_id", "sector_representative", "airport_not_atomic", "sector_disconnected",
)

#: Internal consistency of the matrices. Any count > 0 makes a run a MISMATCH.
CONSISTENCY_CHECKS: Tuple[str, ...] = ("sector_rows", "capacity_matrix", "allocation_width")

#: Informational counts, never a failure.
INFO_FIELDS: Tuple[str, ...] = (
    "n_flights", "matrix_width", "window_W", "n_path_changed", "n_delayed_only",
    "n_airport_waits", "n_landing_after_day", "overload_from_saved_sector_rows",
    "paper_active_sectors", "paper_sector_changes", "paper_reconfigurations",
)

OUTCOMES = {"": "ok", "T": "TIMEOUT", "M": "MEMOUT", "E": "ERROR", "P": "UNPARSED"}

CSV_COLUMNS: List[str] = (
    ["folder", "problem", "instance", "system", "outcome", "status", "failed_checks", "note",
     "matrix_dir", "matrix_owner", "arrival_delay_metric", "timestep_granularity"]
    + [f"{kind}_{m}" for m in METRICS for kind in ("claimed", "recomputed")]
    + [f"v_{c}" for c in HARD_CHECKS]
    + [f"c_{c}" for c in CONSISTENCY_CHECKS]
    + list(INFO_FIELDS)
    + ["claimed_computation_finished", "first_violation", "seconds"]
)

MATRIX_NAMES = ("converted_navpoint_matrix", "converted_instance_matrix",
                "navaid_sector_time_assignment", "capacity_time_matrix")


class ValidatorError(Exception):
    """The validator cannot validate this unit (missing input, failed instance self-check)."""


class MatrixFormatError(Exception):
    """A matrix file exists but cannot be read as an integer matrix."""


# ---------------------------------------------------------------------------------------------
# Model primitives, implemented here
# ---------------------------------------------------------------------------------------------

def edge_timesteps(dist_m, speed_kts, timestep_granularity: int) -> np.ndarray:
    """Timesteps to traverse an edge: max(1, ceil((d / (v * 0.51444)) / (3600 / T_gran))).

    LPNMR Def. 4 (t_{i+1} - t_i = max(ceil(d/u), 1)) and JOAS Sec. 4.3.2 (w_v(e)), with the
    float operations in the generator's order so that a boundary case rounds the same way.
    `dist_m` is the unrounded distance of graph_edges.csv.
    """
    dist_m = np.asarray(dist_m, dtype=np.float64)
    speed_ms = np.asarray(speed_kts, dtype=np.float64) * KNOTS_TO_METRES_PER_SECOND
    slot_seconds = 3600.0 / float(timestep_granularity)
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.ceil((dist_m / speed_ms) / slot_seconds)
    steps = np.where(speed_ms > 0, steps, 1.0)
    return np.maximum(steps, 1.0).astype(np.int64)


def evaluation_window(max_filed_time: int, timestep_granularity: int,
                      max_time_hours: int = MAX_TIME_HOURS) -> int:
    """The timesteps SECTOR-NUMBER and RECONFIG are summed over.

    The width every solver gives its initial allocation: max(last filed timestep + 1,
    (24 + 1) * T_gran), rounded up to a multiple of T_gran.
    """
    tg = int(timestep_granularity)
    width = max(int(max_filed_time) + 1, (int(max_time_hours) + 1) * tg)
    if width % tg:
        width += tg - width % tg
    return width


def apply_arrival_delay_metric(delta: np.ndarray, metric: str) -> np.ndarray:
    """signed: delta; floored: max(0, delta); absolute: |delta| (common/arrival_delay.py)."""
    if metric == "signed":
        return delta
    if metric == "floored":
        return np.maximum(delta, 0)
    if metric == "absolute":
        return np.abs(delta)
    raise ValueError(f"unknown arrival-delay metric {metric!r}")


# ---------------------------------------------------------------------------------------------
# The instance
# ---------------------------------------------------------------------------------------------

def _read_table(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValidatorError(f"instance file missing: {path}")
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidatorError(f"{path}: columns {missing} missing (has {list(frame.columns)})")
    return frame


def _as_int(series: pd.Series, what: str) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        raise ValidatorError(f"{what}: non-integer values")
    return values.astype(np.int64)


class Instance:
    """One parsed instance: graph, capacities, allocation, aircraft and filed flights."""

    def __init__(self, path: Path, timestep_granularity: int):
        self.path = Path(path)
        self.tg = int(timestep_granularity)
        p = self.path

        sectors = _read_table(p / "sectors.csv", ["Sector_ID", "Capacity"])
        sector_ids = _as_int(sectors["Sector_ID"], "sectors.csv Sector_ID")
        self.n_vertices = len(sector_ids)
        if not np.array_equal(sector_ids, np.arange(self.n_vertices)):
            raise ValidatorError("sectors.csv: Sector_ID is not 0..|V|-1 in order "
                                 "(the solvers index capacities by row)")
        self.capacity = _as_int(sectors["Capacity"], "sectors.csv Capacity")
        N = self.n_vertices

        airports = _read_table(p / "airports.csv", ["Airport_Vertex"])
        self.airports = np.unique(_as_int(airports["Airport_Vertex"], "airports.csv"))
        if self.airports.size and (self.airports.min() < 0 or self.airports.max() >= N):
            raise ValidatorError("airports.csv: vertex id outside 0..|V|-1")
        self.is_airport = np.zeros(N, dtype=bool)
        self.is_airport[self.airports] = True

        edges = _read_table(p / "graph_edges.csv", ["source", "target", "dist_m"])
        u = _as_int(edges["source"], "graph_edges.csv source")
        v = _as_int(edges["target"], "graph_edges.csv target")
        dist = edges["dist_m"].to_numpy(dtype=np.float64)
        if u.size and (min(u.min(), v.min()) < 0 or max(u.max(), v.max()) >= N):
            raise ValidatorError("graph_edges.csv: vertex id outside 0..|V|-1")
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = lo * N + hi
        # An undirected graph: one distance per vertex pair. networkx keeps the LAST distance of a
        # repeated pair, so do the same (stable sort, take the last of each run).
        order = np.argsort(keys, kind="stable")
        keys_sorted, dist_sorted = keys[order], dist[order]
        last_of_run = np.r_[keys_sorted[1:] != keys_sorted[:-1], True]
        self.n_duplicate_edges = int(np.count_nonzero(~last_of_run))
        self.edge_keys = keys_sorted[last_of_run]
        self.edge_dist = dist_sorted[last_of_run]
        self.edge_u, self.edge_v = lo, hi

        planes = _read_table(p / "airplanes.csv", ["Airplane_ID", "Speed_kts"])
        plane_ids = _as_int(planes["Airplane_ID"], "airplanes.csv Airplane_ID")
        speeds = np.zeros(int(plane_ids.max()) + 1 if plane_ids.size else 0, dtype=np.float64)
        speeds[plane_ids] = planes["Speed_kts"].to_numpy(dtype=np.float64)

        assignment = _read_table(p / "airplane_flight_assignment.csv", ["Airplane_ID", "Flight_ID"])
        a_plane = _as_int(assignment["Airplane_ID"], "airplane_flight_assignment Airplane_ID")
        a_flight = _as_int(assignment["Flight_ID"], "airplane_flight_assignment Flight_ID")

        flights = _read_table(p / "flights.csv", ["Flight_ID", "Position", "Time"])
        f_id = _as_int(flights["Flight_ID"], "flights.csv Flight_ID")
        f_pos = _as_int(flights["Position"], "flights.csv Position")
        f_time = _as_int(flights["Time"], "flights.csv Time")
        order = np.lexsort((f_time, f_id))
        self.filed_f, self.filed_v, self.filed_t = f_id[order], f_pos[order], f_time[order]
        ids = np.unique(f_id)
        self.n_flights = len(ids)
        if not np.array_equal(ids, np.arange(self.n_flights)):
            raise ValidatorError("flights.csv: Flight_ID is not 0..|F|-1")
        if f_pos.size and (f_pos.min() < 0 or f_pos.max() >= N):
            raise ValidatorError("flights.csv: Position outside 0..|V|-1")
        starts = np.searchsorted(self.filed_f, np.arange(self.n_flights), side="left")
        ends = np.searchsorted(self.filed_f, np.arange(self.n_flights), side="right") - 1
        self.filed_first, self.filed_last = starts, ends
        self.filed_origin = self.filed_v[starts]
        self.filed_dest = self.filed_v[ends]
        self.filed_dep = self.filed_t[starts]
        self.filed_arr = self.filed_t[ends]

        if len(np.unique(a_flight)) != len(a_flight):
            raise ValidatorError("airplane_flight_assignment.csv: a flight has two aircraft")
        if not np.array_equal(np.sort(a_flight), np.arange(self.n_flights)):
            raise ValidatorError("airplane_flight_assignment.csv: not every flight has an aircraft")
        self.flight_aircraft = np.empty(self.n_flights, dtype=np.int64)
        self.flight_aircraft[a_flight] = a_plane
        if self.flight_aircraft.max() >= len(speeds):
            raise ValidatorError("airplane_flight_assignment.csv names an aircraft airplanes.csv lacks")
        self.flight_speed = speeds[self.flight_aircraft]

        # Rotations: each aircraft's legs in filed order (filed departure, then flight id).
        rot = np.lexsort((np.arange(self.n_flights), self.filed_dep, self.flight_aircraft))
        same_plane = self.flight_aircraft[rot[1:]] == self.flight_aircraft[rot[:-1]]
        self.rotation_prev = rot[:-1][same_plane]
        self.rotation_next = rot[1:][same_plane]

        # The allocation: navaid_sector_assignment.csv broadcast, a vertex it does not list in
        # its own sector, then the change-points of navaid_sector_schedule.csv (From_Time > 0).
        alloc = _read_table(p / "navaid_sector_assignment.csv", ["Navaid_ID", "Sector_ID"])
        a_nav = _as_int(alloc["Navaid_ID"], "navaid_sector_assignment Navaid_ID")
        a_sec = _as_int(alloc["Sector_ID"], "navaid_sector_assignment Sector_ID")
        if a_nav.size and (min(a_nav.min(), a_sec.min()) < 0 or max(a_nav.max(), a_sec.max()) >= N):
            raise ValidatorError("navaid_sector_assignment.csv: id outside 0..|V|-1")
        self.static_allocation = np.arange(N, dtype=np.int64)
        self.static_allocation[a_nav] = a_sec
        self.schedule: List[Tuple[int, int, int]] = []
        sched_path = p / "navaid_sector_schedule.csv"
        if sched_path.is_file():
            sched = _read_table(sched_path, ["Navaid_ID", "Sector_ID", "From_Time"])
            s_nav = _as_int(sched["Navaid_ID"], "schedule Navaid_ID")
            s_sec = _as_int(sched["Sector_ID"], "schedule Sector_ID")
            s_from = _as_int(sched["From_Time"], "schedule From_Time")
            if np.any((s_from == 0) & (s_sec != self.static_allocation[s_nav])):
                raise ValidatorError("navaid_sector_schedule.csv: From_Time 0 rows disagree with "
                                     "navaid_sector_assignment.csv")
            self.schedule = sorted((int(n), int(s), int(t))
                                   for n, s, t in zip(s_nav, s_sec, s_from) if t > 0)

        self.max_filed_time = int(self.filed_t.max()) if self.filed_t.size else 0
        self.window = evaluation_window(self.max_filed_time, self.tg)
        self.day_end = MAX_TIME_HOURS * self.tg

    # -- allocation -------------------------------------------------------------------------

    def initial_column(self, t: int) -> np.ndarray:
        """The instance's allocation at timestep t."""
        if not self.schedule:
            return self.static_allocation
        col = self.static_allocation.copy()
        for nav, sec, start in self.schedule:     # sorted by (nav, from_time)
            if start <= t:
                col[nav] = sec
        return col

    def initial_allocation(self, width: int) -> np.ndarray:
        """The instance's allocation as a dense |V| x width array."""
        out = np.repeat(self.static_allocation[:, None], width, axis=1)
        for nav, sec, start in self.schedule:
            if start < width:
                out[nav, start:] = sec
        return out

    # -- edges ------------------------------------------------------------------------------

    def lookup_edges(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(found, dist) for undirected vertex pairs (a, b)."""
        N = self.n_vertices
        keys = np.minimum(a, b) * N + np.maximum(a, b)
        idx = np.searchsorted(self.edge_keys, keys)
        idx_clipped = np.minimum(idx, max(len(self.edge_keys) - 1, 0))
        found = (idx < len(self.edge_keys)) & (self.edge_keys[idx_clipped] == keys) \
            if len(self.edge_keys) else np.zeros(len(keys), dtype=bool)
        dist = np.where(found, self.edge_dist[idx_clipped] if len(self.edge_keys) else 0.0, np.nan)
        return found, dist

    def filed_matrix_entries(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(flight, t, vertex) of the filed plan, sorted by flight then time."""
        return self.filed_f, self.filed_t, self.filed_v

    # -- self-check ---------------------------------------------------------------------------

    def self_check(self) -> List[str]:
        """The filed plan must satisfy the validator's own model.

        A failure here means the validator's edge-time formula, rotation rule or allocation rule
        disagrees with the generator on this instance -- so a verdict on a solution could not be
        trusted either. Every run on such an instance is reported as ERROR.
        """
        problems = []
        f, t, v = self.filed_f, self.filed_t, self.filed_v
        same = f[1:] == f[:-1]
        if np.any(same & (t[1:] == t[:-1])):
            problems.append("flights.csv lists a flight twice at one timestep")
        if np.any(self.filed_origin == self.filed_dest):
            problems.append("a filed flight lands where it departed")
        a, b, dt, fl = v[:-1][same], v[1:][same], (t[1:] - t[:-1])[same], f[:-1][same]
        found, dist = self.lookup_edges(a, b)
        if not np.all(found):
            problems.append(f"filed plan uses {int(np.count_nonzero(~found))} non-edges")
        expected = edge_timesteps(np.where(found, dist, 1.0), self.flight_speed[fl], self.tg)
        bad = found & (dt != expected)
        if np.any(bad):
            i = int(np.flatnonzero(bad)[0])
            problems.append(f"filed plan: {int(np.count_nonzero(bad))} edge times differ from "
                            f"max(1,ceil(d/v/slot)), e.g. flight {int(fl[i])} {int(a[i])}->{int(b[i])} "
                            f"filed {int(dt[i])}, formula {int(expected[i])}")
        prev, nxt = self.rotation_prev, self.rotation_next
        if np.any(self.filed_origin[nxt] != self.filed_dest[prev]):
            problems.append("filed rotations are not continuous in place")
        if np.any(self.filed_dep[nxt] < self.filed_arr[prev] + 1):
            problems.append("filed rotations violate dep >= arr + 1")
        col = self.static_allocation
        if np.any(col[col] != col):
            problems.append("initial allocation: a sector does not contain its own vertex")
        if np.any(col[self.airports] != self.airports) or np.any(~self.is_airport & self.is_airport[col]):
            problems.append("initial allocation: an airport is not a singleton sector")
        return problems


# ---------------------------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------------------------

def find_matrix_file(run_dir: Path, name: str) -> Optional[Path]:
    for ext in (".csv.gz", ".csv", ".npz"):
        candidate = run_dir / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def read_matrix(path: Path) -> np.ndarray:
    """An integer matrix from csv, csv.gz or npz. 02_ASP writes floats with %g: accepted if integral."""
    try:
        if path.suffix == ".npz":
            with np.load(path) as bundle:
                array = bundle[bundle.files[0]]
        else:
            try:
                array = pd.read_csv(path, header=None, dtype=np.int32, engine="c").to_numpy()
            except (ValueError, OverflowError):
                array = pd.read_csv(path, header=None, dtype=np.float64, engine="c").to_numpy()
    except pd.errors.EmptyDataError:
        return np.zeros((0, 0), dtype=np.int64)
    except Exception as exc:                        # truncated gzip, garbage, ...
        raise MatrixFormatError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[None, :]
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
            raise MatrixFormatError(f"{path.name}: non-integer entries")
    if array.size and (array.max() > np.iinfo(np.int32).max or array.min() < np.iinfo(np.int32).min):
        raise MatrixFormatError(f"{path.name}: entries outside int32")
    return array.astype(np.int32)


# ---------------------------------------------------------------------------------------------
# Evaluating one solution
# ---------------------------------------------------------------------------------------------

class Evaluation:
    """Everything recomputed from one set of matrices."""

    def __init__(self):
        self.recomputed: Dict[str, Optional[int]] = {m: None for m in METRICS}
        self.hard: Dict[str, int] = {c: 0 for c in HARD_CHECKS}
        self.consistency: Dict[str, int] = {c: 0 for c in CONSISTENCY_CHECKS}
        self.info: Dict[str, Optional[int]] = {k: None for k in INFO_FIELDS}
        self.examples: List[str] = []
        self.unverified: List[str] = []

    def hard_fail(self, check: str, count: int, example: str = "") -> None:
        if count:
            self.hard[check] += int(count)
            if example and len(self.examples) < 3:
                self.examples.append(f"{check}: {example}")

    def inconsistent(self, check: str, count: int, example: str = "") -> None:
        if count:
            self.consistency[check] += int(count)
            if example and len(self.examples) < 3:
                self.examples.append(f"{check}: {example}")


def _column_groups(allocation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(group id per column, first column of each group): runs of identical columns."""
    width = allocation.shape[1]
    if width == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    changed = np.any(allocation[:, 1:] != allocation[:, :-1], axis=0)
    group = np.concatenate([[0], np.cumsum(changed)]).astype(np.int64)
    starts = np.concatenate([[0], np.flatnonzero(changed) + 1]).astype(np.int64)
    return group, starts


def evaluate_solution(inst: Instance, nav: np.ndarray, sec: Optional[np.ndarray],
                      alloc: Optional[np.ndarray], capm: Optional[np.ndarray],
                      metric: str, check_connectivity: bool = True) -> Evaluation:
    ev = Evaluation()
    N, F, W, tg = inst.n_vertices, inst.n_flights, inst.window, inst.tg

    # ---- shapes --------------------------------------------------------------------------
    width = nav.shape[1]
    ev.info["n_flights"] = F
    ev.info["matrix_width"] = width
    ev.info["window_W"] = W
    if nav.shape[0] < F:
        ev.hard_fail("matrix_shape", 1, f"navpoint matrix has {nav.shape[0]} rows for {F} flights")
    if sec is not None and sec.shape != nav.shape:
        ev.hard_fail("matrix_shape", 1, f"sector matrix {sec.shape} vs navpoint matrix {nav.shape}")
        sec = None
    if alloc is not None and alloc.shape[0] != N:
        ev.hard_fail("matrix_shape", 1, f"allocation has {alloc.shape[0]} rows for {N} vertices")
        alloc = None
    if alloc is not None and alloc.shape[1] == 0:
        ev.hard_fail("matrix_shape", 1, "allocation has no columns")
        alloc = None
    if capm is not None and alloc is not None and capm.shape != alloc.shape:
        ev.hard_fail("matrix_shape", 1, f"capacity matrix {capm.shape} vs allocation {alloc.shape}")
        capm = None
    if ev.hard["matrix_shape"] and nav.shape[0] < F:
        return ev

    # ---- trajectories from the navpoint matrix -------------------------------------------
    extra = nav[F:]
    if extra.size and np.any(extra != -1):
        ev.hard_fail("extra_flight_rows", int(np.count_nonzero(np.any(extra != -1, axis=1))),
                     f"rows >= {F} are not empty")
    nav = nav[:F]
    if sec is not None:
        sec = sec[:F]
    rows, cols = np.nonzero(nav != -1)
    vals = nav[rows, cols].astype(np.int64)
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)

    bad_id = (vals < 0) | (vals >= N)
    if np.any(bad_id):
        i = int(np.flatnonzero(bad_id)[0])
        ev.hard_fail("bad_vertex_id", int(np.count_nonzero(bad_id)),
                     f"flight {rows[i]} t={cols[i]} vertex {vals[i]}")
        keep = ~bad_id
        rows, cols, vals = rows[keep], cols[keep], vals[keep]

    present = np.zeros(F, dtype=bool)
    present[rows] = True
    if not np.all(present):
        missing = np.flatnonzero(~present)
        ev.hard_fail("flight_missing", len(missing), f"flight {int(missing[0])} has no entry")

    first = np.searchsorted(rows, np.arange(F), side="left")
    last = np.searchsorted(rows, np.arange(F), side="right") - 1
    fl_ok = np.flatnonzero(present)
    first_ok, last_ok = first[fl_ok], last[fl_ok]

    sol_origin = vals[first_ok]
    sol_dest = vals[last_ok]
    sol_dep = cols[first_ok]
    sol_arr = cols[last_ok]

    bad = (sol_origin != inst.filed_origin[fl_ok]) | (sol_dest != inst.filed_dest[fl_ok])
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        ev.hard_fail("endpoints", int(np.count_nonzero(bad)),
                     f"flight {int(fl_ok[i])}: {int(sol_origin[i])}->{int(sol_dest[i])}, filed "
                     f"{int(inst.filed_origin[fl_ok[i]])}->{int(inst.filed_dest[fl_ok[i]])}")

    bad = sol_dep < inst.filed_dep[fl_ok]
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        ev.hard_fail("departs_before_filed", int(np.count_nonzero(bad)),
                     f"flight {int(fl_ok[i])} departs t={int(sol_dep[i])}, filed {int(inst.filed_dep[fl_ok[i]])}")

    # destination reached before the last entry
    dest_of_row = inst.filed_dest[rows]
    is_last = np.zeros(len(rows), dtype=bool)
    is_last[last_ok] = True
    early_dest = (vals == dest_of_row) & ~is_last
    if np.any(early_dest):
        bad_flights = np.unique(rows[early_dest])
        ev.hard_fail("continues_after_destination", len(bad_flights),
                     f"flight {int(bad_flights[0])} reaches its destination before its last entry")

    # en-route vertex visited twice
    enroute = ~inst.is_airport[vals]
    keys = rows[enroute] * N + vals[enroute]
    uniq, counts = np.unique(keys, return_counts=True)
    if np.any(counts > 1):
        dup_flights = np.unique(uniq[counts > 1] // N)
        ev.hard_fail("not_simple", len(dup_flights),
                     f"flight {int(dup_flights[0])} visits an en-route vertex twice")

    # consecutive waypoints
    same = rows[1:] == rows[:-1]
    j = np.flatnonzero(same)
    a, b = vals[j], vals[j + 1]
    dt = cols[j + 1] - cols[j]
    seg_flight = rows[j]
    wait = (a == b) & inst.is_airport[a]
    ev.info["n_airport_waits"] = int(np.count_nonzero(wait))
    move = a != b                     # a stay at an en-route vertex is counted by not_simple
    found, dist = inst.lookup_edges(a[move], b[move])
    if not np.all(found):
        k = int(np.flatnonzero(~found)[0])
        idx = j[move][k]
        ev.hard_fail("not_an_edge", int(np.count_nonzero(~found)),
                     f"flight {int(rows[idx])}: {int(vals[idx])}->{int(vals[idx + 1])} at "
                     f"t={int(cols[idx])}->{int(cols[idx + 1])}")
    expected = edge_timesteps(np.where(found, dist, 1.0), inst.flight_speed[seg_flight[move]], tg)
    bad = found & (dt[move] != expected)
    if np.any(bad):
        k = int(np.flatnonzero(bad)[0])
        idx = j[move][k]
        ev.hard_fail("edge_time", int(np.count_nonzero(bad)),
                     f"flight {int(rows[idx])}: {int(vals[idx])}->{int(vals[idx + 1])} takes "
                     f"{int(dt[move][k])} timesteps, model {int(expected[k])}")

    # ---- rotations ----------------------------------------------------------------------
    prev, nxt = inst.rotation_prev, inst.rotation_next
    both = present[prev] & present[nxt]
    prev, nxt = prev[both], nxt[both]
    arr_prev = cols[last[prev]]
    dep_next = cols[first[nxt]]
    place = vals[first[nxt]] != vals[last[prev]]
    if np.any(place):
        i = int(np.flatnonzero(place)[0])
        ev.hard_fail("rotation_place", int(np.count_nonzero(place)),
                     f"aircraft {int(inst.flight_aircraft[nxt[i]])}: flight {int(nxt[i])} departs "
                     f"{int(vals[first[nxt[i]]])}, flight {int(prev[i])} landed {int(vals[last[prev[i]]])}")
    overlap = dep_next < arr_prev
    if np.any(overlap):
        i = int(np.flatnonzero(overlap)[0])
        ev.hard_fail("rotation_overlap", int(np.count_nonzero(overlap)),
                     f"aircraft {int(inst.flight_aircraft[nxt[i]])}: flight {int(nxt[i])} departs "
                     f"t={int(dep_next[i])}, flight {int(prev[i])} lands t={int(arr_prev[i])}")
    touch = dep_next == arr_prev
    if np.any(touch):
        i = int(np.flatnonzero(touch)[0])
        ev.hard_fail("rotation_no_turnaround", int(np.count_nonzero(touch)),
                     f"aircraft {int(inst.flight_aircraft[nxt[i]])}: flight {int(nxt[i])} departs "
                     f"t={int(dep_next[i])}, the timestep flight {int(prev[i])} lands")

    # ---- ARRIVAL-DELAY, REROUTE -----------------------------------------------------------
    delta = sol_arr - inst.filed_arr[fl_ok]
    ev.recomputed["ARRIVAL-DELAY"] = int(apply_arrival_delay_metric(delta, metric).sum())
    ev.info["n_landing_after_day"] = int(np.count_nonzero(sol_arr > inst.day_end))

    ff, ft, fv = inst.filed_matrix_entries()
    span = np.int64(max(width, inst.max_filed_time + 1, W) + 1)
    sol_keys = (rows * span + cols) * N + vals
    filed_keys = (ff * span + ft) * N + fv
    in_window = cols < W
    diff_w = np.setxor1d(sol_keys[in_window], filed_keys[ft < W], assume_unique=True)
    ev.recomputed["REROUTE"] = int(len(np.unique(diff_w // (span * N))))
    changed_flights = np.unique(np.setxor1d(sol_keys, filed_keys, assume_unique=True) // (span * N))
    # path changed: the SEQUENCE of vertices differs (rank within the flight, vertex)
    rank_sol = np.arange(len(rows)) - first[rows]
    rank_filed = np.arange(len(ff)) - inst.filed_first[ff]
    lmax = np.int64(max(int(rank_sol.max()) if len(rank_sol) else 0,
                        int(rank_filed.max()) if len(rank_filed) else 0) + 2)
    seq_diff = np.setxor1d((rows * lmax + rank_sol) * N + vals,
                           (ff * lmax + rank_filed) * N + fv, assume_unique=True)
    path_changed = np.unique(seq_diff // (lmax * N))
    ev.info["n_path_changed"] = int(len(path_changed))
    ev.info["n_delayed_only"] = int(len(np.setdiff1d(changed_flights, path_changed)))

    # ---- everything that needs the sector allocation -------------------------------------
    if alloc is None:
        ev.unverified += ["OVERLOAD", "SECTOR-NUMBER", "SECTOR-DIFF", "RECONFIG"]
        return ev

    a_width = alloc.shape[1]
    if a_width != width:
        ev.inconsistent("allocation_width", abs(a_width - width),
                        f"allocation has {a_width} columns, flight matrices {width}")
    bad_sec = (alloc < 0) | (alloc >= N)
    if np.any(bad_sec):
        ev.hard_fail("bad_sector_id", int(np.count_nonzero(bad_sec)), "allocation entry outside 0..|V|-1")
        return ev

    group, starts = _column_groups(alloc)
    n_groups = len(starts)
    lengths = np.diff(np.r_[starts, a_width])

    def group_of(t: np.ndarray) -> np.ndarray:
        return group[np.minimum(t, a_width - 1)]

    eu, evv = inst.edge_u, inst.edge_v
    enroute_edge = ~inst.is_airport[eu] & ~inst.is_airport[evv]
    eu_en, ev_en = eu[enroute_edge], evv[enroute_edge]
    enroute_vertex = ~inst.is_airport

    distinct = np.zeros(n_groups, dtype=np.int64)
    capvecs: Dict[int, np.ndarray] = {}
    conn_cache: Dict[int, int] = {}
    for g, start in enumerate(starts):
        col = alloc[:, start].astype(np.int64)
        glen = int(lengths[g])
        used = np.unique(col)
        distinct[g] = len(used)
        # representative: sector i contains vertex i
        bad_rep = used[col[used] != used]
        if len(bad_rep):
            ev.hard_fail("sector_representative", len(bad_rep) * glen,
                         f"t={int(start)}: sector {int(bad_rep[0])} does not contain vertex {int(bad_rep[0])}")
        # airports atomic
        n_air = int(np.count_nonzero(col[inst.airports] != inst.airports)) + \
            int(np.count_nonzero(enroute_vertex & inst.is_airport[col]))
        if n_air:
            ev.hard_fail("airport_not_atomic", n_air * glen, f"t={int(start)}: an airport is not a singleton sector")
        # en-route sectors connected
        if check_connectivity:
            h = hash(col.tobytes())
            if h not in conn_cache:
                keep = col[eu_en] == col[ev_en]
                graph = coo_matrix((np.ones(int(keep.sum()), dtype=np.int8), (eu_en[keep], ev_en[keep])),
                                   shape=(N, N))
                _, labels = connected_components(graph, directed=False)
                pairs = np.unique(col[enroute_vertex] * np.int64(N) + labels[enroute_vertex])
                sectors_of_pairs = pairs // N
                _, per_sector = np.unique(sectors_of_pairs, return_counts=True)
                conn_cache[h] = int(np.count_nonzero(per_sector > 1))
            if conn_cache[h]:
                ev.hard_fail("sector_disconnected", conn_cache[h] * glen,
                             f"t={int(start)}: {conn_cache[h]} en-route sector(s) not connected")
        # composite capacity: max of the members' capacities, 0 for an empty sector
        members = np.bincount(col, minlength=N)
        capvec = np.full(N, np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(capvec, col, inst.capacity)
        capvec[members == 0] = 0
        capvecs[g] = capvec
        if capm is not None:
            block = capm[:, start:start + glen] != capvec[:, None]
            mism = int(np.count_nonzero(block))
            if mism:
                s_bad, t_bad = np.nonzero(block)
                ev.inconsistent("capacity_matrix", mism,
                                f"sector {int(s_bad[0])} t={int(start + t_bad[0])}: saved capacity "
                                f"{int(capm[s_bad[0], start + t_bad[0]])}, max-composition {int(capvec[s_bad[0]])}")

    # ---- SECTOR-NUMBER, SECTOR-DIFF, RECONFIG (and the paper's windows) -------------------
    t_window = np.arange(W)
    ev.recomputed["SECTOR-NUMBER"] = int(distinct[group_of(t_window)].sum())
    day = np.arange(1, inst.day_end + 1)
    ev.info["paper_active_sectors"] = int(distinct[group_of(day)].sum())

    sector_diff = 0
    for g in range(1, n_groups):
        sector_diff += int(np.count_nonzero(alloc[:, starts[g]] != alloc[:, starts[g - 1]]))
    ev.recomputed["SECTOR-DIFF"] = sector_diff
    paper_changes = 0
    for t in range(1, inst.day_end + 1):
        g1, g0 = group_of(np.array([t]))[0], group_of(np.array([t - 1]))[0]
        if g1 != g0:
            paper_changes += int(np.count_nonzero(alloc[:, starts[g1]] != alloc[:, starts[g0]]))
    ev.info["paper_sector_changes"] = paper_changes

    def reconfig_over(timesteps: np.ndarray) -> int:
        total = 0
        if not inst.schedule:
            gs, per = np.unique(group_of(timesteps), return_counts=True)
            for g, n in zip(gs, per):
                total += int(np.count_nonzero(alloc[:, starts[g]] != inst.static_allocation)) * int(n)
            return total
        for t in timesteps:
            g = group_of(np.array([t]))[0]
            total += int(np.count_nonzero(alloc[:, starts[g]] != inst.initial_column(int(t))))
        return total

    ev.recomputed["RECONFIG"] = reconfig_over(t_window)
    ev.info["paper_reconfigurations"] = reconfig_over(np.arange(0, inst.day_end + 1))

    # ---- occupancy under the half rule, and OVERLOAD ---------------------------------------
    # A flight at vertex x_j at t_j and x_{j+1} at t_{j+1} occupies sec(x_j, t) for
    # t in [t_j, t_j + floor((t_{j+1} - t_j)/2)] and sec(x_{j+1}, t) after it (LPNMR Def. 6).
    seg = j                                      # index of the first waypoint of each segment
    L = dt
    total = int(L.sum())
    seg_of_cell = np.repeat(np.arange(len(seg)), L)
    offset = np.arange(total) - np.repeat(np.cumsum(L) - L, L) + 1
    cell_t = cols[seg][seg_of_cell] + offset
    first_half = offset <= (L // 2)[seg_of_cell]
    cell_ref = np.where(first_half, vals[seg][seg_of_cell], vals[seg + 1][seg_of_cell])
    cell_f = rows[seg][seg_of_cell]
    # plus the departure cell of every flight
    cell_t = np.r_[cols[first_ok], cell_t]
    cell_ref = np.r_[vals[first_ok], cell_ref]
    cell_f = np.r_[fl_ok, cell_f]
    cell_sec = alloc[cell_ref, np.minimum(cell_t, a_width - 1)].astype(np.int64)

    if sec is not None:
        saved = sec[cell_f, cell_t]
        wrong = int(np.count_nonzero(saved != cell_sec))
        stray = int(np.count_nonzero(sec != -1)) - (len(cell_f) - int(np.count_nonzero(saved == -1)))
        if wrong + stray:
            i = np.flatnonzero(saved != cell_sec)
            example = (f"flight {int(cell_f[i[0]])} t={int(cell_t[i[0]])}: saved sector "
                       f"{int(saved[i[0]])}, trajectory implies {int(cell_sec[i[0]])}") if len(i) else \
                f"{stray} saved cells outside the flights' trajectories"
            ev.inconsistent("sector_rows", wrong + max(stray, 0), example)

    def overload_of(sectors: np.ndarray, timesteps: np.ndarray) -> int:
        if len(sectors) == 0:
            return 0
        span_t = np.int64(max(int(timesteps.max()) + 1, a_width))
        keys = sectors * span_t + timesteps
        uniq_keys, load = np.unique(keys, return_counts=True)
        s_k, t_k = uniq_keys // span_t, uniq_keys % span_t
        cap_k = np.zeros(len(uniq_keys), dtype=np.int64)
        g_k = group_of(t_k)
        for g in np.unique(g_k):
            mask = g_k == g
            valid = (s_k[mask] >= 0) & (s_k[mask] < N)
            cap_k[np.flatnonzero(mask)[valid]] = capvecs[int(g)][s_k[mask][valid]]
        return int(np.maximum(load - cap_k, 0).sum())

    ev.recomputed["OVERLOAD"] = overload_of(cell_sec, cell_t)
    if sec is not None:
        r, c = np.nonzero(sec != -1)
        ev.info["overload_from_saved_sector_rows"] = overload_of(sec[r, c].astype(np.int64), c.astype(np.int64))
    return ev


# ---------------------------------------------------------------------------------------------
# Campaign layout
# ---------------------------------------------------------------------------------------------

def results_dirs() -> Dict[str, str]:
    """system key -> the solver_outputs/ folder it writes to, from build_system_config itself.

    Read from THIS checkout's start_benchmark_caller.py, so run the validator from a checkout whose
    build_system_config is the campaign's: at 818136e 2A_Reroute writes to 0A_Reroute and
    03A_CASA to 03_DELAY. A checkout that gave 03A_CASA its own folder would look for matrices
    the campaign never wrote there.
    """
    fallback = {"2A_Reroute": "0A_Reroute", "03A_CASA": "03_DELAY"}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from start_benchmark_caller import build_arg_parser, build_system_config  # noqa: E402
        args = build_arg_parser().parse_args(["."])
        systems = build_system_config(Path(__file__).resolve().parent, Path("OUT"), "validate", args)
        mapping = {}
        for system in systems:
            for flag in system["cmd"]:
                if flag.startswith("--results-root="):
                    mapping[system["key"]] = Path(flag.split("=", 1)[1]).name
        return mapping
    except Exception as exc:          # pragma: no cover - only without the caller's dependencies
        print(f"[WARN] could not read build_system_config ({exc}); using the built-in folder map",
              file=sys.stderr)
        return fallback


class Folders:
    def __init__(self):
        self.mapping = results_dirs()

    def folder(self, system: str) -> str:
        return self.mapping.get(system, system)

    def sharing(self, system: str) -> List[str]:
        """Other systems that write to the same solver_outputs/ folder."""
        mine = self.folder(system)
        return [s for s, f in self.mapping.items() if f == mine and s != system]


def load_claims(problem_dir: Path, instance: str, system: str) -> Tuple[str, dict]:
    """(outcome, last JSON line) of one run, from individual_outputs/<INSTANCE>_<SYSTEM>.json."""
    path = problem_dir / "individual_outputs" / f"{instance}_{system}.json"
    lines = None
    if path.is_file():
        try:
            lines = json.loads(path.read_text(encoding="utf-8")).get("object")
        except (json.JSONDecodeError, AttributeError):
            lines = None
    if isinstance(lines, str):                 # a propagated failure code without any output
        return OUTCOMES.get(lines, lines), {}
    if not isinstance(lines, list) or not lines or not isinstance(lines[-1], dict):
        return "unknown", {}
    last = lines[-1]
    return OUTCOMES.get(last.get("ERROR"), str(last.get("ERROR"))), last


def discover_runs(problem_dir: Path, systems: Optional[List[str]]) -> List[Tuple[str, str]]:
    """(instance, system) pairs of this problem folder, from individual_outputs/."""
    folder = problem_dir / "individual_outputs"
    if not folder.is_dir():
        raise ValidatorError(f"{folder} does not exist")
    names = sorted(p.name for p in folder.glob("*.json"))
    known = systems
    if known is None:                              # --systems all: every system the caller knows
        known = sorted(set(results_dirs()) | set(PUBLISHED_SYSTEMS), key=len, reverse=True)
    runs = []
    for name in names:
        for system in sorted(known, key=len, reverse=True):
            suffix = f"_{system}.json"
            if name.endswith(suffix):
                runs.append((name[: -len(suffix)], system))
                break
    if systems is not None:
        order = {s: i for i, s in enumerate(systems)}
        runs = [r for r in runs if r[1] in order]
        runs.sort(key=lambda r: (r[0], order[r[1]]))
    return runs


def problem_granularity(problem: str, problem_dir: Path, instance_root: Path,
                        override: Optional[int]) -> int:
    if override:
        return int(override)
    manifest = instance_root / "problems.tsv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                if row.get("problem_dir") == problem and row.get("time_granularity", "").isdigit():
                    return int(row["time_granularity"])
    provenance = problem_dir.parent / f"run_provenance_{problem}.txt"
    if provenance.is_file():
        for line in provenance.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("time_granularity=") and line.split("=", 1)[1].strip().isdigit():
                return int(line.split("=", 1)[1])
    match = re.search(r"-TG(\d+)(?:-|$)", problem)
    if match:
        return int(match.group(1))
    # last resort: what the runs themselves were given (manifest.json of any saved run)
    for manifest in sorted((problem_dir / "solver_outputs").glob("*/*/manifest.json")):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("source", {}).get(
                "timestep_granularity")
        except (json.JSONDecodeError, AttributeError):
            continue
        if value is not None:
            return int(value)
    raise ValidatorError(f"cannot determine the time granularity of {problem}; pass --timestep-granularity")


# ---------------------------------------------------------------------------------------------
# One problem folder
# ---------------------------------------------------------------------------------------------

def _load_run_matrices(run_dir: Path) -> Tuple[Dict[str, Optional[np.ndarray]], dict, List[str]]:
    """The matrices of one run, its manifest, and any read problems."""
    manifest = {}
    if (run_dir / "manifest.json").is_file():
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    mats: Dict[str, Optional[np.ndarray]] = {}
    problems = []
    saved = manifest.get("saved", {}) if isinstance(manifest, dict) else {}
    for name in MATRIX_NAMES:
        path = find_matrix_file(run_dir, name)
        if path is None:
            mats[name] = None
            if name in saved:
                problems.append(f"{name} listed in manifest.json but missing")
            continue
        try:
            mats[name] = read_matrix(path)
        except MatrixFormatError as exc:
            mats[name] = None
            problems.append(str(exc))
            continue
        want = saved.get(name, {}).get("shape")
        if want is not None and list(mats[name].shape) != list(want) and not (
                mats[name].size == 0 and 0 in want):
            problems.append(f"{name} has shape {list(mats[name].shape)}, manifest.json says {want}")
    return mats, manifest, problems


def _claims_match(claims: dict, recomputed: Dict[str, Optional[int]]) -> Tuple[bool, List[str]]:
    wrong = []
    for m in METRICS:
        if recomputed.get(m) is None or m not in claims:
            continue
        try:
            if int(claims[m]) != int(recomputed[m]):
                wrong.append(m)
        except (TypeError, ValueError):
            wrong.append(m)
    return not wrong, wrong


def validate_problem(problem_dir: Path, instance_root: Path, out_dir: Path, systems: Optional[List[str]],
                     folders: Folders, resume: bool, tg_override: Optional[int], default_metric: str,
                     check_connectivity: bool, verbose: bool) -> int:
    problem_dir = problem_dir.resolve()
    problem = problem_dir.name[len("output_"):] if problem_dir.name.startswith("output_") else problem_dir.name
    campaign = problem_dir.parent.name
    target_dir = out_dir / campaign
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"validation_{problem}.csv"

    runs = discover_runs(problem_dir, systems)
    done = set()
    if resume and csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != ERROR:
                    done.add((row["instance"], row["system"]))
        # rewrite without the ERROR rows, which are retried
        kept = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            kept = [row for row in csv.DictReader(fh) if row.get("status") != ERROR]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(kept)
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()

    tg = problem_granularity(problem, problem_dir, instance_root, tg_override)
    worst = 0
    by_instance: Dict[str, List[str]] = {}
    for instance, system in runs:
        by_instance.setdefault(instance, []).append(system)

    t_problem = time.time()
    n_rows = 0
    for instance, inst_systems in by_instance.items():
        todo = [s for s in inst_systems if (instance, s) not in done]
        if not todo:
            continue
        rows: List[dict] = []
        inst = None
        inst_error = ""
        try:
            inst = Instance(instance_root / problem / instance, tg)
            problems = inst.self_check()
            if problems:
                inst_error = "instance self-check failed: " + "; ".join(problems)
        except ValidatorError as exc:
            inst_error = f"instance: {exc}"
        except Exception as exc:
            inst_error = f"instance: {type(exc).__name__}: {exc}"

        # every system's claims, and every matrix folder evaluated once
        claims = {s: load_claims(problem_dir, instance, s) for s in set(inst_systems) |
                  {o for s in todo for o in folders.sharing(s)}}
        evaluations: Dict[Path, Tuple[Optional[Evaluation], List[str], dict, float, str]] = {}

        for system in todo:
            t0 = time.time()
            outcome, claim = claims[system]
            run_dir = problem_dir / "solver_outputs" / folders.folder(system) / instance
            row = {c: "" for c in CSV_COLUMNS}
            row.update(folder=campaign, problem=problem, instance=instance, system=system,
                       outcome=outcome, matrix_dir=str(run_dir.relative_to(problem_dir)),
                       timestep_granularity=tg,
                       claimed_computation_finished=claim.get("COMPUTATION-FINISHED", ""))
            for m in METRICS:
                row[f"claimed_{m}"] = claim.get(m, "")
            failed: List[str] = []
            notes: List[str] = []
            try:
                if inst_error:
                    row["status"] = ERROR
                    row["note"] = inst_error
                    rows.append(row)
                    continue
                has_matrix = run_dir.is_dir() and find_matrix_file(run_dir, "converted_navpoint_matrix") is not None
                if not has_matrix and not (run_dir.is_dir() and any(run_dir.iterdir())):
                    row["status"] = NO_MATRIX
                    row["matrix_dir"] = ""
                    if outcome == "ok":
                        row["failed_checks"] = "finished_run_without_matrix"
                    rows.append(row)
                    continue

                if run_dir not in evaluations:
                    t_eval = time.time()
                    mats, manifest, read_problems = _load_run_matrices(run_dir)
                    source = manifest.get("source", {}) if isinstance(manifest, dict) else {}
                    run_tg = source.get("timestep_granularity")
                    metric = str(claim.get("ARRIVAL-DELAY-METRIC") or source.get("arrival_delay_metric")
                                 or default_metric).lower()
                    if metric not in ARRIVAL_DELAY_METRICS:
                        metric = default_metric
                    ev = None
                    if run_tg is not None and int(run_tg) != tg:
                        read_problems.append(f"granularity_mismatch: the run used timestep granularity "
                                             f"{run_tg}, the problem has {tg}")
                    if mats["converted_navpoint_matrix"] is None:
                        read_problems.append("converted_navpoint_matrix missing or unreadable")
                    else:
                        ev = evaluate_solution(inst, mats["converted_navpoint_matrix"],
                                               mats["converted_instance_matrix"],
                                               mats["navaid_sector_time_assignment"],
                                               mats["capacity_time_matrix"], metric,
                                               check_connectivity=check_connectivity)
                    evaluations[run_dir] = (ev, read_problems, manifest, time.time() - t_eval, metric)
                    del mats
                ev, read_problems, manifest, eval_seconds, metric = evaluations[run_dir]
                row["arrival_delay_metric"] = metric

                if ev is None:
                    row["status"] = INVALID
                    row["v_matrix_unreadable"] = 1
                    row["failed_checks"] = "matrix_unreadable"
                    row["note"] = "; ".join(read_problems)
                    rows.append(row)
                    continue

                for m in METRICS:
                    row[f"recomputed_{m}"] = "" if ev.recomputed[m] is None else ev.recomputed[m]
                for c in HARD_CHECKS:
                    row[f"v_{c}"] = ev.hard[c]
                for c in CONSISTENCY_CHECKS:
                    row[f"c_{c}"] = ev.consistency[c]
                for k in INFO_FIELDS:
                    row[k] = "" if ev.info[k] is None else ev.info[k]
                row["first_violation"] = " | ".join(ev.examples)

                hard = [c for c in HARD_CHECKS if ev.hard[c]]
                if any(p.startswith("granularity_mismatch") for p in read_problems):
                    hard = ["granularity_mismatch"] + hard
                    row["v_granularity_mismatch"] = 1
                if any(not p.startswith("granularity_mismatch") for p in read_problems):
                    hard = ["matrix_unreadable"] + hard
                    row["v_matrix_unreadable"] = 1
                notes += read_problems
                if not check_connectivity:
                    notes.append("connectivity not checked")
                inconsistent = [f"{c}" for c in CONSISTENCY_CHECKS if ev.consistency[c]]
                ok_self, wrong_self = _claims_match(claim, ev.recomputed)

                # a folder shared with other systems: whose matrix is this?
                owner = system
                others = folders.sharing(system)
                if others:
                    matches = [s for s in [system] + others
                               if claims.get(s, ("", {}))[1] and _claims_match(claims[s][1], ev.recomputed)[0]]
                    if system in matches and len(matches) == 1:
                        owner = system
                    elif system in matches:
                        owner = "ambiguous(" + "+".join(matches) + ")"
                        notes.append("folder shared with " + ",".join(others) +
                                     "; objectives identical, trajectories may be the other system's")
                    elif matches:
                        owner = matches[0]
                    else:
                        owner = "unknown"
                        notes.append("folder shared with " + ",".join(others) + "; matches no claim")
                row["matrix_owner"] = owner

                if owner not in (system,) and not owner.startswith("ambiguous") and owner != "unknown":
                    # The folder holds another system's matrix. For a run that finished, its own
                    # matrix was overwritten; for one that did not, there never was one.
                    row["status"] = NO_MATRIX
                    if outcome == "ok":
                        row["failed_checks"] = f"matrix_overwritten_by:{owner}"
                    row["note"] = "; ".join(notes + [f"solver_outputs/{folders.folder(system)}/{instance} "
                                                     f"holds {owner}'s solution"])
                    # every recomputed value and count belongs to the other system's solution
                    for key in ([f"recomputed_{m}" for m in METRICS] + [f"v_{c}" for c in HARD_CHECKS]
                                + [f"c_{c}" for c in CONSISTENCY_CHECKS] + list(INFO_FIELDS)
                                + ["first_violation"]):
                        row[key] = ""
                    rows.append(row)
                    continue

                if not claim:
                    notes.append("no claims (no JSON line)")
                if ev.unverified:
                    notes.append("not recomputable without the sector allocation: " + ",".join(ev.unverified))

                if hard:
                    row["status"] = INVALID
                    failed = hard + inconsistent + [f"claim:{m}" for m in wrong_self]
                elif inconsistent or not ok_self or not claim:
                    row["status"] = MISMATCH
                    failed = inconsistent + [f"claim:{m}" for m in wrong_self] + (["no_claims"] if not claim else [])
                elif ev.unverified:
                    row["status"] = UNVERIFIED
                    failed = [f"unverified:{m}" for m in ev.unverified]
                else:
                    row["status"] = VALID
                row["failed_checks"] = ";".join(failed)
                row["note"] = "; ".join(notes)
                row["seconds"] = round(eval_seconds, 2)
            except Exception as exc:
                row["status"] = ERROR
                row["note"] = f"validator: {type(exc).__name__}: {exc}"
                if verbose:
                    traceback.print_exc()
            finally:
                if not row.get("seconds"):
                    row["seconds"] = round(time.time() - t0, 2)
            rows.append(row)

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            for row in rows:
                writer.writerow(row)
                worst = max(worst, severity(row))
                n_rows += 1
                if verbose or row["status"] not in (VALID, NO_MATRIX):
                    print(f"  {row['status']:<10} {system_col(row['system'])} {row['instance']:<22} "
                          f"{row['outcome']:<8} {row['failed_checks'][:80]}", flush=True)
        del evaluations

    summary = summarize_rows(read_rows([csv_path]))
    summary.update(problem=problem, folder=campaign, csv=str(csv_path),
                   seconds=round(time.time() - t_problem, 1), exit_code=summary["exit_code"])
    (target_dir / f"validation_{problem}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{campaign}/{problem}] {summary['n_runs']} runs "
          f"({n_rows} new): {json.dumps(summary['by_status'], sort_keys=True)}  exit {summary['exit_code']}"
          f"  {summary['seconds']} s", flush=True)
    return max(worst, summary["exit_code"])


def system_col(system: str) -> str:
    return f"{system:<28}"


# ---------------------------------------------------------------------------------------------
# Severity, summaries
# ---------------------------------------------------------------------------------------------

def severity(row: dict) -> int:
    status = row.get("status")
    if status == ERROR:
        return 3
    if status == INVALID:
        return 2
    if status in (MISMATCH, UNVERIFIED):
        return 1
    if status == NO_MATRIX and row.get("failed_checks", "").startswith("finished_run_without_matrix"):
        return 1
    # NO_MATRIX with matrix_overwritten_by:<system> is the known 03_DELAY / 03A_CASA folder
    # collision: reported in every row and in the summary, but not allowed to turn every task's
    # exit code into 1 and so hide a real MISMATCH behind it.
    return 0


def read_rows(paths: Iterable[Path]) -> List[dict]:
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def summarize_rows(rows: List[dict]) -> dict:
    by_status: Dict[str, int] = {}
    by_system: Dict[str, Dict[str, int]] = {}
    by_check: Dict[str, int] = {}
    for row in rows:
        status = row["status"]
        if status == NO_MATRIX and row.get("failed_checks"):
            status = f"{NO_MATRIX}:{row['failed_checks'].split(':')[0]}"
        by_status[status] = by_status.get(status, 0) + 1
        by_system.setdefault(row["system"], {})
        by_system[row["system"]][status] = by_system[row["system"]].get(status, 0) + 1
        for check in filter(None, (row.get("failed_checks") or "").split(";")):
            key = check.split(":")[0] if check.startswith("matrix_overwritten") else check
            by_check[key] = by_check.get(key, 0) + 1
    worst = max((severity(r) for r in rows), default=0)
    return {"n_runs": len(rows), "by_status": by_status, "by_system": by_system,
            "by_check": by_check, "exit_code": worst}


def write_campaign_summary(out_dir: Path) -> int:
    paths = sorted(out_dir.glob("*/validation_*.csv"))
    if not paths:
        print(f"[ERROR] no validation_*.csv under {out_dir}", file=sys.stderr)
        return 3
    rows = read_rows(paths)
    summary = summarize_rows(rows)
    # per system x status table
    statuses = sorted({s for per in summary["by_system"].values() for s in per})
    table = [["system"] + statuses + ["total"]]
    for system in sorted(summary["by_system"], key=lambda s: (s not in PUBLISHED_SYSTEMS, s)):
        per = summary["by_system"][system]
        table.append([system] + [str(per.get(s, 0)) for s in statuses] + [str(sum(per.values()))])
    with (out_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(table)
    flagged = [r for r in rows if severity(r) > 0]
    with (out_dir / "validation_flagged_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(flagged)
    owners: Dict[str, int] = {}
    for r in rows:
        if r.get("matrix_owner") and r["matrix_owner"] != r["system"]:
            key = f"{r['system']} -> {r['matrix_owner']}"
            owners[key] = owners.get(key, 0) + 1
    lines = ["# Solution-matrix validation summary", "",
             f"{len(paths)} problem folder(s), {summary['n_runs']} runs, exit code {summary['exit_code']}.", "",
             "| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    lines += ["| " + " | ".join(r) + " |" for r in table[1:]]
    lines += ["", "Failed checks (runs):", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(summary["by_check"].items(), key=lambda kv: -kv[1])]
    if owners:
        lines += ["", "Matrices found in a shared folder that belong to another system:", ""]
        lines += [f"- {k}: {v}" for k, v in sorted(owners.items())]
    lines += ["", f"Flagged runs (exit code > 0): {len(flagged)} -> validation_flagged_runs.csv"]
    (out_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summary["exit_code"]


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("STATUSES"):])
    ap.add_argument("--problem-dir", type=Path, action="append", default=[],
                    help="a merged problem folder output/<FOLDER>/output_<PROBLEM> (repeatable)")
    ap.add_argument("--instance-root", type=Path, default=Path("../05_instances"),
                    help="folder holding <PROBLEM>/<INSTANCE>/ (default: ../05_instances)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="validation output; rows go to <out-dir>/<FOLDER>/validation_<PROBLEM>.csv")
    ap.add_argument("--systems", default="published",
                    help="'published' (default), 'all', or a comma-separated list of system keys")
    ap.add_argument("--timestep-granularity", type=int, default=None,
                    help="override; default: problems.tsv, run_provenance_<PROBLEM>.txt, -TG<n> in the name")
    ap.add_argument("--arrival-delay-metric", default="signed", choices=ARRIVAL_DELAY_METRICS,
                    help="used only when neither the result line nor manifest.json names the metric "
                         "(04_MIP's manifest does not); the campaign caller's default is signed")
    ap.add_argument("--no-connectivity", action="store_true",
                    help="do not require en-route sectors to be connected (LPNMR's model; ATMOS requires it)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in the problem's CSV (ERROR rows are retried)")
    ap.add_argument("--summarize", action="store_true",
                    help="aggregate every validation_*.csv under --out-dir; no validation")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if a.summarize:
        return write_campaign_summary(a.out_dir)
    if not a.problem_dir:
        ap.error("pass --problem-dir (or --summarize)")

    if a.systems == "published":
        systems: Optional[List[str]] = list(PUBLISHED_SYSTEMS)
    elif a.systems == "all":
        systems = None
    else:
        systems = [s.strip() for s in a.systems.split(",") if s.strip()]

    folders = Folders()
    worst = 0
    for problem_dir in a.problem_dir:
        if not problem_dir.is_dir():
            print(f"[ERROR] {problem_dir} is not a directory", file=sys.stderr)
            worst = max(worst, 3)
            continue
        try:
            worst = max(worst, validate_problem(problem_dir, a.instance_root, a.out_dir, systems, folders,
                                                a.resume, a.timestep_granularity, a.arrival_delay_metric,
                                                not a.no_connectivity, a.verbose))
        except ValidatorError as exc:
            print(f"[ERROR] {problem_dir}: {exc}", file=sys.stderr)
            worst = max(worst, 3)
        except Exception:
            traceback.print_exc()
            worst = max(worst, 3)
    return worst


if __name__ == "__main__":
    sys.exit(main())
