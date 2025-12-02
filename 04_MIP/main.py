#!/usr/bin/env python3
"""A minimal, extensible template for loading CSV inputs via the command line.

The script accepts (optional) paths to three CSV files:
    --path-graph     Path to the graph description
    --path-capacity  Path to capacity information
    --path-instance  Path to an instance specification

Each file (when provided) is loaded into a NumPy array for subsequent processing.
Extend the :py:meth:`Main.run` method to add your application logic.
"""
from __future__ import annotations

import argparse
import sys
import time
import os

import networkx as nx
import numpy as np
import math

import pandas as pd

import pickle

import multiprocessing as mp

from pathlib import Path
from typing import Any, Final, List, Optional

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor


from mip_model import MIPModel, MAX, LINEAR, TRIANGULAR
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_csv(path: Path, *, dtype: Any = int, delimiter: str = ",") -> np.ndarray:
    """Return *path* as a NumPy array.

    Parameters
    ----------
    path
        Location of the CSV file on disk.
    dtype, delimiter
        Passed straight through to :pyfunc:`numpy.loadtxt`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *path* cannot be parsed as a numeric CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=1)
    except ValueError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc

# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class Main:
    """Application entry‑point.

   """

    def __init__(
        self,
        graph_path: Optional[Path],
        sectors_path: Optional[Path],
        flights_path: Optional[Path],
        airports_path: Optional[Path],
        airplanes_path: Optional[Path],
        airplane_flight_path: Optional[Path],
        navaid_sector_path: Optional[Path],
        encoding_path: Optional[Path],
        seed: Optional[int],
        number_threads: Optional[int],
        timestep_granularity: Optional[int],
        max_explored_vertices: Optional[int],
        max_delay_per_iteration: Optional[int],
        max_time: Optional[int],
        verbosity: Optional[int],
        composite_sector_function,
        sector_capacity_factor,
    ) -> None:

        self._graph_path: Optional[Path] = graph_path
        self._sectors_path: Optional[Path] = sectors_path
        self._flights_path: Optional[Path] = flights_path
        self._airports_path: Optional[Path] = airports_path
        self._airplanes_path: Optional[Path] = airplanes_path
        self._airplane_flight_path: Optional[Path] = airplane_flight_path
        self._navaid_sector_path: Optional[Path] = navaid_sector_path

        self._composite_sector_function = composite_sector_function
        self._sector_capacity_factor = sector_capacity_factor

        self._encoding_path: Optional[Path] = encoding_path

        self._seed: Optional[int] = seed
        self._number_threads: Optional[int] = number_threads
        self._timestep_granularity: Optional[int] = timestep_granularity
        self._max_time: Optional[int] = max_time
        self.verbosity: Optional[int] = verbosity

        self._max_explored_vertices: Optional[int] = max_explored_vertices
        self._max_delay_per_iteration: Optional[int] = max_delay_per_iteration

        # Data containers — populated by :pymeth:`load_data`.
        self.graph: Optional[np.ndarray] = None
        self.sectors: Optional[np.ndarray] = None
        self.flights: Optional[np.ndarray] = None
        self.encoding: Optional[np.ndarray] = None

        self.nearest_neighbors_lookup = {}



    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def load_data(self) -> None:
        """Load all CSV files provided on the command line."""
        if self._graph_path is not None:
            self.graph = _load_csv(self._graph_path)
        if self._sectors_path is not None:
            self.sectors = _load_csv(self._sectors_path)
        if self._flights_path is not None:
            self.flights = _load_csv(self._flights_path)
        if self._airports_path is not None:
            self.airports = _load_csv(self._airports_path)
        if self._airplanes_path is not None:
            self.airplanes = _load_csv(self._airplanes_path)
        if self._airplane_flight_path is not None:
            self.airplane_flight = _load_csv(self._airplane_flight_path)
        if self._navaid_sector_path is not None:
            self.navaid_sector = _load_csv(self._navaid_sector_path)

        if self._encoding_path is not None:
            with open(self._encoding_path, "r") as file:
                self.encoding = file.read()



    def run(self) -> None:  # noqa: D401 – imperatives okay here
        """Run the application"""
        self.load_data()
        
        if self.verbosity > 0:
            # --- Demonstration output — remove/replace in production ---------
            print("  graph   :", None if self.graph is None else self.graph.shape)
            print("  capacity:", None if self.sectors is None else self.sectors.shape)
            print("  instance:", None if self.flights is None else self.flights.shape)
            print("  airport-vertices:", None if self.airports is None else self.airports.shape)
            # -----------------------------------------------------------------

        sources = self.graph[:,0]
        targets = self.graph[:,1]
        dists = self.graph[:,2]
        self.networkx_navpoint_graph = nx.Graph()
        self.networkx_navpoint_graph.add_weighted_edges_from(zip(sources, targets, dists))

        self.unit_graphs = {}

        different_speeds = list(set(list(self.airplanes[:,1])))
        for cur_airplane_speed in different_speeds:

            self.unit_graphs[cur_airplane_speed] = self.networkx_navpoint_graph.copy()
            graph = self.unit_graphs[cur_airplane_speed]

            self.nearest_neighbors_lookup[cur_airplane_speed] = {}

            tmp_edges = {}

            for edge in graph.edges(data=True):

                distance = edge[2]["weight"]
                # CONVERT AIRPLANE SPEED TO m/s
                airplane_speed_ms = cur_airplane_speed * 0.51444
                duration_in_seconds = distance/airplane_speed_ms
                factor_to_unit_standard = 3600.00 / float(self._timestep_granularity)
                duration_in_unit_standards = math.ceil(duration_in_seconds / factor_to_unit_standard)

                duration_in_unit_standards = max(duration_in_unit_standards, 1)

                tmp_edges[(edge[0],edge[1])] = {"weight":duration_in_unit_standards}

            nx.set_edge_attributes(graph, tmp_edges)



        airport_instance = []

        for row_index in range(self.airports.shape[0]):
            row = self.airports[row_index]
            airport_instance.append(f"airport({row}).")

        airport_instance = "\n".join(airport_instance)

        navaid_sector_lookup = {}
        for row_index in range(self.navaid_sector.shape[0]):
            navaid_sector_lookup[self.navaid_sector[row_index,0]] = self.navaid_sector[row_index, 1]

        # 0.) Create navpaid sector time assignment (|R|XT):
        navaid_sector_time_assignment = MIPModel.create_initial_navpoint_sector_assignment(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity)

        # 1.) Create flights matrix (|F|x|T|) --> For easier matrix handling
        #converted_instance_matrix, planned_arrival_times = self.instance_to_matrix_vectorized(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)
        converted_instance_matrix, planned_arrival_times = self.instance_to_matrix_vectorized(self.flights, self.airplane_flight, navaid_sector_time_assignment.shape[1], self._timestep_granularity, navaid_sector_time_assignment)
        #converted_instance_matrix, planned_arrival_times = self.instance_to_matrix(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)

        #np.testing.assert_array_equal(converted_instance_matrix, converted_instance_matrix_2)
        #quit()

        # 2.) Create demand matrix (|R|x|T|)
        system_loads = MIPModel.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)

        # 3.) Create capacity matrix (|R|x|T|)
        capacity_time_matrix = MIPModel.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self._sector_capacity_factor,
                                                                    composite_sector_function=self._composite_sector_function)

        # 4.) Subtract demand from capacity (|R|x|T|)
        capacity_demand_diff_matrix = capacity_time_matrix - system_loads
        # 5.) Create capacity overload matrix

        start_time = time.time()
        
        original_converted_instance_matrix = converted_instance_matrix.copy()


        converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment = self.build_MIP_model(self.unit_graphs, converted_instance_matrix, capacity_time_matrix, planned_arrival_times, self.airplane_flight)

        end_time = time.time()
        if self.verbosity > 0:
            print(f">> Elapsed solving time: {end_time - start_time}")

        self.converted_instance_matrix = converted_instance_matrix
        self.converted_navpoint_matrix = converted_navpoint_matrix
        self.navaid_sector_time_assignment = navaid_sector_time_assignment

        #np.savetxt("01_final_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

        t_init  = self.last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
        t_final = self.last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

        if self.verbosity > 0:
            print("<<<<<<<<<<<<<<<<----------------->>>>>>>>>>>>>>>>")
            print("                  FINAL RESULTS")
            print("<<<<<<<<<<<<<<<<----------------->>>>>>>>>>>>>>>>")

        if t_final is not None:
            # --- 3. compute delays --------------------------------------------------------
            # Flights that disappear completely (-1 in *both* files) get a delay of 0
            delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

            # --- 4. aggregate in whichever way you need -----------------------------------
            total_delay  = delay.sum()
            mean_delay   = delay.mean()
            max_delay    = delay.max()
            #per_flight   = delay.tolist()

            if self.verbosity > 0:
                print(f"Total delay (all flights): {total_delay}")
                print(f"Average delay per flight:  {mean_delay:.2f}")
                print(f"Maximum single-flight delay: {max_delay}")

            print(total_delay)
            np.savetxt("tmp_new.csv", converted_instance_matrix, delimiter=",", fmt="%i") 
            np.savetxt("tmp_orig.csv", original_converted_instance_matrix, delimiter=",", fmt="%i") 

        else:
            if self.verbosity > 0:
                print("Could not find a solution.")


    def build_MIP_model(self,
                unit_graphs,
                converted_instance_matrix: np.ndarray,
                capacity_time_matrix: np.ndarray,
                planned_arrival_times,
                airplane_flight,
                ):
        """
        Split the candidate rows into *n_proc* equally‑sized chunks
        (≤ max_rows each) and build one ``OptimizeFlights`` instance per chunk.
        """

        navaid_sector_lookup = {}
        for row_index in range(self.navaid_sector.shape[0]):
            navaid_sector_lookup[self.navaid_sector[row_index,0]] = self.navaid_sector[row_index, 1]

        max_time = max(self._max_time, converted_instance_matrix.shape[1])

        max_delay = 24
        
        mipModel = MIPModel(self.sectors, self.airports, max_time, self._max_explored_vertices, self._seed, self._timestep_granularity, self.verbosity, self._number_threads, navaid_sector_lookup, self._composite_sector_function, self._sector_capacity_factor)

        converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix = mipModel.create_model(converted_instance_matrix, capacity_time_matrix, unit_graphs, self.airplanes, max_delay, planned_arrival_times, airplane_flight, self.flights)

        return converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix
    
    def flight_spans_contiguous(self, matrix: np.ndarray, *, fill_value: int = -1):
        """
        For each row (flight), find the first contiguous block of non-`fill_value`
        entries and return (start, stop, duration), where `stop` is exclusive.

        Assumes each flight occupies one contiguous block; if multiple blocks
        exist, only the *first* is used. Rows with no non-`fill_value` data get
        start = stop = -1 and duration = 0.
        """
        valid = matrix != fill_value              # shape (F, T)
        F, T = valid.shape

        # Pad False on both sides so every run has a start & end transition.
        padded = np.zeros((F, T + 2), dtype=bool)
        padded[:, 1:-1] = valid

        left  = padded[:, :-1]
        right = padded[:, 1:]

        starts_mask = (~left) & right             # False->True transitions
        ends_mask   = left & (~right)             # True->False transitions

        has_run = starts_mask.any(axis=1)         # at least one active cell?

        # argmax returns first True; safe because has_run tells us if any exist.
        start = np.argmax(starts_mask, axis=1)    # index in 0..T  (T=exclusive)
        stop  = np.argmax(ends_mask,   axis=1)    # index in 0..T

        start = np.where(has_run, start, -1)
        stop  = np.where(has_run,  stop,  -1)

        duration = np.where(has_run, stop - start, 0)

        return start, stop, duration


    # --- 2. helper: last non--1 position for every row, fully vectorised ----------
    def last_valid_pos(self, arr: np.ndarray) -> np.ndarray:
        """
        Return a 1-D array with, for every row in `arr`, the **last** column index
        whose value is not -1.  If a row is all -1, we return -1 for that flight.
        """
        # True where value ≠ -1
        mask = arr != -1                                        # same shape as arr

        if not np.any(mask, where=True):
            return None

        # Reverse columns so that the *first* True along axis=1 is really the last
        # in the original orientation
        reversed_first = np.argmax(mask[:, ::-1], axis=1)

        # If the whole row was False, argmax returns 0.  Detect that case:
        no_valid = ~mask.any(axis=1)                            # shape (|I|,)

        # Convert “position in reversed array” back to real column index
        last_pos = arr.shape[1] - 1 - reversed_first            # shape (|I|,)
        last_pos[no_valid] = -1                                 # sentinel value

        return last_pos.astype(np.int64)

   
    def compute_distance_matrix(
        self,
        *,
        directed: bool = False,
        as_int: bool = True,
        remap: bool = True,
    ) -> np.ndarray:
        """
        Return the |V| × |V| matrix of shortest-path lengths (all edges = 1).

        Parameters
        ----------
        directed : bool, default=False
            Treat the graph as directed.  If ``False`` the edge
            list is symmetrised internally.
        as_int : bool, default=False
            Convert the result to ``int`` and replace unreachable
            pairs by ``-1``.  SciPy returns ``float`` with
            ``np.inf`` otherwise.
        remap : bool, default=True
            If vertex IDs are not contiguous ``0 … |V|-1``,
            remap them first.  This costs an ``O(|E| log |E|)``
            sort but guarantees a compact matrix.

        Returns
        -------
        np.ndarray
            Dense distance matrix.  Entry ``D[i, j]`` equals the
            length (in edges) of the shortest path from *i* to *j*
            or –1/np.inf if no path exists.

        Notes
        -----
        * Memory usage is ``O(|V|²)`` for the result, so for very large
          graphs consider streaming single-source BFS instead.
        * Runs in roughly ``O(|E|)`` because the graph is unweighted (double check this).
        """
        # -------------------------- 0. Input hygiene --------------------- #
        edges = self.graph
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("graph must be a (|E|, 2) edge list")

        if remap:
            # Compress vertex IDs to 0…n-1 to avoid huge, sparse matrices.
            unique_ids, idx = np.unique(edges, return_inverse=True)
            edges = idx.reshape(-1, 2)
            n_vertices = unique_ids.size
        else:
            n_vertices = edges.max() + 1

        # -------------------------- 1. Adjacency ------------------------- #
        data = np.ones(len(edges), dtype=np.int8)
        row, col = edges.T
        A = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))

        if not directed:
            A = A + A.T  # mirror the edges

        # -------------------------- 2. BFS ------------------------------- #
        dist = shortest_path(
            A,
            directed=directed,
            unweighted=True,
            return_predecessors=False,
        )

        # -------------------------- 3. Post-process ---------------------- #
        if as_int:
            dist = np.where(np.isinf(dist), -1, dist.astype(np.int32))

        return dist

    def instance_to_matrix_vectorized(self,
                                    flights: np.ndarray,
                                    airplane_flight: np.ndarray,
                                    max_time: int,
                                    time_granularity: int,
                                    navaid_sector_time_assignment: np.ndarray,
                                    *,
                                    fill_value: int = -1,
                                    compress: bool = False):
        """
        Vectorized/semi-vectorized rewrite.
        Dynamic sector assignment via navaid_sector_time_assignment (shape N x T):
        sector_at_event = navaid_sector_time_assignment[navaid_id, time]
        Assumes IDs are non-negative ints (reasonably dense).
        """

        # --- ensure integer views without copies where possible
        flights = flights.astype(np.int64, copy=False)
        airplane_flight = airplane_flight.astype(np.int64, copy=False)

        # --- build flight -> airplane mapping (array is fastest if IDs are dense)
        fid_map_max = int(max(flights[:, 0].max(), airplane_flight[:, 1].max()))
        flight_to_airplane = np.full(fid_map_max + 1, -1, dtype=np.int64)
        flight_to_airplane[airplane_flight[:, 1]] = airplane_flight[:, 0]

        # --- sort by flight, then time (stable contiguous blocks per flight)
        order = np.lexsort((flights[:, 2], flights[:, 0]))
        f_sorted = flights[order]

        fid = f_sorted[:, 0]
        nav = f_sorted[:, 1]
        t   = f_sorted[:, 2]

        # --- dynamic navaid -> sector from (N x T) matrix using pairwise advanced indexing
        N, T = navaid_sector_time_assignment.shape
        if nav.size:
            nav_min, nav_max = int(nav.min()), int(nav.max())
            t_min, t_max     = int(t.min()),   int(t.max())
            if nav_min < 0 or nav_max >= N:
                raise ValueError(
                    f"Navpoint id out of bounds: got range [{nav_min}, {nav_max}], matrix has N={N}."
                )
            if t_min < 0 or t_max >= T:
                raise ValueError(
                    f"Time index out of bounds: got range [{t_min}, {t_max}], matrix has T={T}."
                )

        sec = navaid_sector_time_assignment[nav, t]  # 1D array aligned with f_sorted

        # --- output matrix shape (airplane_id rows, time columns)
        n_rows = int(airplane_flight[:, 1].max()) + 1
        out = np.full((n_rows, int(max_time)), fill_value, dtype=sec.dtype if sec.size else np.int64)

        # --- group boundaries per flight (contiguous in sorted array)
        if fid.size == 0:
            return out, {}

        u, idx_first, counts = np.unique(fid, return_index=True, return_counts=True)

        # planned arrival times = last time per group (vectorized)
        last_idx = idx_first + counts - 1
        planned_arrival_times = dict(zip(u.tolist(), t[last_idx].tolist()))

        # --- fill per-flight via slices (no per-timestep inner loops)
        for g, start in enumerate(idx_first):
            end = start + counts[g]

            flight_index = g

            if flight_index < 0:
                # flight has no airplane mapping; skip defensively
                continue

            times = t[start:end]
            secs  = sec[start:end]
            if times.size == 0:
                continue

            # set the exact event time
            out[flight_index, times[0]] = secs[0]

            if times.size >= 2:
                prev_times = times[:-1]
                next_times = times[1:]
                prev_secs  = secs[:-1]
                next_secs  = secs[1:]

                L = next_times - prev_times
                mids = prev_times + (L // 2)

                # slice-assign per segment
                for i in range(prev_times.size):
                    s0 = prev_times[i] + 1       # start (exclusive)
                    m1 = mids[i] + 1             # first-half end (inclusive) -> slice stop
                    e1 = next_times[i] + 1       # segment end (inclusive) -> slice stop

                    # first half [prev_time+1, mid]
                    if m1 > s0:
                        out[flight_index, s0:m1] = prev_secs[i]
                    # second half [mid+1, next_time]
                    if e1 > m1:
                        out[flight_index, m1:e1] = next_secs[i]

        # Optional: compress width to actually-used time if requested
        if compress and out.shape[1] > 0:
            used_cols = np.any(out != fill_value, axis=0)
            if used_cols.any():
                last_used = np.flatnonzero(used_cols)[-1] + 1
                out = out[:, :last_used]
            else:
                out = out[:, :1]  # keep at least one column

        return out, planned_arrival_times



    def create_initial_navpoint_sector_assignment(self,
                                    flights: np.ndarray,
                                    airplane_flight: np.ndarray,
                                    navaid_sector: np.ndarray,
                                    max_time: int,
                                    time_granularity: int,
                                    *,
                                    fill_value: int = -1,
                                    compress: bool = False):
        """
        Vectorized/semi-vectorized rewrite instance_to_matrix.
        Assumes IDs are non-negative ints (reasonably dense).
        """

        # --- ensure integer views without copies where possible
        flights = flights.astype(np.int64, copy=False)
        airplane_flight = airplane_flight.astype(np.int64, copy=False)

        # --- build flight -> airplane mapping (array is fastest if IDs are dense)
        fid_map_max = int(max(flights[:,0].max(), airplane_flight[:,1].max()))
        flight_to_airplane = np.full(fid_map_max + 1, -1, dtype=np.int64)
        flight_to_airplane[airplane_flight[:,1]] = airplane_flight[:,0]

        # --- sort by flight, then time (stable contiguous blocks per flight)
        order = np.lexsort((flights[:,2], flights[:,0]))
        f_sorted = flights[order]

        t   = f_sorted[:,2]

        # --- output matrix shape (airplane_id rows, time columns)
        max_time_dim = int(max(t.max() + 1, (max_time + 1) * time_granularity))

        sectors = navaid_sector[:, 1]                      # shape (N,)
        return np.repeat(sectors[:, None], max_time_dim, axis=1)  # shape (N, T)
    
# ---------------------------------------------------------------------------
# CLI utilities (with config + bundle directory support)
# ---------------------------------------------------------------------------

import argparse, json
from pathlib import Path
from typing import Optional, List, Dict

DEFAULT_FILENAMES = {
    "graph_path":              "graph_edges.csv",
    "sectors_path":            "sectors.csv",
    "flights_path":            "flights.csv",
    "airports_path":           "airports.csv",
    "airplanes_path":          "airplanes.csv",
    "airplane_flight_path":    "airplane_flight_assignment.csv",
    "navaid_sector_path":      "navaid_sector_assignment.csv",
    "encoding_path":           "encoding.lp",
    "wandb_api_key":               "wandb.key",
}

def _cfg_get(cfg: Dict, key: str, default=None):
    """Get key from cfg with hyphen/underscore tolerance."""
    if key in cfg: return cfg[key]
    alt = key.replace("-", "_")
    if alt in cfg: return cfg[alt]
    alt2 = key.replace("_", "-")
    if alt2 in cfg: return cfg[alt2]
    return default

def _preparse(argv: Optional[List[str]]):
    """Parse only --config and --data-dir early, so we can load config defaults."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    return p.parse_known_args(argv)

def _build_arg_parser(cfg: Dict) -> argparse.ArgumentParser:
    """Build parser with defaults coming from cfg (if present)."""
    def C(key, default=None): return _cfg_get(cfg, key, default)

    parser = argparse.ArgumentParser(
        prog="ATFM-NM-Tool",
        description="Highly efficient ATFM problem solver for the network manager - including XAI.",
    )

    # New: config + bundle directory
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON config file whose values serve as defaults (overridden by CLI)."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=C("data-dir", None),
        help="Directory containing the 7 standard optimizer CSVs. "
             "Any individual --*-path not provided will default to this directory + default filename."
    )

    # Individual paths (CLI overrides config)
    parser.add_argument("--graph-path",           type=Path, metavar="FILE", default=C("graph-path", None),
                        help="Location of the graph CSV file (graph_edges.csv).")
    parser.add_argument("--sectors-path",         type=Path, metavar="FILE", default=C("sectors-path", None),
                        help="Location of the sector (capacity) CSV file.")
    parser.add_argument("--flights-path",         type=Path, metavar="FILE", default=C("flights-path", None),
                        help="Location of the flights CSV file.")
    parser.add_argument("--airports-path",        type=Path, metavar="FILE", default=C("airports-path", None),
                        help="Location of the airport-vertices CSV file.")
    parser.add_argument("--airplanes-path",       type=Path, metavar="FILE", default=C("airplanes-path", None),
                        help="Location of the airplanes CSV file.")
    parser.add_argument("--airplane-flight-path", type=Path, metavar="FILE", default=C("airplane-flight-path", None),
                        help="Location of the airplane-flight-assignment CSV file.")
    parser.add_argument("--navaid-sector-path",   type=Path, metavar="FILE", default=C("navaid-sector-path", None),
                        help="Location of the navaids-sector assignment CSV file.")

    # Results saving
    parser.add_argument(
        "--save-results",
        type=str,
        default=str(C("save-results", "true")),
        help="true/false: save optimizer result matrices to disk (default: true).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(C("results-root", "experiment_output")),
        help="Root folder to store result matrices (default: experiment_output).",
    )
    parser.add_argument(
        "--results-format",
        type=str,
        choices=["csv","csv.gz","npz"],
        default=C("results-format", "csv"),
        help="File format for matrices: 'csv' (default, uncompressed), 'csv.gz', or 'npz'.",
    )

    parser.add_argument("--composite-sector-function", type=str, default=str(C("composite-sector-function", "max")),
                        help="Defines the function of the composite sector - available: max, triangular, linear")

    # Encoding + knobs
    parser.add_argument("--encoding-path", type=Path, default=Path(C("encoding-path", DEFAULT_FILENAMES["encoding_path"])),
                        metavar="FILE", help="Location of the encoding for the optimization problem.")
    parser.add_argument("--seed", type=int, default=int(C("seed", 11904657)),
                        help="Set the random seed.")
    parser.add_argument("--number-threads", type=int, default=int(C("number-threads", 20)),
                        help="Number of parallel ASP solving threads.")
    parser.add_argument("--timestep-granularity", type=int, default=int(C("timestep-granularity", 1)),
                        help="Granularity: 1=1h, 4=15min, etc.")
    parser.add_argument("--max-explored-vertices", type=int, default=int(C("max-explored-vertices", 6)),
                        help="Max vertices explored in parallel.")
    parser.add_argument("--max-delay-per-iteration", type=int, default=int(C("max-delay-per-iteration", -1)),
                        help="Max hours of delay per iteration (−1 = auto).")
    parser.add_argument("--max-time", type=int, default=int(C("max-time", 24)),
                        help="Number of timesteps for one day.")
    parser.add_argument("--verbosity", type=int, default=int(C("verbosity", 0)),
                        help="Verbosity levels (0,1,2).")
    parser.add_argument("--sector-capacity-factor", type=int, default=int(C("sector-capacity-factor", 6)),
                        help="NOT SUPPORTED IN MIP MODEL.")

    # WANDB:
    parser.add_argument("--wandb-enabled", type=str, default=str(C("wandb-enabled", "false")),
                        help="true/false: If enabled, trace run on wandb.")
    parser.add_argument("--wandb-experiment-name-prefix", type=str, default=str(C("wandb-experiment-name-prefix","")),
                        help="Defines the wandb prefix name for tracing experiments (only used when wandb is enabled).")
    parser.add_argument("--wandb-experiment-name-suffix", type=str, default=str(C("wandb-experiment-name-suffix","")),
                        help="Defines the wandb suffix name for tracing experiments (only used when wandb is enabled).")
    parser.add_argument("--wandb-api-key-path", type=Path, default=Path(C("wandb-api-key-path", DEFAULT_FILENAMES["wandb_api_key"])),
                        metavar="FILE", help="Location of the wandb API key file (only searched when wandb is enabled).")
    parser.add_argument("--wandb-project", type=str, default=str(C("wandb-project", "ASPaeroFlow")),
                        help="Weights & Biases project name (default: ASPaeroFlow).")
    parser.add_argument("--wandb-entity", type=str, default=C("wandb-entity", None),
                        help="Weights & Biases entity (username or team/organization). Leave empty to use your default entity.")



    return parser

def _apply_data_dir_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """For any missing *-path, use data_dir / default_filename."""
    if not args.data_dir:
        return args
    base = args.data_dir

    def fill(cur: Optional[Path], fname_key: str) -> Path:
        return cur if cur else (base / DEFAULT_FILENAMES[fname_key])

    args.graph_path           = fill(args.graph_path,           "graph_path")
    args.sectors_path         = fill(args.sectors_path,         "sectors_path")
    args.flights_path         = fill(args.flights_path,         "flights_path")
    args.airports_path        = fill(args.airports_path,        "airports_path")
    args.airplanes_path       = fill(args.airplanes_path,       "airplanes_path")
    args.airplane_flight_path = fill(args.airplane_flight_path, "airplane_flight_path")
    args.navaid_sector_path   = fill(args.navaid_sector_path,   "navaid_sector_path")
    # encoding_path already has a default; leave as-is

    return args

def _validate_inputs(args: argparse.Namespace):
    missing = []
    for k in [
        "graph_path","sectors_path","flights_path","airports_path",
        "airplanes_path","airplane_flight_path","navaid_sector_path","encoding_path"
    ]:
        p: Path = getattr(args, k)

        if p is None or not Path(p).exists():
            missing.append((k, str(p)))
    if missing:
        lines = ["Input files not found:"]
        lines += [f"  - {k}: {v}" for k, v in missing]
        raise FileNotFoundError("\n".join(lines))

def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI with priority: CLI > config > built-in defaults."""
    # 1) preparse to get --config
    pre, _ = _preparse(argv)
    cfg = {}
    if pre.config and pre.config.exists():
        with open(pre.config, "r") as fh:
            cfg = json.load(fh) or {}

    # allow config to set a default data-dir as well
    if pre.data_dir is None:
        cfg_data_dir = _cfg_get(cfg, "data-dir", None)
        if cfg_data_dir:
            pre.data_dir = Path(cfg_data_dir)

    # 2) build the full parser with cfg-derived defaults
    parser = _build_arg_parser(cfg)
    args = parser.parse_args(argv)

    # Keep the config path in args for traceability
    if args.config is None and pre.config:
        args.config = pre.config

    # 3) If data-dir provided (CLI or config), auto-fill missing file paths
    if args.data_dir is None and pre.data_dir:
        args.data_dir = pre.data_dir
    args = _apply_data_dir_defaults(args)

    # 4) Final validation
    _validate_inputs(args)

    # normalize booleans
    def _str2bool(v):
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        return s in ("1","true","t","yes","y","on")
    args.save_results = _str2bool(args.save_results)
    args.wandb_enabled = _str2bool(args.wandb_enabled)

    return args

def _derive_output_name(args: argparse.Namespace) -> str:
    """
    Use the last segment of --data-dir as the experiment name, e.g. 0000763_SEED42.
    Assumes --data-dir is provided (as per user workflow).
    """
    if args.data_dir:
        return Path(args.data_dir).name
    # Fallback: try to use flights' parent directory name
    if args.flights_path:
        return Path(args.flights_path).parent.name or "RESULTS"
    return "RESULTS"

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_npz(path: Path, arr, key: str):
    a = np.asarray(arr)
    np.savez_compressed(path, **{key: a})

def _save_csv_gz(path: Path, arr):
    a = np.asarray(arr)
    # Write as integer CSV (or fall back to float if needed)
    # We avoid pandas for speed/dep-minimization; numpy.savetxt with gzip works well.
    import gzip
    fmt = "%d" if a.dtype.kind in ("i","u","b") else "%g"
    with gzip.open(path, "wt", encoding="utf-8") as gz:
        np.savetxt(gz, a, fmt=fmt, delimiter=",")

def _save_csv(path: Path, arr):
    a = np.asarray(arr)
    fmt = "%d" if a.dtype.kind in ("i","u","b") else "%g"
    np.savetxt(path, a, fmt=fmt, delimiter=",")

def _save_results(args: argparse.Namespace, app) -> None:
    """
    Persist the three result matrices if present on `app`:
      - navaid_sector_time_assignment  (|N| x |T|)
      - converted_instance_matrix      (|F| x |T|)
      - converted_navpoint_matrix      (|F| x |T|)
    """
    out_name = _derive_output_name(args)
    out_dir  = _ensure_dir(Path(args.results_root) / out_name)

    mats = {
        "navaid_sector_time_assignment": getattr(app, "navaid_sector_time_assignment", None),
        "converted_instance_matrix":     getattr(app, "converted_instance_matrix", None),
        "converted_navpoint_matrix":     getattr(app, "converted_navpoint_matrix", None),
    }

    saved = {}
    for key, val in mats.items():
        if val is None:
            continue
        if args.results_format == "npz":
            _save_npz(out_dir / f"{key}.npz", val, key)
        elif args.results_format == "csv.gz":
            _save_csv_gz(out_dir / f"{key}.csv.gz", val)
        else:  # "csv"
            _save_csv(out_dir / f"{key}.csv", val)

        a = np.asarray(val)
        saved[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}

    # Write a small manifest
    manifest = {
        "saved": saved,
        "format": args.results_format,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "seed": args.seed,
            "timestep_granularity": args.timestep_granularity,
        }
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if args.verbosity > 0:
        print(f"[✓] Saved results → {out_dir}")


# ---------------------------------------------------------------------------
# Top‑level script wrapper
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Script entry‑point compatible with both `python -m` and `poetry run`."""
    args = parse_cli(argv)
 
    experiment_name = _derive_output_name(args)
    # W&B setup (optional)
    run = None
    wandb_log = None
    if args.wandb_enabled:
        try:
            import wandb  # installed by user
        except ImportError as e:
            raise ImportError("wandb is not installed, but --wandb-enabled was set to true.") from e
        api_key_path: Path = args.wandb_api_key_path
        if not api_key_path.exists():
            raise FileNotFoundError(f"W&B API key file not found: {api_key_path}")
        key = api_key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"W&B API key file is empty: {api_key_path}")
        os.environ["WANDB_API_KEY"] = key
        wandb.login(key=key, relogin=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity if args.wandb_entity else None),
            name=f"{args.wandb_experiment_name_prefix}{experiment_name}{args.wandb_experiment_name_suffix}",
            config={
                "timestep_granularity": args.timestep_granularity,
                "max_explored_vertices": args.max_explored_vertices,
                "number_threads": args.number_threads,
                "max_delay_per_iteration": args.max_delay_per_iteration,
                "seed": args.seed,
                "max_time": args.max_time,
            },
        )
        wandb_log = run.log

    composite_sector_function = args.composite_sector_function.lower()
    if composite_sector_function not in [MAX, LINEAR, TRIANGULAR]:
        raise Exception(f"Specified composite sector function {composite_sector_function} not in {[MAX,LINEAR,TRIANGULAR]}")

    app = Main(args.graph_path, args.sectors_path, args.flights_path,
               args.airports_path, args.airplanes_path,
               args.airplane_flight_path, args.navaid_sector_path,
               args.encoding_path,
               args.seed,args.number_threads, args.timestep_granularity,
               args.max_explored_vertices, args.max_delay_per_iteration,
               args.max_time, args.verbosity,
               composite_sector_function, args.sector_capacity_factor)
    app.run()

    if run is not None:
        run.finish()

    # Save results if requested
    if args.save_results:
        _save_results(args, app)




if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


 
