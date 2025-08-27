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


from mip_model import MIPModel


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
    ) -> None:

        self._graph_path: Optional[Path] = graph_path
        self._sectors_path: Optional[Path] = sectors_path
        self._flights_path: Optional[Path] = flights_path
        self._airports_path: Optional[Path] = airports_path
        self._airplanes_path: Optional[Path] = airplanes_path
        self._airplane_flight_path: Optional[Path] = airplane_flight_path
        self._navaid_sector_path: Optional[Path] = navaid_sector_path

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

                #print(f"{distance}/{airplane_speed_ms} = {duration_in_seconds}")
                #print(f"{duration_in_seconds}/{factor_to_unit_standard} = {duration_in_unit_standards}")

            nx.set_edge_attributes(graph, tmp_edges)


        airport_instance = []

        for row_index in range(self.airports.shape[0]):
            row = self.airports[row_index]
            airport_instance.append(f"airport({row}).")

        airport_instance = "\n".join(airport_instance)

        navaid_sector_lookup = {}
        for row_index in range(self.navaid_sector.shape[0]):
            navaid_sector_lookup[self.navaid_sector[row_index,0]] = self.navaid_sector[row_index, 1]


        # 1.) Create flights matrix (|F|x|T|) --> For easier matrix handling
        converted_instance_matrix, planned_arrival_times = self.instance_to_matrix_vectorized(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)
        #converted_instance_matrix, planned_arrival_times = self.instance_to_matrix(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)

        #np.testing.assert_array_equal(converted_instance_matrix, converted_instance_matrix_2)
        #quit()

        # 2.) Create demand matrix (|R|x|T|)
        system_loads = self.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)

        # 3.) Create capacity matrix (|R|x|T|)
        capacity_time_matrix = self.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity)

        # 4.) Subtract demand from capacity (|R|x|T|)
        capacity_demand_diff_matrix = capacity_time_matrix - system_loads
        # 5.) Create capacity overload matrix

        start_time = time.time()


        converted_instance_matrix = self.build_MIP_model(self.unit_graphs, converted_instance_matrix, capacity_time_matrix)

        end_time = time.time()
        if self._verbosity > 0:
            print(f">> Elapsed solving time: {end_time - start_time}")


        #np.savetxt("01_final_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

        t_init  = self.last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
        t_final = self.last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

        if self._verbosity > 0:
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

            if self._verbosity > 0:
                print(f"Total delay (all flights): {total_delay}")
                print(f"Average delay per flight:  {mean_delay:.2f}")
                print(f"Maximum single-flight delay: {max_delay}")

            print(total_delay)
            np.savetxt(sys.stdout, converted_instance_matrix, delimiter=",", fmt="%i") 

        else:
            if self._verbosity > 0:
                print("Could not find a solution.")


    def build_MIP_model(self,
                unit_graphs,
                converted_instance_matrix: np.ndarray,
                capacity_time_matrix: np.ndarray,
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
        
        mipModel = MIPModel(self.airports, max_time, self._max_explored_vertices, self._seed, self._timestep_granularity, self.verbosity, self._number_threads, navaid_sector_lookup)

        converted_instance_matrix = mipModel.create_model(converted_instance_matrix, capacity_time_matrix, unit_graphs, self.airplanes, max_delay)

        return converted_instance_matrix
    
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
                                    navaid_sector: np.ndarray,
                                    max_time: int,
                                    time_granularity: int,
                                    navaid_sector_lookup,
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

        # --- build navaid -> sector mapping
        # prefer an array map (fast); fall back to fromiter if needed
        navaids_used = flights[:,1]
        try:
            max_nav_id = int(max(navaids_used.max(), max(navaid_sector_lookup.keys())))
            nav2sec = np.full(max_nav_id + 1, fill_value, dtype=np.int64)
            for k, v in navaid_sector_lookup.items():
                nav2sec[int(k)] = int(v)
            sectors = nav2sec[navaids_used]
        except Exception:
            # works even if lookup is a dict with sparse/large keys
            sectors = np.fromiter((int(navaid_sector_lookup[int(x)]) for x in navaids_used),
                                dtype=np.int64, count=navaids_used.size)

        # --- sort by flight, then time (stable contiguous blocks per flight)
        order = np.lexsort((flights[:,2], flights[:,0]))
        f_sorted = flights[order]
        s_sorted = sectors[order]

        fid = f_sorted[:,0]
        t   = f_sorted[:,2]
        sec = s_sorted

        # --- output matrix shape (airplane_id rows, time columns)
        n_rows = int(airplane_flight[:,0].max()) + 1
        max_time_dim = int(max(t.max(), (max_time + 1) * time_granularity))
        out = np.full((n_rows, max_time_dim), fill_value, dtype=sec.dtype)

        # --- group boundaries per flight (contiguous in sorted array)
        u, idx_first, counts = np.unique(fid, return_index=True, return_counts=True)

        # planned arrival times = last time per group (vectorized)
        last_idx = idx_first + counts - 1
        planned_arrival_times = dict(zip(u.tolist(), t[last_idx].tolist()))

        # --- fill per-flight via slices (no per-timestep loops)
        # For each consecutive pair (prev_time -> next_time):
        #   first half  -> prev_sector
        #   second half -> next_sector
        for g, start in enumerate(idx_first):
            end = start + counts[g]

            a_id = int(flight_to_airplane[int(u[g])])
            if a_id < 0:
                # flight has no airplane mapping; skip defensively
                continue

            times = t[start:end]
            secs  = sec[start:end]
            if times.size == 0:
                continue

            # first hop: set the exact event time
            out[a_id, times[0]] = secs[0]

            if times.size >= 2:
                prev_times = times[:-1]
                next_times = times[1:]
                prev_secs  = secs[:-1]
                next_secs  = secs[1:]

                L = next_times - prev_times
                mids = prev_times + (L // 2)

                # slice-assign per segment (loop over segments; no inner time loop)
                for i in range(prev_times.size):
                    s0 = prev_times[i] + 1       # start (exclusive of prev_time)
                    m1 = mids[i] + 1             # first-half end (inclusive) -> slice stop
                    e1 = next_times[i] + 1       # segment end (inclusive) -> slice stop

                    # first half [prev_time+1, mid]
                    if m1 > s0:
                        out[a_id, s0:m1] = prev_secs[i]
                    # second half [mid+1, next_time]
                    if e1 > m1:
                        out[a_id, m1:e1] = next_secs[i]

        return out, planned_arrival_times
   

    def capacity_time_matrix(self,
                            cap: np.ndarray,
                            n_times: int,
                            time_granularity: int) -> np.ndarray:
        """
        cap: shape (N, >=2), capacity in column 1
        returns: (N, n_times)
        """
        N = cap.shape[0]
        T = int(time_granularity)

        # Integer math: base fill + remainder
        capacity = np.asarray(cap[:, 1], dtype=np.int64)
        base = capacity // T                 # per-slot baseline
        rem  = capacity %  T                 # how many +1 to sprinkle

        # Start with the baseline replicated across T columns
        # (broadcast then copy to get a writeable array)
        template = np.broadcast_to(base[:, None], (N, T)).astype(np.int32, copy=True)

        # Fast remainder placement: for each row i, add +1 at
        # columns: step[i] * np.arange(rem[i]), where step = floor(T/rem)
        max_r = int(rem.max())
        if max_r > 0:
            # step is irrelevant where rem==0, but we still fill an array (won't be used)
            step = np.empty_like(rem, dtype=np.int64)
            # Avoid division-by-zero; values where rem==0 are ignored by mask below
            np.floor_divide(T, rem, out=step, where=rem > 0)

            J = np.arange(max_r, dtype=np.int64)                  # 0..max(rem)-1
            mask = J[None, :] < rem[:, None]                      # N x max_r (True only for first rem[i])
            rows2d = np.broadcast_to(np.arange(N)[:, None], (N, max_r))
            cols2d = step[:, None] * J[None, :]

            r_idx = rows2d[mask]
            c_idx = cols2d[mask]

            # Scatter-add the remainders
            np.add.at(template, (r_idx, c_idx), 1)

        # Repeat the base block to cover n_times
        if n_times % T != 0:
            raise ValueError("n_times must be a multiple of time_granularity")
        reps = n_times // T

        cap_mat = np.tile(template, (1, reps))   # (N, n_times)

        #np.savetxt("20250819_cap_mat.csv", cap_mat, delimiter=",",fmt="%i")

        return cap_mat


   
    def bucket_histogram(self, instance_matrix: np.ndarray,
                         sectors: np.ndarray,                # unused, kept for signature compat
                         num_buckets: int,
                         n_times: int,
                         timestep_granularity: int,          # unused, kept for signature compat
                         *,
                         fill_value: int = -1) -> np.ndarray:

        inst = np.asarray(instance_matrix)
        if inst.ndim != 2:
            raise ValueError("instance_matrix must be 2D (flights x time)")
        F, T = inst.shape
        if T != n_times:
            raise ValueError(f"n_times ({n_times}) != instance_matrix.shape[1] ({T})")

        # Early exit if everything is fill_value
        valid = inst != fill_value
        if not valid.any():
            return np.zeros((num_buckets, T), dtype=np.int32)

        # Mark entries (new sector occurrences) at each time:
        # - t = 0: any valid value
        # - t > 0: valid and changed vs previous time
        change = np.zeros_like(valid, dtype=bool)
        change[:, 0] = valid[:, 0]
        if T > 1:
            change[:, 1:] = valid[:, 1:] & (inst[:, 1:] != inst[:, :-1])

        # Gather (sector_id, time_idx) pairs where an entry happens
        sectors_at_entries = inst[change]
        time_idx = np.nonzero(change)[1]  # column indices where change==True

        # Optional safety: ensure sector ids are in [0, num_buckets)
        if sectors_at_entries.size:
            mn = int(sectors_at_entries.min())
            mx = int(sectors_at_entries.max())
            if mn < 0 or mx >= num_buckets:
                raise ValueError(
                    f"sector id(s) out of range [0, {num_buckets}): found min={mn}, max={mx}"
                )

        # Scatter-add 1 for each (sector, time) event
        hist = np.zeros((num_buckets, T), dtype=np.int32)
        np.add.at(hist, (sectors_at_entries, time_idx), 1)

        #np.savetxt("20250819_histogram.csv", hist, delimiter=",",fmt="%i")

        return hist


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Return a fully‑configured :pyclass:`argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        prog="ATFM-NM-Tool",
        description="Highly efficient ATFM problem solver for the network manager - including XAI.",
    )

    parser.add_argument(
        "--graph-path",
        type=Path,
        metavar="FILE",
        help="Location of the graph CSV file.",
    )
    parser.add_argument(
        "--sectors-path",
        type=Path,
        metavar="FILE",
        help="Location of the sector (capacity) CSV file.",
    )
    parser.add_argument(
        "--flights-path",
        type=Path,
        metavar="FILE",
        help="Location of the flights CSV file.",
    )
    parser.add_argument(
        "--airports-path",
        type=Path,
        metavar="FILE",
        help="Location of the airport-vertices CSV file.",
    )
    parser.add_argument(
        "--airplanes-path",
        type=Path,
        metavar="FILE",
        help="Location of the airplanes CSV file.",
    )
    parser.add_argument(
        "--airplane-flight-path",
        type=Path,
        metavar="FILE",
        help="Location of the airplane-flight-assignment CSV file.",
    )
    parser.add_argument(
        "--navaid-sector-path",
        type=Path,
        metavar="FILE",
        help="Location of the navaids-sector assignment CSV file.",
    )

    parser.add_argument(
        "--encoding-path",
        type=Path,
        default=Path("encoding.lp"), 
        metavar="FILE",
        help="Location of the encoding for the optimization problem.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11904657,
        help="Set the random see."
    )
    parser.add_argument(
        "--number-threads",
        type=int,
        default=20,
        help="Number of parallel ASP solving threads."
    )
    parser.add_argument(
        "--timestep-granularity",
        type=int,
        default=1,
        help="Specifies how long one stimulated timestep is (time-step=1h/granularity). So granularity=1 means 1h, granularity=4 means 15 minutes."
    )
    parser.add_argument(
        "--max-explored-vertices",
        type=int,
        default=6,
        help="Maximum vertices explored in parallel - effectively restricts number of explored paths (larger=possibly better solution, but more compute time needed)."
    )
    parser.add_argument(
        "--max-delay-per-iteration",
        type=int,
        default=-1,
        help="Maximum hours of delay per solve iteration explored (larger=more compute time, but faster descent; -1 is automatically fetch  according to max. time steps)."
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=24,
        help="Specifies how many timesteps are one day. "
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity levels (0,1,2)"
    )

    return parser




def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Return parsed command‑line arguments for *argv* (or *sys.argv*)."""
    parser = _build_arg_parser()
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Top‑level script wrapper
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Script entry‑point compatible with both `python -m` and `poetry run`."""
    args = parse_cli(argv)

    app = Main(args.graph_path, args.sectors_path, args.flights_path,
               args.airports_path, args.airplanes_path,
               args.airplane_flight_path, args.navaid_sector_path,
               args.encoding_path,
               args.seed,args.number_threads, args.timestep_granularity,
               args.max_explored_vertices, args.max_delay_per_iteration,
               args.max_time, args.verbosity)
    app.run()


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


 
