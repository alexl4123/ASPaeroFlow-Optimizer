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
import numpy as np

from pathlib import Path
from typing import Any, Final, List, Optional

from solver import Solver, Model


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
        capacity_path: Optional[Path],
        instance_path: Optional[Path],
        airport_vertices_path: Optional[Path],
        encoding_path: Optional[Path],
    ) -> None:
        self._graph_path: Optional[Path] = graph_path
        self._capacity_path: Optional[Path] = capacity_path
        self._instance_path: Optional[Path] = instance_path
        self._airport_vertices_path: Optional[Path] = airport_vertices_path
        self._encoding_path: Optional[Path] = encoding_path

        # Data containers — populated by :pymeth:`load_data`.
        self.graph: Optional[np.ndarray] = None
        self.capacity: Optional[np.ndarray] = None
        self.instance: Optional[np.ndarray] = None
        self.encoding: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def load_data(self) -> None:
        """Load all CSV files provided on the command line."""
        if self._graph_path is not None:
            self.graph = _load_csv(self._graph_path)
        if self._capacity_path is not None:
            self.capacity = _load_csv(self._capacity_path)
        if self._instance_path is not None:
            self.instance = _load_csv(self._instance_path)
        if self._airport_vertices_path is not None:
            self.airport_vertices = _load_csv(self._airport_vertices_path)
        if self._encoding_path is not None:
            print(self._encoding_path)
            with open(self._encoding_path, "r") as file:
                self.encoding = file.read()

    def run(self) -> None:  # noqa: D401 – imperatives okay here
        """Run the application"""
        self.load_data()

        # --- Demonstration output — remove/replace in production ---------
        print("  graph   :", None if self.graph is None else self.graph.shape)
        print("  capacity:", None if self.capacity is None else self.capacity.shape)
        print("  instance:", None if self.instance is None else self.instance.shape)
        print("  airport-vertices:", None if self.airport_vertices is None else self.airport_vertices.shape)
        # -----------------------------------------------------------------

        edges_instance = []

        for row_index in range(self.graph.shape[0]):
            row = self.graph[row_index,: ]
            edges_instance.append(f"sectorEdge({row[0]},{row[1]}).")

        edges_instance = "\n".join(edges_instance)


        airport_instance = []

        for row_index in range(self.airport_vertices.shape[0]):
            row = self.airport_vertices[row_index]
            airport_instance.append(f"airport({row}).")

        airport_instance = "\n".join(airport_instance)

        # |T| = 24 (1h-simulation), or 24*4 (15 minute simulation)
        # |F| = number of flights
        # |R| = number of regions (self.capacity.shape[0])

        # 1.) Create flights matrix (|F|x|T|) --> For easier matrix handling
        converted_instance_matrix = self.instance_to_matrix(self.instance)
        # 2.) Create demand matrix (|R|x|T|)
        system_loads = self.bucket_histogram(converted_instance_matrix, self.capacity.shape[0])
        # 3.) Create capacity matrix (|R|x|T|)
        capacity_time_matrix = self.capacity_time_matrix(self.capacity,system_loads.shape[1])
        # 4.) Subtract demand from capacity (|R|x|T|)
        capacity_demand_diff_matrix = capacity_time_matrix - system_loads
        # 5.) Create capacity overload matrix
        capacity_overload_mask = capacity_demand_diff_matrix < 0

        number_of_conflicts = capacity_overload_mask.sum()
        number_of_conflicts_prev = None
        counter_equal_solutions = 0
        additional_time_increase = 0

        iteration = 0


        while np.any(capacity_overload_mask, where=True):
            print(f"<ITER:{iteration}> REMAINING ISSUES: {str(capacity_overload_mask.sum())}")

            # For efficiency, we can still improve stuff a lot here
            # Like, only provide graph space that is actually necessary 
            # And constrain sectors to relevant section of time
            # ... many other things, like hot-starting clingo solver with multi-shot-solving, etc.
            # maybe iteratively increase max-time (from 24h to sth. else...)

            time_index,bucket_index = self.first_overload(capacity_overload_mask)

            rows = np.flatnonzero(converted_instance_matrix[:, time_index] == bucket_index)

            flights_affected = converted_instance_matrix[rows, :]

            flight_plan_instance = self.flight_plan_strings(rows, flights_affected)
            flight_plan_instance = "\n".join(flight_plan_instance)

            restricted_loads = self.system_loads_restricted(converted_instance_matrix, self.capacity.shape[0], time_idx = time_index, bucket_idx = bucket_index)
 
            capacity_demand_diff_matrix_restricted = capacity_time_matrix - restricted_loads

            # HEURISTIC SELECTION OF COMPLEXITY OF TASK
            """
            if number_of_conflicts_prev is None:
                timed_capacities = self.sector_string_matrix(capacity_demand_diff_matrix_restricted) 
            elif number_of_conflicts < number_of_conflicts_prev:
                timed_capacities = self.sector_string_matrix(capacity_demand_diff_matrix_restricted) 
                counter_equal_solutions = 0
            else:
                counter_equal_solutions += 1

                if counter_equal_solutions > 5:
                    additional_time_increase += 1
                else:
                    timed_capacities = self.sector_string_matrix(capacity_demand_diff_matrix_restricted) 
            """
                
            timed_capacities = self.sector_string_matrix(capacity_demand_diff_matrix_restricted) 

            timed_capacities_instance = '\n'.join(timed_capacities)


            instance = edges_instance + "\n" + timed_capacities_instance + "\n" + flight_plan_instance + "\n" + airport_instance

            encoding = self.encoding

            solver: Model = Solver(encoding, instance)
            model = solver.solve()

            for flight in model.get_flights():
                #print(flight)

                flight_id = int(str(flight.arguments[0]))
                time_id = int(str(flight.arguments[1]))
                position_id = int(str(flight.arguments[2]))

                converted_instance_matrix[flight_id, time_id] = position_id

            # Rerun check if there are still things to solve:
            system_loads = self.bucket_histogram(converted_instance_matrix, self.capacity.shape[0])
            capacity_demand_diff_matrix = capacity_time_matrix - system_loads
            capacity_overload_mask = capacity_demand_diff_matrix < 0


            number_of_conflicts_prev = number_of_conflicts
            number_of_conflicts = capacity_overload_mask.sum()

            iteration += 1



            # RECOMPUTE OVERALL CAPACITY PROBLEMS!
            # FIX IN LATER ITERATION 
            #capacity_overload_mask = capacity_demand_diff_matrix < -100


        # 1.) Create matrix for capacities
        # 2.) Subtract: capacities-system_loads
        # 3.) If anything below 0 -> Capacity problem!
        # 4.) Then iterate over time t:
        #   a.) If at time t, there is a problem with sector x
        #   b.) Consider all flights that have already landed, prior (and including?) to t, as fixed
        #   c.) Ignore all flights that have not been started yet (starting time > t) (STRICTLY GREATER!)
        #   d.) From this, create an ASP instance with a handful of flights that can be re-scheduled accordingly

        #print(system_loads)
        #print(system_loads.shape)
        #np.savetxt("test.csv", converted_instance_matrix,delimiter=",",fmt="%i")
        #np.savetxt("test_loads.csv", capacity_demand_diff_matrix,delimiter=",",fmt="%i")

    def instance_to_matrix(self, inst: np.ndarray,
                        *,
                        fill_value: int = -1,
                        compress: bool = False) -> np.ndarray:
        """
        Convert an *instance* array with shape (n, 3) into a 2-D matrix.

        Parameters
        ----------
        inst
            Each row is (row_id, value, col_id).
        fill_value
            Placeholder put where no entry is defined (defaults to -1).
        compress
            • False  ➜ allocate rows = max(row_id)+1, cols = max(col_id)+1  
            (fastest, but can be large if indices are sparse).

            • True   ➜ map the *unique* row/col labels to consecutive
            indices before building the matrix.  Returns the compact
            matrix plus the label mappings (see below).

        Returns
        -------
        If `compress is False`:
            2-D `np.ndarray` with dtype taken from `inst[:,1]`.
        If `compress is True`:
            (matrix, row_labels, col_labels)
        """
        rows = inst[:, 0].astype(int)
        vals = inst[:, 1]
        cols = inst[:, 2].astype(int)

        out = np.full((rows.max() + 1, cols.max() + 1),
                        fill_value,
                        dtype=vals.dtype)
        out[rows, cols] = vals

        return out


    def bucket_histogram(self, mat: np.ndarray,
                        num_buckets: int,
                        *,
                        fill_value: int = -1) -> np.ndarray:
        """
        Count how many elements occupy each *bucket* at each *time step*.

        Parameters
        ----------
        mat
            2-D array produced by `instance_to_matrix`  
            shape = (num_elements, num_time_steps)
        num_buckets
            Equals `self.capacity.shape[0]`
        fill_value
            The placeholder used for “no assignment” (default −1)

        Returns
        -------
        counts : np.ndarray
            shape = (num_buckets, num_time_steps)  
            `counts[bucket, t]` is the occupancy of *bucket* at time *t*.
        """
        n_elems, n_times = mat.shape

        # ------------------------------------------------------------------
        # 1.  Mask out empty cells (if any)
        # ------------------------------------------------------------------

        # Generate mask of values that differ from fill_value = -1
        valid_mask = mat != fill_value
        if not np.any(valid_mask):                       
            # Shortcut if all cells empty
            return np.zeros((num_buckets, n_times), dtype=int)

        # ------------------------------------------------------------------
        # 2.  Gather bucket IDs and their time indices
        # ------------------------------------------------------------------
        buckets = mat[valid_mask]                        # (K,) bucket id
        # 

        # np.nonzero --> Get indices of non-zero elements of valid_mask (non false)
        # --> with np.nonzero(valid_mask)[1] we take the time indices
        times   = np.nonzero(valid_mask)[1]              # (K,) time index

        # ------------------------------------------------------------------
        # 3.  Vectorised scatter using `np.add.at`
        # ------------------------------------------------------------------
        counts = np.zeros((num_buckets, n_times), dtype=int)

        # Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’.
        # ufunc.at(a, indices, b=None, /)
        # --> Buckets and times specify the indices
        np.add.at(counts, (buckets, times), 1)

        return counts
    
    def capacity_time_matrix(self, cap: np.ndarray,
                            n_times: int,
                            *,
                            sort_by_bucket: bool = False) -> np.ndarray:
        """
        Expand the per-bucket capacity vector to shape (|B|, n_times).

        Parameters
        ----------
        cap
            2-column array ``[bucket_id, capacity_value]`` with shape (|B|, 2).
        n_times
            Number of time steps (usually ``counts.shape[1]``).
        sort_by_bucket
            If *True* (default) the rows are first sorted by *bucket_id*
            so that row *i* corresponds to bucket *i* (0 … |B|–1).

        Returns
        -------
        cap_mat : np.ndarray
            Shape (|B|, n_times) where every row is the bucket’s capacity.
        """
        if sort_by_bucket:
            cap = cap[np.argsort(cap[:, 0])]

        # Extract the capacity column → shape (|B|,)
        cap_vals = cap[:, 1]

        # Broadcast to (|B|, n_times) without an explicit loop
        cap_mat = np.broadcast_to(cap_vals[:, None], (cap_vals.size, n_times))
        # If you *need* a writable array, uncomment the next line
        # cap_mat = cap_mat.copy()

        return cap_mat
    
    def first_overload(self, mask: np.ndarray):
        """
        Return (time, bucket) of the first *True* entry in `mask`
        using lexicographic order (time, bucket).  If none found
        return `None`.

        `mask` shape: (num_buckets, num_times)
                    rows   → bucket IDs
                    columns→ time steps
        """
        # 1. Find the first time step that has *any* overload
        time_candidates = np.flatnonzero(mask.any(axis=0))
        if time_candidates.size == 0:
            return None                # no overloads at all

        t = time_candidates[0]         # earliest time

        # 2. Within that column, find the first bucket
        b = np.flatnonzero(mask[:, t])[0]

        return (t, b)                  # (time, bucket)


    def system_loads_restricted(self, mat: np.ndarray,
                                num_buckets: int,
                                *,
                                time_idx: int,
                                bucket_idx: int,
                                fill_value: int = -1) -> np.ndarray:
        """
        Build a |B| × |T| bucket-occupancy matrix *excluding*:

        • every element sitting in `bucket_idx` at `time_idx`
        • every element whose first scheduled time step is > `time_idx`

        Parameters
        ----------
        mat
            converted_instance_matrix  (shape = |F| × |T|)
        num_buckets
            Equals `self.capacity.shape[0]`
        time_idx
            The reference time step
        bucket_idx
            The reference bucket
        fill_value
            Placeholder that means “empty slot” (default –1)

        Returns
        -------
        loads : np.ndarray
            Bucket-by-time step counts with the requested rows removed.
        """
        n_elems, n_times = mat.shape

        # ---------------------------------------------------------------
        # 1.  Identify *all* rows to drop
        # ---------------------------------------------------------------
        #
        # 1a. Elements located in (time_idx, bucket_idx)
        offending_mask = mat[:, time_idx] == bucket_idx       # shape (|F|,)

        # 1b. Elements that *start* after `time_idx`
        #
        #     First non-fill_value column for each row.
        valid_mask   = mat != fill_value                      # True where real data
        has_any      = valid_mask.any(axis=1)                 # rows that are not empty
        # np.where is ternary operator (np.where(condition,x,y)): result[i] = x[i] if condition[i] else y[i]
        first_nonneg = np.where(has_any,
                                valid_mask.argmax(axis=1),    # first True along row
                                n_times)                      # unused rows ⇒ > time_idx

        late_start_mask = first_nonneg > time_idx

        # Consolidate the exclusion rule
        keep_mask = ~(offending_mask | late_start_mask)       # rows we *keep*

        # Quick exit if nothing survives
        if not np.any(keep_mask):
            return np.zeros((num_buckets, n_times), dtype=int)

        mat_keep = mat[keep_mask]

        # ---------------------------------------------------------------
        # 2.  Re-use the earlier bucket-histogram
        # ---------------------------------------------------------------
        return self.bucket_histogram(mat_keep,
                                num_buckets=num_buckets,
                                fill_value=fill_value)
    
    def sector_string_matrix(self, values: np.ndarray) -> np.ndarray:
        """
        Return a `dtype=object` array of the same shape as *values*
        whose entries are the strings ``"bucket(r,c,v)"``.

        Parameters
        ----------
        values : np.ndarray
            Your numeric matrix (e.g. `capacity_demand_diff_matrix_restricted`).

        Notes
        -----
        *Pure* NumPy — no Python‐level loops — by chaining `np.char.add`.
        """
        rows, cols = np.indices(values.shape)       # same shape as `values`

        # Flatten once to keep the broadcasting simple
        r, c, v = rows.ravel(), cols.ravel(), values.ravel()

        s = np.char.add('sector(', r.astype(str))
        s = np.char.add(np.char.add(s, ','), c.astype(str))
        s = np.char.add(np.char.add(s, ','), v.astype(str))
        s = np.char.add(s, ').')

        #return s.reshape(values.shape)              # back to 2-D
        return s

    def flight_plan_strings(self, rows_original_indices, mat: np.ndarray,
                        *,
                        fill_value: int = -1) -> np.ndarray:
        """
        Build a 1-D array of strings ``"instance(r,c,v)."``
        for every cell whose value ≥ 0 (i.e., `!= fill_value`).

        Parameters
        ----------
        mat
            Your (possibly sliced) `converted_instance_matrix`
            – shape = (n_rows, n_cols)
        fill_value
            Placeholder that marks “empty” entries (default −1)

        Returns
        -------
        out : np.ndarray
            1-D array (dtype=object) where each element is an
            `"instance(r,c,v)."` string.
        """
        # ------------------------------------------------------------------
        # 1.  Locate valid positions
        # ------------------------------------------------------------------
        mask = mat != fill_value                # Boolean matrix
        if not np.any(mask):
            return np.empty(0, dtype=object)    # nothing to report

        r, c = np.nonzero(mask)                 # row/col indices (1-D)

        rows_original_indices = np.asarray(rows_original_indices)
        r = rows_original_indices[r]
        v = mat[mask]                           # corresponding values

        # ------------------------------------------------------------------
        # 2.  Element-wise concatenation (pure NumPy, no Python loop)
        # ------------------------------------------------------------------
        s = np.char.add('flightPlan(', r.astype(str))
        s = np.char.add(np.char.add(s, ','), c.astype(str))
        s = np.char.add(np.char.add(s, ','), v.astype(str))
        s = np.char.add(s, ').')                # final dot

        return s            # shape (K,)  —  K = number of non-empty cells


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
        "--path-graph",
        type=Path,
        default=Path("instances","edges.csv"),  
        metavar="FILE",
        help="Location of the graph CSV file.",
    )
    parser.add_argument(
        "--path-capacity",
        type=Path,
        default=Path("instances","capacity.csv"), 
        metavar="FILE",
        help="Location of the capacity CSV file.",
    )
    parser.add_argument(
        "--path-instance",
        type=Path,
        default=Path("instances","instance_100.csv"), 
        metavar="FILE",
        help="Location of the instance CSV file.",
    )
    parser.add_argument(
        "--airport-vertices-path",
        type=Path,
        default=Path("instances","airport_vertices.csv"), 
        metavar="FILE",
        help="Location of the airport-vertices CSV file.",
    )
    parser.add_argument(
        "--encoding-path",
        type=Path,
        default=Path("encoding.lp"), 
        metavar="FILE",
        help="Location of the encoding for the optimization problem.",
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

    app = Main(args.path_graph, args.path_capacity, args.path_instance, args.airport_vertices_path, args.encoding_path)
    app.run()


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


