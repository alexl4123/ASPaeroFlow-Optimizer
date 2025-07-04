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
from pathlib import Path
from typing import Any, Final, List, Optional

import numpy as np

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
    ) -> None:
        self._graph_path: Optional[Path] = graph_path
        self._capacity_path: Optional[Path] = capacity_path
        self._instance_path: Optional[Path] = instance_path

        # Data containers — populated by :pymeth:`load_data`.
        self.graph: Optional[np.ndarray] = None
        self.capacity: Optional[np.ndarray] = None
        self.instance: Optional[np.ndarray] = None

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

    def run(self) -> None:  # noqa: D401 – imperatives okay here
        """Run the application"""
        self.load_data()

        # --- Demonstration output — remove/replace in production ---------
        print("  graph   :", None if self.graph is None else self.graph.shape)
        print("  capacity:", None if self.capacity is None else self.capacity.shape)
        print("  instance:", None if self.instance is None else self.instance.shape)
        # -----------------------------------------------------------------


        # 1-hour steps
        time_steps = 24 * 1

        converted_instance_matrix = self.instance_to_matrix(self.instance)


        system_loads = self.bucket_histogram(converted_instance_matrix, self.capacity.shape[0])

        print(system_loads)
        np.savetxt("test_loads.csv", system_loads,delimiter=",",fmt="%i")





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
        default=Path("instances/capacity.csv"), 
        metavar="FILE",
        help="Location of the capacity CSV file.",
    )
    parser.add_argument(
        "--path-instance",
        type=Path,
        default=Path("instances/instance_98.csv"), 
        metavar="FILE",
        help="Location of the instance CSV file.",
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

    app = Main(args.path_graph, args.path_capacity, args.path_instance)
    app.run()


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


