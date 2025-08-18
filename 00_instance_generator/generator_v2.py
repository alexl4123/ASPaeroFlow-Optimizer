
#!/usr/bin/env python3
"""
ATFCM Instance Generator
========================

This utility creates benchmark instances for the Air‑Traffic Flow and Capacity
Management (ATFCM) problem.  For every requested number of flights it produces a
folder with the following layout::

    ├── 0000003/
    │   ├── instance.csv   # (Flight_ID, Position, Time)
    │   ├── capacity.csv   # (Sector_ID, Capacity)
    │   ├── edges.csv      # (source, target)
    │   └── airports.csv   # (Airport_Vertex)
    ├── 0000005/
    │   └── ...
    └── 0100000/
        └── ...

The graph is a rectangular grid generated with
NetworkX.  Additional *airport* vertices are attached to every second grid
vertex (step = 2) giving a realistic network topology.

Flight trajectories are the shortest paths between two randomly chosen distinct
airport vertices.  Departure times are sampled from a truncated normal
distribution and rounded to the nearest 15‑minute slot.  Each arc traversal
advances the time by one slot.

Airport Capacities are kept simple:
* airport sectors: CAP_AIRPORT (default 100)

The script is fully parameterised via CLI flags but sensible defaults are
provided so a single command is enough to reproduce the paper’s benchmark::

    python generate_instances.py --out data

Dependencies:  networkx, pandas, numpy (and Python ≥ 3.9).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

################################################################################
# Parameters
################################################################################


def _compute_graph_capacity_factor(grid_width, grid_height):

    number_vertices = float(grid_width * grid_height)
    constant_vertices = float(11*11)

    if number_vertices <= constant_vertices - 1:
        # Log10 -> So if eg., number_vertices=25, constant_vertices=121 => log10(121-25)=1.98
        capacity_factor = math.log(constant_vertices - number_vertices, 10) + 1

    else:
        # number_vertices >= constant_vertices
        capacity_factor = constant_vertices / number_vertices

    return capacity_factor

def default_flight_counts() -> List[int]:
    """The flight‑set sizes requested in the project spec."""
    return [
        3,
        5,
        10,
        30,
        60,
        100,
    ]
    """
        30,
        60,
        100,
        300,
        600,
        1000,
        3000,
        6000,
        10000,
        30000,
        60000,
        100000,
    """

# default constants
GRID_WIDTH = GRID_HEIGHT = 5
AIRPORT_STEP = 2  # attach an airport to every 2nd grid vertex (row & column)
TIME_GRANULARITY = 1  # ← 4 × 15‑minute slots = 1 h
MU_IN_H = 10  # mean start time (in hours)
SIGMA_H = 4  # std‑dev of start time (hours)
LOW_H, HIGH_H = 0, 24  # start time window (hours)
CAP_AIRPORT = 100000  # capacity of airport vertices per slot
RND_SEED = 42  # for reproducibility; set None to disable seeding

################################################################################
# Helper functions
################################################################################

def generate_grid(width: int, height: int, step: int) -> Tuple[nx.Graph, List[int]]:
    """Return (graph, list_of_airport_vertices)."""

    # 1. Grid core ﹙en‑route sectors﹚
    G = nx.grid_2d_graph(height, width)  # nodes are (row, col)

    # Map to contiguous ints (row‑major order)
    mapping = {(r, c): r * width + c for r in range(height) for c in range(width)}
    G = nx.relabel_nodes(G, mapping)

    # 2. Attach airports
    airport_nodes: List[int] = []
    next_id = width * height  # first id after the grid

    for r in range(0, height, step):
        for c in range(0, width, step):
            anchor = mapping[(r, c)]
            G.add_node(next_id)
            G.add_edge(anchor, next_id)
            airport_nodes.append(next_id)
            next_id += 1

    return G, airport_nodes


def sample_departure(
    rng: random.Random,
    *,
    mu_h: float,
    sigma_h: float,
    low_h: int,
    high_h: int,
    time_granularity: int,
) -> int:
    """Return a start time in *slots* (integer) using truncated normal sampling."""
    while True:
        t = rng.normalvariate(mu_h, sigma_h)
        if low_h <= t < high_h:
            return int(round(t)) * time_granularity


def generate_flights(
    rng: random.Random,
    G: nx.Graph,
    airport_nodes: List[int],
    n_flights: int,
    time_granularity: int,
) -> List[Tuple[int, int, int]]:
    """Return a list of tuples (Flight_ID, Position, Time)."""
    flights: List[Tuple[int, int, int]] = []
    added = 0

    # If the graph might be disconnected, be robust:
    max_trials_per_flight = 200

    while added < n_flights:
        f_id = added
        trials = 0
        path = None
        src = tgt = None

        # Find a valid airport pair with a path
        while trials < max_trials_per_flight and path is None:
            src, tgt = rng.sample(airport_nodes, 2)
            try:
                path = nx.shortest_path(G, src, tgt)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = None
            trials += 1

        if path is None:
            # Could not find a connected airport pair → give up on this flight index and move on
            # (prevents infinite loops on sparse/disconnected graphs)
            added += 1
            continue

        start_slot = sample_departure(
            rng,
            mu_h=MU_IN_H,
            sigma_h=SIGMA_H,
            low_h=LOW_H,
            high_h=HIGH_H,
            time_granularity=time_granularity,
        )

        traj = []
        for hop, vertex in enumerate(path):
            t_slot = start_slot + hop  # one slot per edge traversal
            if t_slot >= HIGH_H * time_granularity:
                break  # over horizon → discard whole flight
            traj.append((f_id, vertex, t_slot))

        if traj and traj[-1][1] == tgt:
            flights.extend(traj)
            added += 1

    return flights


def capacity_table(G: nx.Graph, airport_nodes: Iterable[int], cap_enroute: int) -> List[Tuple[int, int]]:
    airports = set(airport_nodes)
    return [
        (v, CAP_AIRPORT if v in airports else cap_enroute) for v in G.nodes()
    ]


def write_csv(data: Iterable[Tuple], header: List[str], file: Path) -> None:
    pd.DataFrame(list(data), columns=header).to_csv(file, index=False)

def load_graph_from_csv(edges_csv: Path, airports_csv: Path) -> Tuple[nx.Graph, List[int]]:
    """Load an undirected graph and list of airport nodes from CSV files.

    edges.csv must have columns: source,target
    airports.csv must have column: Airport_Vertex
    """
    if not edges_csv.exists():
        raise FileNotFoundError(f"edges.csv not found: {edges_csv}")
    if not airports_csv.exists():
        raise FileNotFoundError(f"airports.csv not found: {airports_csv}")

    # Read edges
    edf = pd.read_csv(edges_csv)
    # Be tolerant to column names / order
    if not {"source", "target"}.issubset({c.lower() for c in edf.columns}):
        # Fallback: assume first two columns are source/target
        if edf.shape[1] < 2:
            raise ValueError("edges.csv must have at least two columns (source, target).")
        edf.columns = [str(c).lower() for c in edf.columns]
        if "source" not in edf.columns or "target" not in edf.columns:
            edf = edf.rename(columns={edf.columns[0]: "source", edf.columns[1]: "target"})
    else:
        # Normalize exact case
        lowmap = {c.lower(): c for c in edf.columns}
        edf = edf.rename(columns={lowmap["source"]: "source", lowmap["target"]: "target"})

    # Coerce node ids to ints where possible (fallback to str)
    def _coerce(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return str(v).strip()

    sources = edf["source"].map(_coerce)
    targets = edf["target"].map(_coerce)

    G = nx.Graph()
    G.add_edges_from(zip(sources, targets))

    # Read airports
    adf = pd.read_csv(airports_csv)
    col = None
    for candidate in adf.columns:
        if str(candidate).strip().lower() in {"airport_vertex", "airport", "vertex", "node"}:
            col = candidate
            break
    if col is None:
        # Fallback: use the first column
        col = adf.columns[0]

    airport_nodes = [ _coerce(v) for v in adf[col].tolist() ]

    # Ensure airport nodes exist in graph (allow isolated airports)
    for a in airport_nodes:
        if a not in G:
            G.add_node(a)

    return G, airport_nodes

################################################################################
# Main driver
################################################################################
##############################################################################

def main(argv: List[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Generate ATFCM benchmark instances.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(f"DEFAULT"),
        help="Destination directory for all instance folders.",
    )
    parser.add_argument(
        "--time-granularity",
        type=int,
        default=1,
        help="Time Granularity (1=1h, 4=15minutes,...).",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=5,
        help="Airspace grid width.",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=5,
        help="Airspace grid height.",
    )
    parser.add_argument(
        "--flights",
        type=int,
        nargs="*",
        default=default_flight_counts(),
        help="Space‑separated list of flight counts (defaults to the paper set).",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Directory containing edges.csv and airports.csv to reuse an existing graph.",
    )
    parser.add_argument(
        "--edges-file",
        type=Path,
        default=None,
        help="Path to edges.csv (overrides --graph-dir).",
    )
    parser.add_argument(
        "--airports-file",
        type=Path,
        default=None,
        help="Path to airports.csv (overrides --graph-dir).",
    )
    parser.add_argument("--seed", type=int, default=RND_SEED, help="RNG seed or −1 for random.")
    args = parser.parse_args(argv)

    SEED = 0 if args.seed == -1 else args.seed
    rng = random.Random(SEED)
    TIME_GRANULARITY = args.time_granularity
    GRID_WIDTH = args.grid_width
    GRID_HEIGHT = args.grid_height

    # Build or load the airspace graph
    if args.edges_file or args.graph_dir:
        edges_csv = args.edges_file or (args.graph_dir / "edges.csv")
        airports_csv = args.airports_file or (args.graph_dir / "airports.csv")
        G, airport_nodes = load_graph_from_csv(edges_csv, airports_csv)
        print(f"[i] Reusing graph from {edges_csv.parent if args.graph_dir else edges_csv}")
    else:
        G, airport_nodes = generate_grid(GRID_WIDTH, GRID_HEIGHT, AIRPORT_STEP)
        print(f"[i] Generated grid {grid_h}×{grid_w} with airports every {AIRPORT_STEP} nodes")

    # Pre-compute static tables
    edges = list(G.edges())

    # Ensure output root exists
    output_folder = args.out
    if output_folder.name == "DEFAULT":
        output_folder = Path(f"instances_{GRID_HEIGHT:03d}_{GRID_WIDTH:03d}_TG{TIME_GRANULARITY}_SEED{SEED}")

    output_folder.mkdir(parents=True, exist_ok=True)

    #base_capacity = math.ceil(float(400)/((1/3) * float(TIME_GRANULARITY) + (2/3)))
    #base_capacity = math.ceil(float(400)/float(TIME_GRANULARITY))
    #base_capacity = math.ceil(float(400)/((1/6) * float(TIME_GRANULARITY) + (5/6)))
    base_capacity = math.ceil(float(400)/((1/3) * float(TIME_GRANULARITY) + (2/3)))
    base_aircraft_number = float(30000)

    if GRID_WIDTH and GRID_HEIGHT:
        capacity_factor_graph = _compute_graph_capacity_factor(GRID_WIDTH, GRID_HEIGHT)
    else:
        capacity_factor_graph = 1

    #capacity_factor = float(400) * float(11) * float(11) / (float(30000) * float(GRID_WIDTH) * float(GRID_HEIGHT))

    for n in args.flights:
        folder = output_folder / f"{n:07d}"
        folder.mkdir(exist_ok=True)

        capacity_factor_flights = float(n) / base_aircraft_number
        cap_enroute = math.ceil(base_capacity * capacity_factor_flights * capacity_factor_graph) + 2
        capacities = capacity_table(G, airport_nodes,cap_enroute)

        capacities.sort()

        # 1. Flight trajectories
        flights = generate_flights(rng, G, airport_nodes, n, TIME_GRANULARITY)

        # 2. Persist CSVs --------------------------------------------------------
        write_csv(flights, ["Flight_ID", "Position", "Time"], folder / "instance.csv")
        write_csv(capacities, ["Sector_ID", "Capacity"], folder / "capacity.csv")
        write_csv(edges, ["source", "target"], folder / "edges.csv")
        write_csv([(v,) for v in airport_nodes], ["Airport_Vertex"], folder / "airports.csv")

        print(f"[✓] Generated {n:>7,} flights → {folder}")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)








