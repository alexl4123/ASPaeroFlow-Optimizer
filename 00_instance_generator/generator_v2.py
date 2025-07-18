
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

def default_flight_counts() -> List[int]:
    """The flight‑set sizes requested in the project spec."""
    return [
        3,
        5,
        10,
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


def sample_departure(rng: random.Random, *, mu_h: float, sigma_h: float, low_h: int, high_h: int) -> int:
    """Return a start time in *slots* (integer) using truncated normal sampling."""
    while True:
        t = rng.normalvariate(mu_h, sigma_h)
        if low_h <= t < high_h:
            return int(round(t)) * TIME_GRANULARITY  # convert hours→slots


def generate_flights(
    rng: random.Random,
    G: nx.Graph,
    airport_nodes: List[int],
    n_flights: int,
) -> List[Tuple[int, int, int]]:
    """Return a list of tuples (Flight_ID, Position, Time)."""

    flights: List[Tuple[int, int, int]] = []
    added = 0

    while added < n_flights:
        f_id = added
        src, tgt = rng.sample(airport_nodes, 2)
        path = nx.shortest_path(G, src, tgt)  # list of vertices

        start_slot = sample_departure(
            rng,
            mu_h=MU_IN_H,
            sigma_h=SIGMA_H,
            low_h=LOW_H,
            high_h=HIGH_H,
        )

        # Append (vertex, time) pairs as long as we stay within the planning horizon
        traj = []
        for hop, vertex in enumerate(path):
            t_slot = start_slot + hop  # one slot per edge traversal
            if t_slot >= HIGH_H * TIME_GRANULARITY:
                break  # flight overruns the horizon → discard whole flight
            traj.append((f_id, vertex, t_slot))

        if traj and traj[-1][1] == tgt:  # reached destination ⇒ accept flight
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

################################################################################
# Main driver
################################################################################

def main(argv: List[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Generate ATFCM benchmark instances.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(f"instances_{GRID_HEIGHT:03d}_{GRID_WIDTH:03d}"),
        help="Destination directory for all instance folders.",
    )
    parser.add_argument(
        "--flights",
        type=int,
        nargs="*",
        default=default_flight_counts(),
        help="Space‑separated list of flight counts (defaults to the paper set).",
    )
    parser.add_argument("--seed", type=int, default=RND_SEED, help="RNG seed or −1 for random.")
    args = parser.parse_args(argv)

    rng = random.Random(None if args.seed == -1 else args.seed)

    # Build the static airspace graph once
    G, airport_nodes = generate_grid(GRID_WIDTH, GRID_HEIGHT, AIRPORT_STEP)

    # Pre‑compute static tables
    edges = list(G.edges())

    # Ensure output root exists
    output_folder = args.out

    args.out.mkdir(parents=True, exist_ok=True)

    capacity_factor = float(400) / float(30000)

    for n in args.flights:
        folder = args.out / f"{n:07d}"
        folder.mkdir(exist_ok=True)

        cap_enroute = math.ceil(float(n)*capacity_factor)

        capacities = capacity_table(G, airport_nodes,cap_enroute)

        # 1. Flight trajectories
        flights = generate_flights(rng, G, airport_nodes, n)

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
