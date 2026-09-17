"""The cost of a graph edge in timesteps, shared by every solver folder in this repository.

An instance's ``graph_edges.csv`` gives each edge's length as ``dist_m``, in metres with a
fractional part. A flight's filed plan was timed by the data generator
(ASPaeroFlow-DataGenerator, ``04_simplified_filed_flight_plan_generator.py``,
``_edge_duration_slots``) from that float distance, and a solver must charge exactly the same
number of timesteps for the same edge and airframe speed, or the filed plan itself becomes
infeasible.

The solvers used to read ``dist_m`` with the same ``int(ceil(float(x)))`` converter as every
integer column, so they rounded the distance up to whole metres before dividing. Where a
timestep boundary lies between ``dist_m`` and ``ceil(dist_m)`` that charges one timestep more
than the generator (V2 DACH TG60, edge 347-1202, 13272.548 m at 430 kts: 1 timestep in the
generator, 2 in the solvers). The distance is therefore kept as a float and rounded exactly
once, at the timestep level, by ``edge_duration_timesteps``, which repeats the generator's
expression in the generator's operation order.

Every solver folder reaches this module through its own ``edge_cost_bootstrap.py`` shim, the
same way ``common/navpoint_sector_allocation.py`` is reached.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

#: Metres per second in one knot, as the data generator uses it.
KNOTS_TO_METRES_PER_SECOND = 0.51444


def edge_duration_timesteps(distance_m, speed_kts, timestep_granularity) -> int:
    """Timesteps an airframe at ``speed_kts`` needs for an edge of ``distance_m`` metres.

    The same expression, in the same operation order, as the generator's
    ``_edge_duration_slots``: ``max(1, ceil((distance / (speed * 0.51444)) / (3600 / TG)))``.
    ``distance_m`` must be the float distance from ``graph_edges.csv``, not a rounded one.
    """
    speed_ms = float(speed_kts) * KNOTS_TO_METRES_PER_SECOND
    if speed_ms <= 0:
        return 1  # defensive, as in the generator
    duration_seconds = float(distance_m) / speed_ms
    slot_seconds = 3600.0 / float(timestep_granularity)
    return max(int(math.ceil(duration_seconds / slot_seconds)), 1)


def load_graph_edges(path) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``graph_edges.csv`` (``source,target,dist_m``).

    Returns ``(edges, dist_m)``: ``edges`` is the ``(|E|, 2)`` integer array of source and target
    vertex ids, converted with ``int(ceil(float(x)))`` like every other integer column the solvers
    read; ``dist_m`` is the ``(|E|,)`` float array of distances, read with ``float`` and not
    rounded.

    Raises FileNotFoundError if *path* does not exist and ValueError if it cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        edges = np.loadtxt(path, delimiter=",", dtype=int, skiprows=1, usecols=(0, 1), ndmin=2,
                           converters=lambda x: int(math.ceil(float(x))))
        dist_m = np.loadtxt(path, delimiter=",", dtype=float, skiprows=1, usecols=(2,), ndmin=1,
                            converters=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    return edges, dist_m
