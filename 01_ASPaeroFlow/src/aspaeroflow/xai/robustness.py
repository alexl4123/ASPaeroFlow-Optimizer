"""Exposure of a fixed ASPaeroFlow plan to capacity loss (prototype; no re-optimisation).

A plan is the solution matrices the optimizer saves (main.py --save-results):
    converted_instance_matrix   flights x time, the sector a flight is in (-1: not flying)
    converted_navpoint_matrix   flights x time, the navpoint a flight is at (-1: none)
    navaid_sector_time_assignment  navpoints x time, the sector of every navpoint
and the instance's atomic capacities (sectors.csv: one capacity per navpoint).

Demand q(s,t) is the number of flights in sector s at time t. Capacity c(s,t) is, under the
default composite rule, the largest atomic capacity of the navpoints that form s at t. The
overload of a plan is the sum over (s,t) of max(0, q(s,t) - c(s,t)).

Measures (all on the fixed plan):
    slack(s,t)            c(s,t) - q(s,t)
    attribution(f)        overload units charged to flight f: every overloaded cell's overload is
                          shared equally among the flights in it. This uniform share is the
                          Shapley value of the game "overload of this cell" (the flights in a cell
                          are symmetric), summed over the cells.
    exposure(f)           time steps flight f spends at a navpoint whose capacity is reduced
Scenarios degrade atomic capacities, which is where weather acts in the formal model:
    independent  every navpoint loses capacity in a time window with probability p
    storms       a few cells move across the region; navpoints inside a cell during its life
                 lose a fraction of their capacity
Note that with the max rule a storm over part of a sector leaves that sector's capacity
untouched as long as one of its navpoints is unaffected; the path exposure shows the flights
that fly through the storm anyway.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Plan:
    occupancy: np.ndarray        # F x T, sector id or -1
    navpoints: np.ndarray        # F x T, navpoint id or -1
    allocation: np.ndarray       # N x T, sector of every navpoint
    atomic: np.ndarray           # N, atomic capacity per navpoint

    @classmethod
    def load(cls, results_dir: Path, instance_dir: Path) -> "Plan":
        def read(name):
            for suffix in (".csv", ".csv.gz"):
                path = Path(results_dir) / (name + suffix)
                if path.exists():
                    return np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
            raise FileNotFoundError(f"{name} not in {results_dir}")
        occupancy = read("converted_instance_matrix")
        navpoints = read("converted_navpoint_matrix")
        allocation = read("navaid_sector_time_assignment")
        sectors = np.loadtxt(Path(instance_dir) / "sectors.csv", delimiter=",", skiprows=1, ndmin=2)
        atomic = np.zeros(allocation.shape[0], dtype=np.int64)
        atomic[sectors[:, 0].astype(int)] = sectors[:, 1].astype(np.int64)
        T = min(occupancy.shape[1], navpoints.shape[1], allocation.shape[1])
        return cls(occupancy[:, :T], navpoints[:, :T], allocation[:, :T], atomic)

    @property
    def shape(self):
        return self.allocation.shape            # (N, T)

    def demand(self) -> np.ndarray:
        N, T = self.shape
        q = np.zeros((N, T), dtype=np.int64)
        f, t = np.nonzero(self.occupancy >= 0)
        np.add.at(q, (self.occupancy[f, t], t), 1)
        return q

    def capacity(self, atomic_nt: Optional[np.ndarray] = None) -> np.ndarray:
        """Composite capacity (max rule) for atomic capacities per navpoint and time."""
        N, T = self.shape
        if atomic_nt is None:
            atomic_nt = np.repeat(self.atomic[:, None], T, axis=1)
        c = np.zeros((N, T), dtype=np.int64)
        t_idx = np.tile(np.arange(T), N)
        np.maximum.at(c, (self.allocation.ravel(), t_idx), atomic_nt.ravel())
        return c


def overload_cells(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum(0, q - c)


def attribution(plan: Plan, over: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Overload units charged to every flight (uniform share per cell = Shapley value per cell)."""
    F, T = plan.occupancy.shape
    share = np.zeros_like(over, dtype=float)
    mask = over > 0
    share[mask] = over[mask] / q[mask]
    out = np.zeros(F)
    f, t = np.nonzero(plan.occupancy >= 0)
    np.add.at(out, f, share[plan.occupancy[f, t], t])
    return out


# ---------------------------------------------------------------------------- scenarios

def scenario_independent(plan: Plan, rng: np.random.Generator, p: float = 0.05, loss: float = 0.5,
                         window: int = 3) -> np.ndarray:
    """Every navpoint loses `loss` of its capacity in a random window with probability p."""
    N, T = plan.shape
    factor = np.ones((N, T))
    hit = rng.random(N) < p
    for v in np.flatnonzero(hit):
        start = rng.integers(0, max(1, T - window))
        factor[v, start:start + window] = 1.0 - loss
    return factor


def scenario_storms(plan: Plan, rng: np.random.Generator, coords: np.ndarray, cells: int = 2,
                    radius: float = 1.5, loss: float = 0.75, life: int = 6) -> np.ndarray:
    """`cells` storms with a random start, direction and lifetime; navpoints within `radius`
    (in the units of `coords`) of a storm centre lose `loss` of their capacity while it lives."""
    N, T = plan.shape
    factor = np.ones((N, T))
    lo, hi = np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    for _ in range(cells):
        t0 = int(rng.integers(0, max(1, T - life)))
        centre = lo + rng.random(2) * span
        velocity = (rng.random(2) - 0.5) * span / max(life, 1)
        for k in range(life):
            t = t0 + k
            if t >= T:
                break
            c = centre + k * velocity
            inside = np.linalg.norm(coords - c, axis=1) <= radius
            factor[inside, t] = np.minimum(factor[inside, t], 1.0 - loss)
    return factor


def analyse(plan: Plan, scenarios: int = 200, model: str = "storms", seed: int = 1,
            coords: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
    """Nominal measures and the distribution of overload and exposure over sampled scenarios."""
    rng = np.random.default_rng(seed)
    N, T = plan.shape
    F = plan.occupancy.shape[0]
    q = plan.demand()
    c0 = plan.capacity()
    over0 = overload_cells(q, c0)

    totals = np.zeros(scenarios)
    flight_overload = np.zeros((scenarios, F))
    flight_exposed = np.zeros((scenarios, F))
    sector_overloaded = np.zeros(N)              # in how many scenarios a sector has any overload
    sector_overload = np.zeros(N)                # summed overload units
    base = np.repeat(plan.atomic[:, None], T, axis=1).astype(float)
    flying = plan.navpoints >= 0
    for k in range(scenarios):
        if model == "independent":
            factor = scenario_independent(plan, rng, **kwargs)
        else:
            if coords is None:
                raise ValueError("the storm model needs navpoint coordinates")
            factor = scenario_storms(plan, rng, coords, **kwargs)
        atomic_nt = np.floor(base * factor).astype(np.int64)
        over = overload_cells(q, plan.capacity(atomic_nt))
        totals[k] = over.sum()
        flight_overload[k] = attribution(plan, over, q)
        reduced = factor < 1.0
        f, t = np.nonzero(flying)
        np.add.at(flight_exposed[k], f, reduced[plan.navpoints[f, t], t])
        per_sector = over.sum(axis=1)
        sector_overloaded += per_sector > 0
        sector_overload += per_sector

    order = np.sort(totals)
    tail = order[int(0.9 * scenarios):] if scenarios >= 10 else order
    expected_attr = flight_overload.mean(axis=0)
    p_exposed = (flight_exposed > 0).mean(axis=0)
    top_flights = np.argsort(-(expected_attr + 1e-6 * p_exposed))[:10]
    top_sectors = np.argsort(-sector_overload)[:10]
    slack = c0 - q
    occupied = q > 0
    return {
        "scenarios": scenarios, "model": model, "seed": seed, "parameters": kwargs,
        "nominal": {
            "overload": int(over0.sum()),
            "tight_cells": int(((slack == 0) & occupied).sum()),
            "occupied_cells": int(occupied.sum()),
        },
        "overload": {
            "mean": float(totals.mean()), "median": float(np.median(totals)),
            "p_any": float((totals > 0).mean()), "cvar90": float(tail.mean()),
            "max": float(totals.max()), "histogram": np.bincount(totals.astype(int)).tolist(),
        },
        "flights": [{"flight": int(f), "expected_overload_share": round(float(expected_attr[f]), 3),
                     "p_exposed": round(float(p_exposed[f]), 3)} for f in top_flights],
        "sectors": [{"sector": int(s), "p_overloaded": round(float(sector_overloaded[s] / scenarios), 3),
                     "expected_overload": round(float(sector_overload[s] / scenarios), 3)} for s in top_sectors],
    }


def coordinates_of(instance_dir: Path) -> np.ndarray:
    """Navpoint coordinates (lat, lon) or grid positions, indexed by navpoint id."""
    from .session import instance_graph
    graph = instance_graph(Path(instance_dir))
    n = max(v["id"] for v in graph["vertices"]) + 1
    coords = np.full((n, 2), np.nan)
    for v in graph["vertices"]:
        if v["lat"] is not None:
            coords[v["id"]] = (v["lat"], v["lon"])
    return coords


def save(result: Dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(result, indent=1), encoding="utf-8")
