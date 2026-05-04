from __future__ import annotations

import argparse
import sys
import time
import math
import networkx as nx
import os
import json

import base64
import zmq

import numpy as np
import warnings

import pickle

import multiprocessing as mp

from pathlib import Path
from typing import Any, Final, List, Optional, Callable
from datetime import datetime, timezone


from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
    
import numpy as np
import networkx as nx

from .solver import Solver, Model
from .optimize_flights import OptimizeFlights, MAX, TRIANGULAR, LINEAR
from .auxiliaries.dto_helpers import convert_dto_to_global_vars, convert_global_vars_to_dto
from .auxiliaries.computation_helpers import compute_total_number_sectors, last_valid_pos, system_loads_computation

from .main_loop_components.setup_before_optimization import SetupBeforeOptimization
from .main_loop_components.after_optimization import AfterOptimization
from .main_loop_components.iteration_step import IterationStep
from .main_loop_components.evaluate_solution import EvaluateSolution


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
        sector_capacity_factor: Optional[int],
        number_capacity_management_configs: Optional[int],
        capacity_management_enabled: Optional[bool],
        composite_sector_function: Optional[str],
        experiment_name: Optional[str],
        wandb_log: Optional[Callable[[dict],None]] = None,
        optimizer = "ASP",
        max_number_navpoints_per_sector = -1,
        max_number_sectors = -1,
        minimize_number_sectors = False,        
        convex_sectors = 0,
        control_context = None,
        control_ctrl_socket = None,
        control_pub_socket = None,
        control_poller = None,
        controller_enabled = False,
        data_dir = None,
        max_considered_aircraft = 2,
        explainability_context = None,
        ) -> None:

        self._graph_path: Optional[Path] = graph_path
        self._sectors_path: Optional[Path] = sectors_path
        self._flights_path: Optional[Path] = flights_path
        self._airports_path: Optional[Path] = airports_path
        self._airplanes_path: Optional[Path] = airplanes_path
        self._airplane_flight_path: Optional[Path] = airplane_flight_path
        self._navaid_sector_path: Optional[Path] = navaid_sector_path
        self._data_dir = data_dir

        self.max_number_navpoints_per_sector = max_number_navpoints_per_sector
        self.max_number_sectors = max_number_sectors
        self.max_considered_aircraft = max_considered_aircraft
        self.minimize_number_sectors = minimize_number_sectors

        self._control_context = control_context
        self._control_ctrl_socket = control_ctrl_socket
        self._control_pub_socket = control_pub_socket
        self._control_poller = control_poller
        self._controller_enabled = controller_enabled

        self._explainability_context = explainability_context

        self._optimizer = optimizer

        self._wandb_log = wandb_log

        self._convex_sectors = convex_sectors

        self.number_capacity_management_configs = number_capacity_management_configs
        if capacity_management_enabled in ["True","true"]:
            self.capacity_management_enabled = True
        elif capacity_management_enabled in ["False","false"]:
            self.capacity_management_enabled = False
        else:
            raise Exception("Invalid input for capacity_management_enabled")
        
        
        self.composite_sector_function  = composite_sector_function

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

        self.sector_capacity_factor = sector_capacity_factor


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(self) -> None:  
        """Run the application"""
        
        global_dto = convert_global_vars_to_dto(self)
        optimization_dto, global_dto = SetupBeforeOptimization(global_dto).setup_before_optimization()
        convert_dto_to_global_vars(self, global_dto)
               
        while np.any(optimization_dto["capacity_overload_mask"], where=True):

            global_dto = convert_global_vars_to_dto(self)
            optimization_dto, global_dto, command  = IterationStep(global_dto).optimization_step(optimization_dto)
            convert_dto_to_global_vars(self, global_dto)

            if command == "continue":
                continue
            elif len(command) == 2 and command[0].startswith("<LOAD>"):
                return command

            global_dto = convert_global_vars_to_dto(self)
            optimization_dto, global_dto = EvaluateSolution(global_dto).evaluate_solution(optimization_dto)
            convert_dto_to_global_vars(self, global_dto)

        global_dto = convert_global_vars_to_dto(self)
        r0,r1,global_dto = AfterOptimization(global_dto).post_processing(optimization_dto)
        convert_dto_to_global_vars(self, global_dto)
        return r0,r1

    def get_total_atfm_delay(self):
        return self.total_atfm_delay
    

  
    def run_parallel(self, jobs):
        """
        Fire ``start()`` for every ``OptimizeFlights`` instance in parallel.
        """
        ctx = mp.get_context("spawn")          # <- fork‑safe
        with ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=ctx
        ) as pool:
            futures = [pool.submit(job.start) for job in jobs]
            return [f.result() for f in futures]
        
     
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
   
   
    def capacity_time_matrix_reference(self,
                            cap: np.ndarray,
                            n_times: int,
                            time_granularity: int,
                            navaid_sector_time_assignment: np.ndarray,
                            z = 1) -> np.ndarray:

        template_matrix = np.zeros((navaid_sector_time_assignment.shape[0],navaid_sector_time_assignment.shape[1]))

        for cap_index in range(navaid_sector_time_assignment.shape[0]):

            for timestep_t in range(navaid_sector_time_assignment.shape[1]):
                
                atomic_sector_boolean_matrix = np.nonzero(navaid_sector_time_assignment[:,timestep_t] == cap_index)

                total_capacity = self.compute_sector_capacity(cap[atomic_sector_boolean_matrix[0],1],z)

                #if timestep_t == 0:
                #    print(total_capacity)

                if total_capacity > 0:

                    #total_capacity = np.sum(cap[atomic_sector_boolean_matrix[0],1])
                    sample_array = np.zeros(((time_granularity)))
                    per_timestep_capacity = math.floor(total_capacity / time_granularity)
                    sample_array = sample_array[:] + per_timestep_capacity

                    # rem_cap < time_granularity per construction
                    rem_cap = total_capacity - (per_timestep_capacity * time_granularity)


                    if rem_cap > 0:
                        step_size = math.ceil(time_granularity / rem_cap)
                        time_index = 0

                        while rem_cap > 0:

                            sample_array[time_index] += 1
                            rem_cap -= 1

                            time_index += step_size
                            time_index = time_index % time_granularity

                    template_matrix[cap_index, timestep_t] = sample_array[timestep_t % time_granularity]

        # DEBUG ONLY:
        np.savetxt("20251004_cap_mat.csv", template_matrix, delimiter=",",fmt="%i")

        return template_matrix



    def minimize_number_of_sectors(self, navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, max_number_navpoints_per_sector, max_number_sectors, t_start, t_end, navpoint_networkx_graph, airport_vertices):
        # --- Basic setup ----------------------------------------------------
        # Work on inclusive time window [t_start, t_end]
        time_slice = slice(t_start, t_end + 1)
        window_length = t_end - t_start + 1

        # We treat sector IDs as integers in [0, max_sector_id]
        # (this is consistent with capacity_time_matrix / system_loads rows).
        if navaid_sector_time_assignment.size == 0:
            return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads 

        n_sectors = navaid_sector_time_assignment.shape[0]

        # Boolean mask of "active" sectors (can be merged / still exist)
        active = np.ones(n_sectors, dtype=bool)

        # --- Precompute navpoint counts per sector and time -----------------
        # navaid_sector_time_assignment: shape (n_navaids, n_times)
        navpoint_counts = np.zeros((n_sectors, window_length), dtype=np.int32)

        # For each time in the window, count how many navaids are assigned
        # to each sector. Using bincount is usually quite efficient.
        for local_t, global_t in enumerate(range(t_start, t_end + 1)):
            col = navaid_sector_time_assignment[:, global_t]

            # If you use -1 or similar as "no sector", ignore those.
            valid_mask = col >= 0
            if not np.any(valid_mask):
                continue

            counts = np.bincount(col[valid_mask], minlength=n_sectors)
            navpoint_counts[:, local_t] = counts

        # A sector "exists" in the interval iff it has at least one navpoint
        # assigned in that window.
        sector_has_navpoints = navpoint_counts.sum(axis=1) > 0

        # --- Demand / capacity aggregates over the window -------------------
        demand_window = system_loads[:, time_slice]
        capacity_window = capacity_time_matrix[:, time_slice]

        agg_demand = demand_window.sum(axis=1)          # shape (n_sectors,)
        agg_capacity = capacity_window.sum(axis=1)      # shape (n_sectors,)

        # Light sectors: exist, have capacity, and demand <= 50% of capacity
        has_capacity = agg_capacity > 0
        below_threshold = agg_demand <= 0.2 * agg_capacity

        light_sector_mask = sector_has_navpoints & has_capacity & below_threshold
        light_sectors = np.nonzero(light_sector_mask)[0]

        # Sort by aggregated demand ascending (lowest demand first)
        light_sectors = list(light_sectors)
        light_sectors.sort(key=lambda s: agg_demand[s])


        # If there are no light sectors, nothing to do.
        if not light_sectors:
            #print("NO LIGHT SECTORS")
            #print(np.any(sector_has_navpoints))
            #print(np.any(has_capacity))
            #print(np.any(below_threshold))
            return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads 

        # --- Helper to count current active sectors in the window ----------
        def active_sector_count() -> int:
            # A sector is active if it's still marked active AND has some
            # positive demand in the considered window.
            # (You can adapt this condition if your notion of "active"
            #  differs, e.g., also check capacities / navpoints.)
            return int(
                np.logical_and(
                    active,
                    (system_loads[:, time_slice].sum(axis=1) > 0),
                ).sum()
            )

        tmp_sectors = []
        for sec in light_sectors:
            if sec not in airport_vertices:
                tmp_sectors.append(sec)
        light_sectors = tmp_sectors


        # --- Main fixed-point loop -----------------------------------------
        changed = True
        while changed:
            changed = False

            # Scan through sectors in ascending demand order.
            for sec in light_sectors:
                if not active[sec]:
                    continue

                all_navaids = list(set(np.nonzero(navaid_sector_time_assignment[:,t_start:t_end+1] == sec)[0]))

                if np.any(navaid_sector_time_assignment[all_navaids,t_start:t_end+1] != sec):
                    active[sec] = False

                    continue


                # It may happen that sec lost all navpoints / demand due to
                # earlier merges; then it's no longer meaningful.
                if navpoint_counts[sec].sum() == 0:
                    continue

                # If the graph does not contain this node, we cannot merge it.
                if not navpoint_networkx_graph.has_node(sec):
                    continue

                # Try to merge 'sec' with one of its neighbours.
                all_navaids_set = set(all_navaids)
                neighbors = {
                    nb
                    for v in all_navaids if v in navpoint_networkx_graph
                    for nb in navpoint_networkx_graph.neighbors(v)
                } - all_navaids_set

                if not neighbors:
                    continue

                for nbr_navaid in neighbors:
                    nbr_navaid = int(nbr_navaid)

                    nbr = navaid_sector_time_assignment[nbr_navaid,t_start]

                    if nbr == sec or not active[nbr]:
                        continue

                    if nbr in airport_vertices:
                        active[nbr] = False
                        continue

                    # Check combined capacity and demand constraints over
                    # the time window.
                    sec_load = system_loads[sec, time_slice]
                    nbr_load = system_loads[nbr, time_slice]
                    combined_demand = sec_load + nbr_load

                    sec_cap = capacity_time_matrix[sec, time_slice]
                    nbr_cap = capacity_time_matrix[nbr, time_slice]
                    combined_capacity = np.maximum(sec_cap, nbr_cap)

                    # Capacity constraint: for every timestep, demand <= capacity
                    if np.any(combined_demand > combined_capacity):
                        continue

                    # Navpoint constraint: per timestep, number of navpoints
                    # must not exceed max_number_navpoints_per_sector.
                    combined_navpoints = navpoint_counts[sec, :] + navpoint_counts[nbr, :]
                    if (
                        max_number_navpoints_per_sector is not None
                        and max_number_navpoints_per_sector > 0
                        and np.any(combined_navpoints > max_number_navpoints_per_sector)
                    ):
                        continue

                    # If we reach here, merging sec and nbr is allowed.
                    # We "merge nbr into sec":
                    #   - update demand and capacity rows,
                    #   - reassign navpoints from nbr to sec,
                    #   - update navpoint counts,
                    #   - deactivate nbr and remove from graph.

                    # Update loads and capacity
                    system_loads[sec, time_slice] = combined_demand
                    system_loads[nbr, time_slice] = 0

                    capacity_time_matrix[sec, time_slice] = combined_capacity
                    capacity_time_matrix[nbr, time_slice] = 0

                    # Update navpoint assignments in the window
                    sector_slice = navaid_sector_time_assignment[:, time_slice]
                    sector_slice[sector_slice == nbr] = sec

                    # Update navpoint counts
                    navpoint_counts[sec, :] = combined_navpoints
                    navpoint_counts[nbr, :] = 0

                    # Mark neighbour as inactive and remove from graph
                    active[nbr] = False

                    changed = True

                    # The ordering of sectors could be recomputed here if
                    # desired, but for a simple heuristic we keep it fixed.
                    # Break to restart scanning from the lightest sector.
                    break

                if changed:
                    # Restart outer loop after any successful merge

                    index_sec = light_sectors.index(sec)

                    for _ in range(index_sec):
                        active[light_sectors[0]] = False
                        del light_sectors[0]

                    break

        converted_instance_matrix = OptimizeFlights.instance_computation_after_sector_change(list(range(converted_instance_matrix.shape[0])),
                            converted_navpoint_matrix, converted_instance_matrix, navaid_sector_time_assignment)
        

        return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads




