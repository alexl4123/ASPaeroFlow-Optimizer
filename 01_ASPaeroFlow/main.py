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
import math
import networkx as nx
import os
import json

import numpy as np
import warnings

import pickle

import multiprocessing as mp

from pathlib import Path
from typing import Any, Final, List, Optional, Callable
from datetime import datetime, timezone

from solver import Solver, Model

from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    
from concurrent.futures import ProcessPoolExecutor
from optimize_flights import OptimizeFlights, MAX, TRIANGULAR, LINEAR
from concurrent.futures import ProcessPoolExecutor
    
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _run(job):
    """Run parallelized job.

    Parameters
    ----------
    job
        Tuple of arguments for OptimizeFlights
    """

    optimizer = OptimizeFlights(*job)
    model = optimizer.start()

    return model


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
        ) -> None:

        self._graph_path: Optional[Path] = graph_path
        self._sectors_path: Optional[Path] = sectors_path
        self._flights_path: Optional[Path] = flights_path
        self._airports_path: Optional[Path] = airports_path
        self._airplanes_path: Optional[Path] = airplanes_path
        self._airplane_flight_path: Optional[Path] = airplane_flight_path
        self._navaid_sector_path: Optional[Path] = navaid_sector_path

        self.max_number_navpoints_per_sector = max_number_navpoints_per_sector
        self.max_number_sectors = max_number_sectors
        self.minimize_number_sectors = minimize_number_sectors

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

    def run(self) -> None:  
        """Run the application"""
        self.load_data()

        original_start_time = time.time()

        # For NumPy floating-point issues (divide/overflow/invalid/underflow):
        np.seterr(all='raise')  # or: with np.errstate(divide='raise', invalid='raise', over='raise')

        # For NumPy warnings that use Python’s warnings system (e.g., RuntimeWarning):
        warnings.filterwarnings("error", category=RuntimeWarning)

        if self.verbosity > 0:
            # --- Demonstration output ---------
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

        # 0.) Create navpaid sector time assignment (|R|XT):
        navaid_sector_time_assignment = self.create_initial_navpoint_sector_assignment(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity)

        # 1.) Create flights matrix (|F|x|T|) --> For easier matrix handling
        converted_navpoint_matrix, _ = self.instance_navpoint_matrix(self.flights, navaid_sector_time_assignment.shape[1], fill_value=-1)
        #converted_instance_matrix, planned_arrival_times = OptimizeFlights.instance_to_matrix_vectorized(self.flights, self.airplane_flight, navaid_sector_time_assignment.shape[1], self._timestep_granularity, navaid_sector_time_assignment)
        converted_instance_matrix, planned_arrival_times = OptimizeFlights.instance_to_matrix(self.flights, self.airplane_flight, navaid_sector_time_assignment.shape[1], self._timestep_granularity, navaid_sector_time_assignment)

        #converted_instance_matrix, planned_arrival_times = self.instance_to_matrix(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)
        #
        #np.testing.assert_array_equal(converted_instance_matrix, converted_instance_matrix_2)

        # 2.) Create demand matrix (|R|x|T|)
        system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)

        # 3.) Create capacity matrix (|R|x|T|)
        #capacity_time_matrix = self.capacity_time_matrix_reference(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = z)
        #capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor)

        capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor,
                                                                    composite_sector_function=self.composite_sector_function)
        #np.testing.assert_array_equal(capacity_time_matrix, capacity_time_matrix_test_2)

        #capacity_time_matrix_test = OptimizeFlights.capacity_time_matrix_test(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor)
        #np.testing.assert_array_equal(capacity_time_matrix, capacity_time_matrix_test)

        #np.savetxt("20251003_navaid_sector_time_assignment.csv", navaid_sector_time_assignment, delimiter=",",fmt="%i")
        #np.savetxt("20251003_instance_to_matrix.csv", converted_instance_matrix, delimiter=",",fmt="%i")
        #quit()

        if self.verbosity > 0:
            # --- Demonstration output ---------
            print("  converted-instance:   ", converted_instance_matrix.shape)
            print("  converted-navpoint:   ", converted_navpoint_matrix.shape)
            print("  navpoint-time-sector:   ", navaid_sector_time_assignment.shape)
            print("  system-loads-matrix:   ", system_loads.shape)
            print("  capacity-matrix:   ", capacity_time_matrix.shape)
            # -----------------------------------------------------------------



        # 4.) Subtract demand from capacity (|R|x|T|)
        capacity_demand_diff_matrix = capacity_time_matrix - system_loads
        # 5.) Create capacity overload matrix
        capacity_overload_mask = capacity_demand_diff_matrix < 0

        #number_of_conflicts = capacity_overload_mask.sum()
        number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()
        number_of_conflicts_prev = None

        counter_equal_solutions = 0
        additional_time_increase = 0
        max_number_airplanes_considered_in_ASP = 2

        iteration = 0
        fill_value = -1

        max_number_processors = self._number_threads
        seed = self._seed

        if self.verbosity > 1:
            np.savetxt("20250826_initial_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

        original_converted_instance_matrix = converted_instance_matrix.copy()

        original_max_explored_vertices = self._max_explored_vertices
        original_max_time = original_converted_instance_matrix.shape[1]
        self.original_max_time = original_max_time

        if self._max_delay_per_iteration < 0:
            #self._max_delay_per_iteration = original_max_time
            self._max_delay_per_iteration = 20

        if np.any(capacity_overload_mask, where=True):
            old_converted_instance = converted_instance_matrix.copy()
            old_converted_navpoint_matrix = converted_navpoint_matrix.copy()
            old_navaid_sector_time_assignment = navaid_sector_time_assignment.copy()
            
            default_number_capacity_management_configs = self.number_capacity_management_configs

            _, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)

        # Track to Weights & Biases when enabled
        current_time = time.time() - original_start_time

        number_sectors = self.compute_total_number_sectors(navaid_sector_time_assignment)

        if self._wandb_log is not None:

            self._wandb_log({
                "iteration": int(iteration), # FIRST ONE:
                "number_of_conflicts": int(number_of_conflicts),
                "total_delay": int(0),
                "number_sectors": int(number_sectors),
                "sector_diff": int(0),
                "number_reroutes": int(0),
                "number_reconfigurations": int(0),
                "current_time": int(0),
            })

            if self.verbosity > 1:
                print(f"""
    iteration -> {int(iteration)}
    number_of_conflicts -> {int(number_of_conflicts)}
    total_delay -> {int(0)}
    number_sectors -> {len(uniq.values)}
    current_time -> {int(0)}
                      """)
                
        output_dict = {}
        output_dict["OVERLOAD"] = int(number_of_conflicts)
        output_dict["ARRIVAL-DELAY"] = int(0)
        output_dict["SECTOR-NUMBER"] = int(number_sectors)
        output_dict["SECTOR-DIFF"] = int(0)
        output_dict["REROUTE"] = int(0)
        output_dict["RECONFIG"] = int(0)
        output_dict["TOTAL-TIME-TO-THIS-POINT"] =  int(current_time)
        output_dict["COMPUTATION-FINISHED"] = False
        output_string = json.dumps(output_dict)
        print(output_string)

        last_time_bucket_updated = 0

        if self.max_number_navpoints_per_sector == -1 or self.max_number_sectors == -1:
            if self.max_number_navpoints_per_sector == -1:
                max_number_navpoints_per_sector_update = True
            else:
                max_number_navpoints_per_sector_update = False

            if self.max_number_sectors == -1:
                max_number_sectors_update = True
            else:
                max_number_sectors_update = False

            for time_index in range(navaid_sector_time_assignment.shape[1]):

                uniq = np.unique_counts(navaid_sector_time_assignment[:,time_index])

                if max_number_navpoints_per_sector_update is True:
                    if max(uniq.counts) > self.max_number_navpoints_per_sector:
                        self.max_number_navpoints_per_sector = max(uniq.counts)

                if max_number_sectors_update is True:
                    if len(uniq.values) > self.max_number_sectors:
                        self.max_number_sectors = len(uniq.values)

        if self.minimize_number_sectors is True:

            if self.verbosity > 1:
                print("MINIMIZE NUMBER OF SECTORS INITIALIZED")

            t_start = 1
            t_end = 60

            for t_start in range(1,converted_instance_matrix.shape[1] + 1, self._timestep_granularity):
                t_end = t_start + self._timestep_granularity - 1

                if t_end >= converted_instance_matrix.shape[1]:
                    t_end = converted_instance_matrix.shape[1] - 1

                converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = self.minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, t_start, t_end, self.networkx_navpoint_graph, self.airports)

        global_t_start = 1
        time_bucket_updated = 0

        while np.any(capacity_overload_mask, where=True):
            if self.verbosity > 0:
                print(f"<ITER:{iteration}><REMAINING ISSUES:{str(number_of_conflicts)}>")

            cols_with_values = (converted_instance_matrix != -1).any(axis=0)           
            idxs = np.flatnonzero(cols_with_values)            
            last_idx = int(idxs[-1]) if idxs.size else -1     
            safety_factor = self._timestep_granularity * 2

            if last_idx + self._max_delay_per_iteration * (additional_time_increase + 1) + safety_factor >= converted_instance_matrix.shape[1]:
                # INCREASE MATRIX SIZE (TIME) AUTOMATICALLY
                diff = last_idx + self._max_delay_per_iteration * (additional_time_increase + 1) + safety_factor - converted_instance_matrix.shape[1] + 1
                in_units = math.ceil(diff / self._timestep_granularity)
                number_new_cols = in_units * self._timestep_granularity

                # 0.) Handle Sector Assignments:
                new_cols = np.repeat(navaid_sector_time_assignment[:,[-1]], number_new_cols, axis=1)  # shape (N,k)
                navaid_sector_time_assignment = np.concatenate([navaid_sector_time_assignment, new_cols], axis=1)
                # 1.) Handle Instance Matrix:
                extra_col = -1 * np.ones((converted_instance_matrix.shape[0], number_new_cols), dtype=int)
                converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                extra_col = -1 * np.ones((converted_navpoint_matrix.shape[0], number_new_cols), dtype=int)
                converted_navpoint_matrix = np.hstack((converted_navpoint_matrix, extra_col)) 
                
                # 2.) Create demand matrix (|R|x|T|):
                system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                # 3.) Create capacity matrix (|R|x|T|):
                #capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, z = self.sector_capacity_factor)
                capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor, composite_sector_function=self.composite_sector_function)

                # 3.) Create capacity matrix (|R|x|T|)
                #capacity_time_matrix = self.capacity_time_matrix_reference(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = z)
                # 4.) Subtract demand from capacity (|R|x|T|):
                capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                # 5.) Create capacity overload matrix:
                capacity_overload_mask = capacity_demand_diff_matrix < 0

            if navaid_sector_time_assignment.shape[1] != converted_instance_matrix.shape[1]:
                raise Exception(f"Navaid time must correspond to instance time: {navaid_sector_time_assignment.shape[1]} != {converted_instance_matrix.shape[1]}")

            #time_index,bucket_index = self.first_overload(capacity_overload_mask)

            max_number_processors = 1
            time_bucket_tuples = self.first_overloads(capacity_overload_mask, k=max_number_processors)

            start_time = time.time()

            all_jobs = []
            all_candidates = {}
            for time_index, sector_index in time_bucket_tuples:
            
                time_bucket_updated = time_index

                #print(capacity_demand_diff_matrix[sector_index, time_index])
                
                #_, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)
                job, candidates = self.build_job(time_index, sector_index, converted_instance_matrix, converted_navpoint_matrix,
                                capacity_time_matrix,
                                system_loads, capacity_demand_diff_matrix, additional_time_increase, fill_value, 
                                max_number_airplanes_considered_in_ASP, max_number_processors, original_max_time,
                                self.networkx_navpoint_graph, self.unit_graphs, planned_arrival_times, self.airplane_flight,
                                self.airplanes, navaid_sector_time_assignment, flight_durations,
                                iteration, self.sector_capacity_factor, self.flights
                                )

                
                all_new = True
                for candidate in candidates:
                    if candidate in all_candidates:
                        all_new = False
                    else:
                        all_candidates[candidate] = True

                if all_new:
                    all_jobs.append(job)

            solutions = Parallel(n_jobs=max_number_processors, backend="loky")(
                        delayed(_run)(job) for job in all_jobs)

            #models = [_run(job) for job in jobs]

            end_time = time.time()
            if self.verbosity > 1:
                print(f">> Elapsed solving time: {end_time - start_time}")

            #models = self.run_parallel(jobs)

            flight_ids = {}
            all_sector_flights = []
            all_navpoint_flights = []
            sector_configs = []

            for model, sector_config_restore_dict in solutions:
                if self._optimizer == "ASP":
                    if int(str(model.get_sector_config().arguments[0])) > 0:
                        sector_configs.append((int(str(model.get_sector_config().arguments[0])), sector_config_restore_dict))

                    #for reroute in model.get_reroutes():
                    #    flight_id = int(str(reroute.arguments[0]))
                    #    flight_ids.append(flight_id)

                    for flight in model.get_sector_flights():

                        flight_id = int(str(flight.arguments[0]))
                        flight_ids[flight_id] = True

                        all_sector_flights.append(flight)

                    for flight in model.get_navpoint_flights():

                        flight_id = int(str(flight.arguments[0]))
                        flight_ids[flight_id] = True


                        all_navpoint_flights.append(flight)

                else:
                    new_flight_durations = {}
                    current_config = model[0][2]
                    sector_configs.append((current_config, sector_config_restore_dict))
                    flight_paths = model[0][3]
                    flight_path_dict = model[1]

                    for flight_index, path_number in flight_paths:

                        converted_instance_matrix[flight_index, :] = -1
                        converted_navpoint_matrix[flight_index, :] = -1

                        for navpoint, cur_time in flight_path_dict[flight_index][path_number]["navpoint_flight"]:
                            #navpoint, cur_time = flight_path_dict[flight_index][path_number]["navpoint_flight"][step_index]
                            converted_navpoint_matrix[flight_index, cur_time] = navpoint

                        for sector, cur_time in flight_path_dict[flight_index][path_number]["sector_flight"][current_config]:
                            #sector, cur_time = flight_path_dict[flight_index][path_number]["sector_flight"][step_index]
                            converted_instance_matrix[flight_index, cur_time] = sector

                            if flight_index not in new_flight_durations:
                                new_flight_durations[flight_index] = {}
                                new_flight_durations[flight_index]["min"] = cur_time
                                new_flight_durations[flight_index]["max"] = cur_time
                                new_flight_durations[flight_index]["duration"] = new_flight_durations[flight_index]["max"] - new_flight_durations[flight_index]["min"]

                            if cur_time < new_flight_durations[flight_index]["min"]:
                                new_flight_durations[flight_index]["min"] = cur_time

                            if cur_time > new_flight_durations[flight_index]["max"]:
                                new_flight_durations[flight_index]["max"] = cur_time

                            new_flight_durations[flight_index]["duration"] = new_flight_durations[flight_index]["max"] - new_flight_durations[flight_index]["min"]



                        for potentially_affected_flight_index in flight_path_dict[flight_index][path_number]["potential_flights_affected"].keys():
 
                            converted_instance_matrix[potentially_affected_flight_index, :] = -1
                            converted_navpoint_matrix[potentially_affected_flight_index, :] = -1

                            for navpoint, cur_time in flight_path_dict[flight_index][path_number]["potential_flights_affected"][potentially_affected_flight_index]["navpoint_flight"]:
                                #navpoint, cur_time = flight_path_dict[flight_index][path_number]["potential_flights_affected"][potentially_affected_flight_index]["navpoint_flight"][step_index]
                                converted_navpoint_matrix[flight_index, cur_time] = navpoint

                            for sector, cur_time in flight_path_dict[flight_index][path_number]["potential_flights_affected"][potentially_affected_flight_index]["sector_flight"][current_config]:
                                #sector, cur_time = flight_path_dict[flight_index][path_number]["potential_flights_affected"][potentially_affected_flight_index]["sector_flight"][step_index]
                                converted_instance_matrix[flight_index, cur_time] = sector

                                if potentially_affected_flight_index not in new_flight_durations:
                                    new_flight_durations[potentially_affected_flight_index] = {}
                                    new_flight_durations[potentially_affected_flight_index]["min"] = cur_time
                                    new_flight_durations[potentially_affected_flight_index]["max"] = cur_time
                                    new_flight_durations[potentially_affected_flight_index]["duration"] = new_flight_durations[potentially_affected_flight_index]["max"] - new_flight_durations[potentially_affected_flight_index]["min"]

                                if cur_time < new_flight_durations[potentially_affected_flight_index]["min"]:
                                    new_flight_durations[potentially_affected_flight_index]["min"] = cur_time

                                if cur_time > new_flight_durations[potentially_affected_flight_index]["max"]:
                                    new_flight_durations[potentially_affected_flight_index]["max"] = cur_time

                                new_flight_durations[potentially_affected_flight_index]["duration"] = new_flight_durations[potentially_affected_flight_index]["max"] - new_flight_durations[potentially_affected_flight_index]["min"]

            flight_ids = np.array(list(flight_ids.keys()), dtype=int)

            if len(sector_configs) > 0:
                if len(sector_configs) > 1:
                    print("[WARNING] - Multiple sector configs found which are different from the default one - taking the first one.")

                sector_config_number, sector_config_restore_dict = sector_configs[0]

                if self.verbosity > 2:
                    print("")
                    print(sector_config_number)
                    print("")

                if sector_config_number != 0:

                    tmp_composition_navpoints = sector_config_restore_dict[sector_config_number]["composition_navpoints"]
                    tmp_composition_sectors = sector_config_restore_dict[sector_config_number]["composition_sectors"]
                    tmp_loads_matrix = sector_config_restore_dict[sector_config_number]["demand"]
                    tmp_capacity_time_matrix = sector_config_restore_dict[sector_config_number]["capacity"]
                    tmp_navaid_sector_time_assignment = sector_config_restore_dict[sector_config_number]["composition"]
                    tmp_flights = sector_config_restore_dict[sector_config_number]["affected_flights"]
                    tmp_time_index = sector_config_restore_dict[sector_config_number]["time_index"]
                    tmp_sector_index = sector_config_restore_dict[sector_config_number]["sector_index"]

                    if tmp_navaid_sector_time_assignment.shape[1]  > navaid_sector_time_assignment.shape[1]:
                        # INCREASE MATRIX SIZE (TIME) AUTOMATICALLY
                        diff = tmp_navaid_sector_time_assignment.shape[1] - navaid_sector_time_assignment.shape[1]

                        in_units = math.ceil(diff / self._timestep_granularity)
                        number_new_cols = in_units * self._timestep_granularity

                        # 0.) Handle Sector Assignments:
                        new_cols = np.repeat(navaid_sector_time_assignment[:,[-1]], number_new_cols, axis=1)  # shape (N,k)
                        navaid_sector_time_assignment = np.concatenate([navaid_sector_time_assignment, new_cols], axis=1)
                        # 1.) Handle Instance Matrix:
                        extra_col = -1 * np.ones((converted_instance_matrix.shape[0], number_new_cols), dtype=int)
                        converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                        extra_col = -1 * np.ones((converted_navpoint_matrix.shape[0], number_new_cols), dtype=int)
                        converted_navpoint_matrix = np.hstack((converted_navpoint_matrix, extra_col)) 

                        # 2.) Create demand matrix (|R|x|T|):
                        system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                        # 3.) Create capacity matrix (|R|x|T|):
                        capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor, composite_sector_function=self.composite_sector_function)

                        # 4.) Subtract demand from capacity (|R|x|T|):
                        capacity_demand_diff_matrix = capacity_time_matrix - system_loads


                    #capacity_time_matrix[tmp_composition_sectors,:] = tmp_capacity_time_matrix
                    navaid_sector_time_assignment[tmp_composition_navpoints,:] = tmp_navaid_sector_time_assignment

                    converted_instance_matrix = OptimizeFlights.instance_computation_after_sector_change(tmp_flights, converted_navpoint_matrix, converted_instance_matrix, navaid_sector_time_assignment)

                    capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor,
                                                                                composite_sector_function=self.composite_sector_function)
                    

                    system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)

                    #capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                    #capacity_overload_mask = capacity_demand_diff_matrix < 0

                    #converted_instance_matrix[:,tmp_time_index:][np.isin(converted_instance_matrix[:,tmp_time_index:], tmp_composition_sectors)] = tmp_sector_index
                    #aggregated_demand = system_loads[tmp_composition_sectors, tmp_time_index:].sum(axis=0)
                    #system_loads[tmp_composition_sectors, tmp_time_index:] = 0
                    #system_loads[tmp_sector_index, tmp_time_index:] = aggregated_demand

                    #capacity_demand_diff_matrix = capacity_time_matrix - system_loads

            if self._optimizer == "ASP":

                flight_ids = np.array(flight_ids)
                # RMV CAPACITY FROM THIS
                capacity_demand_diff_matrix = self.system_loads_computation_v2(converted_instance_matrix, fill_value, flight_ids, capacity_demand_diff_matrix)

                converted_instance_matrix[flight_ids, :] = -1
                converted_navpoint_matrix[flight_ids, :] = -1

                new_flight_durations = {}
                for flight in all_sector_flights:
                    flight_id = int(str(flight.arguments[0]))
                    position_id = int(str(flight.arguments[1]))
                    time_id = int(str(flight.arguments[2]))

                    if time_id >= converted_instance_matrix.shape[1]:
                        print("[WARN] --> ADDED EXTRA TIME DUE TO NEW FLIGHT ASSIGNMENT")
                        #extra_col = -1 * np.ones((converted_instance_matrix.shape[0], 1), dtype=int)
                        #converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                        number_new_cols = 1 * self._timestep_granularity

                        # 0.) Handle Sector Assignments:
                        new_cols = np.repeat(navaid_sector_time_assignment[:,[-1]], number_new_cols, axis=1)  # shape (N,k)
                        navaid_sector_time_assignment = np.concatenate([navaid_sector_time_assignment, new_cols], axis=1)
                        # 1.) Converted instance matrix
                        extra_col = -1 * np.ones((converted_instance_matrix.shape[0], number_new_cols), dtype=int)
                        converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                        extra_col = -1 * np.ones((converted_navpoint_matrix.shape[0], number_new_cols), dtype=int)
                        converted_navpoint_matrix = np.hstack((converted_navpoint_matrix, extra_col)) 
                        
                        # 2.) Create demand matrix (|R|x|T|)
                        system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                        # 3.) Create capacity matrix (|R|x|T|)
                        #capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, z = self.sector_capacity_factor)
                        capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor, composite_sector_function=self.composite_sector_function)
                        # 4.) Subtract demand from capacity (|R|x|T|)
                        capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                        # 5.) Create capacity overload matrix
                        capacity_overload_mask = capacity_demand_diff_matrix < 0

                    #print(f"converted_instance_matrix[{flight_id},{time_id}] = {position_id}")
                    converted_instance_matrix[flight_id, time_id] = position_id

                    if flight_id not in new_flight_durations:
                        new_flight_durations[flight_id] = {}
                        new_flight_durations[flight_id]["min"] = time_id
                        new_flight_durations[flight_id]["max"] = time_id
                        new_flight_durations[flight_id]["duration"] = new_flight_durations[flight_id]["max"] - new_flight_durations[flight_id]["min"]

                    if time_id < new_flight_durations[flight_id]["min"]:
                        new_flight_durations[flight_id]["min"] = time_id

                    if time_id > new_flight_durations[flight_id]["max"]:
                        new_flight_durations[flight_id]["max"] = time_id

                    new_flight_durations[flight_id]["duration"] = new_flight_durations[flight_id]["max"] - new_flight_durations[flight_id]["min"]

                for navpoint_flight in all_navpoint_flights:

                    flight_id = int(str(navpoint_flight.arguments[0]))
                    navpoint_id = int(str(navpoint_flight.arguments[1]))
                    time_id = int(str(navpoint_flight.arguments[2]))
                    
                    #print(f"converted_navpoint_matrix[{flight_id},{time_id}] = {navpoint_id}")
                    converted_navpoint_matrix[flight_id, time_id] = navpoint_id

            # Rerun check if there are still things to solve:

            # TODO -> TIME DELTA EVALUTAION!
            # BETTER SYSTEM_LOADS/CAPACITY_TIME_MATRIX/etc. computation (as only 2 flights changed... never all sectors!)
            # MAYBE ALSO BETTER DURATION COMPUTATION?

            """
            flights_affected = converted_instance_matrix[flight_ids,:]
            to_change = (flights_affected[:,1:] != fill_value) & (flights_affected[:,1:] != flights_affected[:,:-1])
            to_change_first = np.reshape(flights_affected[:,0] != fill_value, (flights_affected.shape[0],1))
            to_change = np.hstack((to_change_first, to_change)) 
            to_change_indices = np.nonzero(to_change)
            flight_affected_buckets = flights_affected[to_change_indices]
            np.add.at(capacity_demand_diff_matrix, (flight_affected_buckets, to_change_indices[1]), -1)
            """

            system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
            capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor, composite_sector_function=self.composite_sector_function)

            if self.minimize_number_sectors is True:

                while global_t_start + self._timestep_granularity - 1 < time_bucket_updated:

                    global_t_end = global_t_start + self._timestep_granularity - 1

                    if global_t_end >= converted_instance_matrix.shape[1]:
                        global_t_end = converted_instance_matrix.shape[1] - 1

                    converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = self.minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, t_start, t_end, self.networkx_navpoint_graph, self.airports)
                    #converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = self.minimize_number_of_sectors(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, global_t_start, global_t_end, self.networkx_navpoint_graph, self.airports)

                    global_t_start =  global_t_start + self._timestep_granularity


            capacity_demand_diff_matrix = capacity_time_matrix - system_loads


            #np.testing.assert_array_equal(capacity_demand_diff_matrix, capacity_demand_diff_matrix_cpy)

            # OLD - Just number of conflicting sectors:
            #number_of_conflicts_prev = number_of_conflicts
            #number_of_conflicts = capacity_overload_mask.sum()

            # NEW - With absolute overload:
            number_of_conflicts_prev = number_of_conflicts
            capacity_overload_mask = capacity_demand_diff_matrix < 0
            number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()

            #print(number_of_conflicts)
            #quit()

            # HEURISTIC SELECTION OF COMPLEXITY OF TASK
            # -----------------------------------------------------------------------------
            if number_of_conflicts is not None and number_of_conflicts_prev is not None:
                if number_of_conflicts >= number_of_conflicts_prev:
                    counter_equal_solutions += 1

                    #if counter_equal_solutions >= 2 and counter_equal_solutions % 2 == 0:
                    additional_time_increase += 1
                    if self.verbosity > 1:
                        print(f">>> INCREASED TIME TO:{additional_time_increase}")

                    if counter_equal_solutions >= 5:
                        self.number_capacity_management_configs = 1
                        if self.verbosity > 1:
                            print(f">>> SET NUMBER DYNAMIC SECTOR ALLOCATION TO:{self.number_capacity_management_configs}")
                    
                    if counter_equal_solutions >= 10:
                        # Otherwise it can be that the considered planes hinder a solution through MUTEX
                        max_number_airplanes_considered_in_ASP = 1
                        if self.verbosity > 1:
                            print(f">>> SET MAX NUMBER AIRPLANES CONSIDERED TO:{max_number_airplanes_considered_in_ASP}")

                    if counter_equal_solutions >= 32 and counter_equal_solutions % 16 == 0:
                        self._max_explored_vertices = max(1,int(self._max_explored_vertices/2))
                        if self.verbosity > 1:
                            print(f">>> CONSIDERED VERTICES REDUCED TO:{self._max_explored_vertices}")

                    if counter_equal_solutions >= 64 and counter_equal_solutions % 32 == 0:
                        max_number_processors = max(1,int(max_number_processors / 2))
                        if self.verbosity > 1:
                            print(f">>> PARALLEL PROCESSORS REDUCED TO:{max_number_processors}")

                    navaid_sector_time_assignment = old_navaid_sector_time_assignment.copy()
                    converted_instance_matrix = old_converted_instance.copy()
                    converted_navpoint_matrix = old_converted_navpoint_matrix.copy()

                    system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)

                    capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self.sector_capacity_factor, composite_sector_function=self.composite_sector_function)
                    #capacity_time_matrix = OptimizeFlights.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, z = self.sector_capacity_factor)
                    capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                    capacity_overload_mask = capacity_demand_diff_matrix < 0

                    # Ensure: number_of_conflicts == number_of_conflicts_prev
                    number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()
                    number_of_conflicts_prev = number_of_conflicts


                else:
                    if self.verbosity > 1:
                        print(f">> ACCEPT SOLUTION| OLD = {number_of_conflicts_prev} > {number_of_conflicts} = NEW")
                    old_navaid_sector_time_assignment = navaid_sector_time_assignment.copy()
                    old_converted_instance = converted_instance_matrix.copy()
                    old_converted_navpoint_matrix = converted_navpoint_matrix.copy()

                    counter_equal_solutions = 0
                    max_number_processors = 20
                    max_number_airplanes_considered_in_ASP = 2
                    additional_time_increase = 0
                    self._max_explored_vertices = original_max_explored_vertices
                    self.number_capacity_management_configs = default_number_capacity_management_configs

                    for flight_id in new_flight_durations.keys():
                        flight_durations[flight_id] = new_flight_durations[flight_id]["duration"]

                    if max_number_processors < 20 or max_number_airplanes_considered_in_ASP > 2:
                        if self.verbosity > 1:
                            print(f">>> RESET PROCESSOR COUNT TO:{max_number_processors}; AIRPLANES TO: {max_number_airplanes_considered_in_ASP}")

            # -----------------------------------------------------------------------------
            t_init  = self.last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
            t_final = self.last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

            # --- 3. compute delays --------------------------------------------------------
            # Flights that disappear completely (-1 in *both* files) get a delay of 0
            delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

            # --- 4. aggregate in whichever way you need -----------------------------------
            total_delay  = delay.sum()

            if self.verbosity > 1:
                print(f">>> Current total delay: {total_delay}")
            
            iteration += 1

            number_sectors = self.compute_total_number_sectors(navaid_sector_time_assignment)
            sector_diff = np.count_nonzero(navaid_sector_time_assignment[:, 1:] != navaid_sector_time_assignment[:, :-1])

            number_sector_reconfigurations = np.count_nonzero(navaid_sector_time_assignment != old_navaid_sector_time_assignment)

            original_max_time_converted = original_converted_instance_matrix.shape[1]  # original_max_time
            rerouted_mask = np.any(converted_instance_matrix[:, :original_max_time_converted] != original_converted_instance_matrix, axis=1)     # True if flight differs anywhere
            number_reroutes = int(np.count_nonzero(rerouted_mask))

            if self._wandb_log is not None:
                if time_bucket_updated >= navaid_sector_time_assignment.shape[1]:
                    time_bucket_updated = navaid_sector_time_assignment.shape[1] - 1

                self._wandb_log({
                    "iteration": int(iteration),
                    "number_of_conflicts": int(number_of_conflicts),
                    "total_delay": int(total_delay),
                    "number_sectors": int(number_sectors),
                    "sector_diff": int(sector_diff),
                    "number_reroutes": int(number_reroutes),
                    "number_reconfigurations": int(number_sector_reconfigurations),
                    "current_time": int(time_bucket_updated),
                })

            if self.verbosity > 1:
                print(f"""
    iteration -> {int(iteration)}
    number_of_conflicts -> {int(number_of_conflicts)}
    total_delay -> {int(total_delay)}
    number_sectors -> {number_sectors}
    current_time -> {int(time_bucket_updated)}
                    """)
                    
            current_time = time.time() - original_start_time
            output_dict = {}
            output_dict["OVERLOAD"] = int(number_of_conflicts)
            output_dict["ARRIVAL-DELAY"] = int(total_delay)
            output_dict["SECTOR-NUMBER"] = int(number_sectors)
            output_dict["SECTOR-DIFF"] = int(sector_diff)
            output_dict["REROUTE"] = int(number_reroutes)
            output_dict["RECONFIG"] = int(number_sector_reconfigurations)
            output_dict["TOTAL-TIME-TO-THIS-POINT"] =  int(current_time)
            output_dict["COMPUTATION-FINISHED"] = False
            output_string = json.dumps(output_dict)
            print(output_string)


        if self.minimize_number_sectors is True:
            time_bucket_updated = converted_navpoint_matrix.shape[1] - 1

            while global_t_start + self._timestep_granularity - 1 <= time_bucket_updated:
                global_t_end = global_t_start + self._timestep_granularity - 1

                if global_t_end >= converted_instance_matrix.shape[1]:
                    global_t_end = converted_instance_matrix.shape[1] - 1

                converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = self.minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, t_start, t_end, self.networkx_navpoint_graph, self.airports)
                #converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = self.minimize_number_of_sectors(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, global_t_start, global_t_end, self.networkx_navpoint_graph, self.airports)
                global_t_start =  global_t_start + self._timestep_granularity




        #np.savetxt("01_final_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

        t_init  = self.last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
        t_final = self.last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

        # --- 3. compute delays --------------------------------------------------------
        # Flights that disappear completely (-1 in *both* files) get a delay of 0
        delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

        # --- 4. aggregate in whichever way you need -----------------------------------
        total_delay  = delay.sum()
        mean_delay   = delay.mean()
        max_delay    = delay.max()
        #per_flight   = delay.tolist()

        # Track to Weights & Biases when enabled
        number_sectors = self.compute_total_number_sectors(navaid_sector_time_assignment)
        sector_diff = np.count_nonzero(navaid_sector_time_assignment[:, 1:] != navaid_sector_time_assignment[:, :-1])

        number_sector_reconfigurations = np.count_nonzero(navaid_sector_time_assignment != old_navaid_sector_time_assignment)

        original_max_time_converted = original_converted_instance_matrix.shape[1]  # original_max_time
        rerouted_mask = np.any(converted_instance_matrix[:, :original_max_time_converted] != original_converted_instance_matrix, axis=1)     # True if flight differs anywhere
        number_reroutes = int(np.count_nonzero(rerouted_mask))

        if self._wandb_log is not None:
            self._wandb_log({
                "iteration": int(iteration+1),
                "number_of_conflicts": int(number_of_conflicts),
                "total_delay": int(total_delay),
                "number_sectors": int(number_sectors),
                "sector_diff": int(sector_diff),
                "number_reroutes": int(number_reroutes),
                "number_reconfigurations": int(number_sector_reconfigurations),
                "current_time": int(time_bucket_updated),
            })


        current_time = time.time() - original_start_time
        output_dict = {}
        output_dict["OVERLOAD"] = int(number_of_conflicts)
        output_dict["ARRIVAL-DELAY"] = int(total_delay)
        output_dict["SECTOR-NUMBER"] = int(number_sectors)
        output_dict["SECTOR-DIFF"] = int(sector_diff)
        output_dict["REROUTE"] = int(number_reroutes)
        output_dict["RECONFIG"] = int(number_sector_reconfigurations)
        output_dict["TOTAL-TIME-TO-THIS-POINT"] =  int(current_time)
        output_dict["COMPUTATION-FINISHED"] = True
        output_string = json.dumps(output_dict)
        print(output_string)



        if self.verbosity > 0:

            number_flights_delayed = sum([1 if cur_delay > 0 else 0 for cur_delay in delay])

            print("<<<<<<<<<<<<<<<<----------------->>>>>>>>>>>>>>>>")
            print("                  FINAL RESULTS")
            print("<<<<<<<<<<<<<<<<----------------->>>>>>>>>>>>>>>>")
            print(f"Number of delayed flights: {number_flights_delayed}")
            print(f"Total delay (all flights): {total_delay}")
            print(f"Average delay per flight:  {mean_delay:.2f}")
            print(f"Maximum single-flight delay: {max_delay}")

        self.total_atfm_delay = total_delay
        self.navaid_sector_time_assignment = navaid_sector_time_assignment
        self.converted_instance_matrix = converted_instance_matrix
        self.converted_navpoint_matrix = converted_navpoint_matrix
        self.capacity_time_matrix = capacity_time_matrix

        if self.verbosity > 1:
            np.savetxt("20250826_final_matrix.csv", converted_instance_matrix, delimiter=",", fmt="%i") 

    def get_total_atfm_delay(self):
        return self.total_atfm_delay
    
    def compute_total_number_sectors(self, navaid_sector_time_assignment):

        if navaid_sector_time_assignment.shape[0] == 0:
            number_sectors = 0
        else:
            S = np.sort(navaid_sector_time_assignment, axis=0)                       # sort within each column
            changes = (S[1:, :] != S[:-1, :])            # True where a new value starts
            uniq_per_col = 1 + changes.sum(axis=0)       # unique count per column
        number_sectors = int(uniq_per_col.sum())     # sum over all columns

        return number_sectors


    def build_job(self, time_index: int,
                sector_index: int,
                converted_instance_matrix: np.ndarray,
                converted_navpoint_matrix: np.ndarray,
                capacity_time_matrix: np.ndarray,
                system_loads: np.ndarray,
                capacity_demand_diff_matrix: np.ndarray,
                additional_time_increase: int,
                fill_value: int,
                max_rows: int,
                n_proc: int,
                original_max_time: int,
                networkx_graph,
                unit_graphs,
                planned_arrival_times,
                airplane_flight,
                airplanes,
                navaid_sector_time_assignment,
                duration,
                iteration,
                sector_capacity_factor,
                filed_flights,
                ):
        """
        Split the candidate rows into *n_proc* equally‑sized chunks
        (≤ max_rows each) and build one ``OptimizeFlights`` instance per chunk.
        """


        # DEMAND COMPUTATION WHEN FLIGHTS ENTER:
        #if time_index > 0:
        #    bucket_index_mask = (converted_instance_matrix[:, time_index] == sector_index) & (converted_instance_matrix[:, time_index - 1] != sector_index)
        #else:
        #    bucket_index_mask = converted_instance_matrix[:, time_index] == sector_index
        
        sector_flight_index_mask = converted_instance_matrix[:, time_index] == sector_index

        # Get flights for possible rerouting/delaying:
        candidate = np.flatnonzero(sector_flight_index_mask)

        #print(f"===> Trying to solve sector:{bucket_index} for time: {time_index}, with overload: {capacity_demand_diff_matrix[bucket_index, time_index]}, candidates: {len(candidate)}")
        #bucket_index_mask = bucket_index_mask.any(axis=1)


        candidate_duration = duration[candidate]

        # Sort Flights according to duration (ascending) and take the first ones:
        order = np.argsort(candidate_duration, kind="stable")
        candidate_sorted = candidate[order]
        candidate = candidate_sorted
        #rng.shuffle(candidate)

        capacity_difference = abs(capacity_demand_diff_matrix[sector_index, time_index])
        if capacity_difference < 2:
            capacity_difference = 2

        needed   = n_proc * max_rows
        needed = min(needed, capacity_difference)

        if self.verbosity > 2:
            print(f"---> ACTUALLY INVESTIGATED AIRPLANES: <<{needed}>>")

        rows_pool = candidate[:needed]

        chunk_size = max_rows
        n_chunks   = min(n_proc, len(rows_pool) // chunk_size)
        n_chunks = max(n_chunks, 3)

        chunk_index = 0
        problematic_flight_indices = rows_pool[chunk_index*chunk_size:(chunk_index+1)*chunk_size]

        # Get all associated problematic flights:
        airplane_indices = self.airplane_flight[np.isin(self.airplane_flight[:,1],problematic_flight_indices),0]
        airplane_flight_map = {}

        all_potentially_problematic_flight_indices = list(problematic_flight_indices)

        if self.verbosity > 1:
            print(f"--> Investigated sector: sec={sector_index}; time={time_index} with problematic flights: {all_potentially_problematic_flight_indices}")

        for airplane_index in airplane_indices:

            flights_per_airplane = self.airplane_flight[self.airplane_flight[:,0] == airplane_index, 1]
            flight_start_time_list = []

            for flight in flights_per_airplane:
                
                flight_operation_start_time = np.flatnonzero(converted_instance_matrix[flight,:] != fill_value)[0]

                if flight in problematic_flight_indices:
                    problematic_start_time = flight_operation_start_time

                flight_start_time_list.append((flight_operation_start_time,flight))

            flight_start_time_list = sorted(flight_start_time_list)

            considered_flight_indices = []

            for start_time, flight in flight_start_time_list:

                if start_time < problematic_start_time:
                    continue
                else:
                    considered_flight_indices.append(flight)

                    all_potentially_problematic_flight_indices.append(flight)

            airplane_flight_map[airplane_index] = considered_flight_indices

        all_potentially_problematic_flight_indices = np.array(list(set(all_potentially_problematic_flight_indices)))

        problematic_flights = converted_instance_matrix[problematic_flight_indices, :]
        #de_facto_max_time = converted_instance_matrix.shape[1]

        #capacity_demand_diff_matrix_tmp = self.system_loads_computation_v0(converted_instance_matrix, rows, capacity_time_matrix, de_facto_max_time)
        #capacity_demand_diff_matrix_cpy = self.system_loads_computation_v1(time_index, converted_instance_matrix, capacity_time_matrix, system_loads, fill_value, rows)

        # REMOVE PROBLEMATIC FLIGHTS FROM CAPACITY DEMAND MATRIX:
        # Create capacity demand diff matrix where only the values for the candidates have changed:
        capacity_demand_diff_matrix_cpy_2 = capacity_demand_diff_matrix.copy()
        capacity_demand_diff_matrix_cpy_2 = self.system_loads_computation_v2(converted_instance_matrix, fill_value, all_potentially_problematic_flight_indices, capacity_demand_diff_matrix_cpy_2)

        #np.testing.assert_array_equal(capacity_demand_diff_matrix_tmp, capacity_demand_diff_matrix_cpy)
        #np.testing.assert_array_equal(capacity_demand_diff_matrix_tmp, capacity_demand_diff_matrix_cpy_2)

        job = (self.encoding, self.sectors, self.graph, self.airports,
                                    problematic_flight_indices, 
                                    all_potentially_problematic_flight_indices,
                                    converted_instance_matrix, converted_navpoint_matrix,
                                    time_index,
                                    sector_index, capacity_time_matrix, capacity_demand_diff_matrix_cpy_2,
                                    additional_time_increase,
                                    networkx_graph, unit_graphs, planned_arrival_times,
                                    airplane_flight, airplanes, problematic_flights,
                                    navaid_sector_time_assignment,
                                    self.nearest_neighbors_lookup,
                                    sector_capacity_factor, filed_flights,
                                    airplane_flight_map,
                                    fill_value, self._timestep_granularity, self._seed,
                                    self._max_explored_vertices, self._max_delay_per_iteration, 
                                    original_max_time, iteration, self.verbosity,
                                    self.number_capacity_management_configs,
                                    self.capacity_management_enabled,
                                    self.composite_sector_function,
                                    self._optimizer,
                                    self.max_number_sectors,
                                    self._convex_sectors,
                                    )

        return job, rows_pool

    def system_loads_computation_v1(self, time_index, converted_instance_matrix, capacity_time_matrix, system_loads, fill_value, rows):
        system_loads_cpy = system_loads.copy()


        # THESE TWO METHODS ARE TOO SLOW:
        for row in rows:
            flight_affected = converted_instance_matrix[row,:]
            
            if time_index > 0:
                to_change = (flight_affected[1:] != fill_value) & (flight_affected[1:] != flight_affected[:-1])
                to_change_first = flight_affected[0] != fill_value
                to_change = np.hstack((to_change_first, to_change)) 
            else:
                raise NotImplementedError("NOT YET IMPLEMENTED!!!")
            
            to_change_indices = np.nonzero(to_change)[0]
            system_loads_cpy[flight_affected[to_change_indices], to_change_indices] = system_loads_cpy[flight_affected[to_change_indices], to_change_indices] - 1

        capacity_demand_diff_matrix_cpy = capacity_time_matrix - system_loads_cpy
        return capacity_demand_diff_matrix_cpy

    def system_loads_computation_v0(self, converted_instance_matrix, rows, capacity_time_matrix, de_facto_max_time):

        instance_matrix_cpy = converted_instance_matrix.copy()
        instance_matrix_cpy[rows,:] = -1

        system_loads_tmp = OptimizeFlights.bucket_histogram(instance_matrix_cpy, self.sectors, self.sectors.shape[0], de_facto_max_time, self._timestep_granularity)
        capacity_demand_diff_matrix_tmp = capacity_time_matrix - system_loads_tmp
        return capacity_demand_diff_matrix_tmp

    def system_loads_computation_v2(self, converted_instance_matrix, fill_value, rows, capacity_demand_diff_matrix):

        #system_loads_cpy_2 = system_loads.copy()

        flights_affected = converted_instance_matrix[rows,:]
        # First entered capacity:
        #to_change = (flights_affected[:,1:] != fill_value) & (flights_affected[:,1:] != flights_affected[:,:-1])
        #to_change_first = np.reshape(flights_affected[:,0] != fill_value, (flights_affected.shape[0],1))
        #to_change = np.hstack((to_change_first, to_change)) 

        # Capacity slots:
        to_change = flights_affected != fill_value
        to_change_indices = np.nonzero(to_change)
        flight_affected_buckets = flights_affected[to_change_indices]


        #np.subtract.at(system_loads_cpy_2, (flight_affected_buckets, to_change_indices[1]), 1)
        np.add.at(capacity_demand_diff_matrix, (flight_affected_buckets, to_change_indices[1]), 1)
        # THIS CAN BE IMPROVED

        #capacity_demand_diff_matrix_cpy_2 = capacity_time_matrix - system_loads_cpy_2

        return capacity_demand_diff_matrix
    
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
        
   
    # --- 2. helper: last non--1 position for every row, fully vectorised ----------
    def last_valid_pos(self, arr: np.ndarray) -> np.ndarray:
        """
        Return a 1-D array with, for every row in `arr`, the **last** column index
        whose value is not -1.  If a row is all -1, we return -1 for that flight.
        """
        # True where value ≠ -1
        mask = arr != -1                                        # same shape as arr

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

        if max_time_dim % time_granularity != 0:
            remainder = max_time_dim % time_granularity
            max_time_dim += time_granularity - remainder

            if max_time_dim % time_granularity != 0:
                print("[ERROR] - Should never occur - failure in maths")
                raise Exception("[ERROR IN COMPUTATION]")

        sectors = navaid_sector[:, 1]                      # shape (N,)

        largest_navaid = navaid_sector[navaid_sector.shape[0]-1,0]

        output = np.ones((largest_navaid+1, max_time_dim), dtype=int)  * (-1)

        for index in range(navaid_sector.shape[0]):
            output_index = navaid_sector[index,0]
            output[output_index,:] = navaid_sector[index,1]

        for index in range(output.shape[0]):

            if output[index,0] == -1:
                output[index,:] = index

        return output
        #np.repeat(sectors[:, None], max_time_dim, axis=1)  # shape (N, T)
     
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
        
    def first_overloads(self, mask: np.ndarray, k: int = 10):
        """
        Return up to the first `k` overloads as (time, bucket) pairs,
        ordered lexicographically by (time, bucket). If none found,
        return None.

        `mask` shape: (num_buckets, num_times)
                    rows   → bucket IDs
                    columns→ time steps
        """
        # 1) Find all time steps that have *any* overload (kept in ascending order)
        time_candidates = np.flatnonzero(mask.any(axis=0))

        results = []
        # 2) For each time, collect buckets with overloads (already ascending)
        for t in time_candidates:
            buckets = np.flatnonzero(mask[:, t])
            for b in buckets:
                results.append((t, b))
                if len(results) == k:
                    return results
        return results 

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
   
    def instance_navpoint_matrix(self, triplets, max_time, *, t0=0, fill_value=-1):
        """
        triplets: (N x 3) array-like of [flight_id, navpoint_id, time_first_reached]
        max_time: largest time index to include (inclusive)
        t0:       optional time offset (use t0=min_time if your times don't start at 0)
        """
        a = np.asarray(triplets, dtype=int)
        f = a[:, 0]
        n = a[:, 1]
        t = a[:, 2] - t0  # shift if needed so time starts at 0

        # Map possibly non-contiguous flight IDs to row indices
        flights, f_idx = np.unique(f, return_inverse=True)
        F = flights.size
        T = max_time - t0  # inclusive of max_time

        out = np.full((F, T), fill_value, dtype=int)

        # keep only times that land inside [0, T-1]
        m = (t >= 0) & (t < T)
        out[f_idx[m], t[m]] = n[m]  # last write wins if duplicates

        return out, flights  # 'flights' tells you which row corresponds to which flight_id

    def minimize_number_of_sectors_new(self, navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, max_number_navpoints_per_sector, max_number_sectors, t_start, t_end, navpoint_networkx_graph, airport_vertices):
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

        if t_start >= navaid_sector_time_assignment.shape[1]:
            return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads 

        if t_end - t_start > 0:
            not_changed_sectors = np.nonzero(np.all(navaid_sector_time_assignment[:,t_start:t_end] == navaid_sector_time_assignment[:,t_start+1:t_end+1], axis=1))[0]
        else:
            not_changed_sectors = np.nonzero(navaid_sector_time_assignment[:,t_start] != -1)[0]

        light_sector_mask = sector_has_navpoints & has_capacity & below_threshold
        light_sectors = np.nonzero(light_sector_mask)[0]

        light_sectors = np.intersect1d(not_changed_sectors, light_sectors)

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

        all_relevant_navaids = np.nonzero(np.isin(navaid_sector_time_assignment[:,t_start],light_sectors))[0]

        sub_graph = self.networkx_navpoint_graph.subgraph(list(all_relevant_navaids)).copy()

        for sector in light_sectors:
            navaids = np.nonzero(navaid_sector_time_assignment[:,t_start] == sector)[0]

            demand = system_loads[sector, t_start:t_end+1]
            capacity = capacity_time_matrix[sector, t_start:t_end+1]

            master_node = sector

            for navaid in navaids[1:]:
                nx.contracted_nodes(sub_graph, master_node, navaid, copy=False, self_loops=False)

            #sub_graph.collapse(navaids, into_node)
            nx.set_node_attributes(sub_graph, {master_node:{"demand":demand,"capacity":capacity, "navaids":list(navaids)}})

        unmarked = list(light_sectors).copy()

        new_sectors = {}

        while len(unmarked) > 0:

            seed = unmarked.pop(0)

            cur_demand = sub_graph.nodes[seed]["demand"]
            cur_capacity = sub_graph.nodes[seed]["capacity"]
            cur_nodes = sub_graph.nodes[seed]["navaids"]

            neighbors = list(sub_graph.neighbors(seed))

            neighbors = [neighbor for neighbor in neighbors if neighbor in unmarked]

            sector_bag = [seed]

            while len(neighbors) > 0:
                neighbor = neighbors.pop(0)
                if neighbor not in unmarked:
                    continue

                neigh_demand = sub_graph.nodes[neighbor]["demand"]
                neigh_capacity = sub_graph.nodes[neighbor]["capacity"]
                neigh_nodes = sub_graph.nodes[neighbor]["navaids"]

                demand = cur_demand + neigh_demand
                capacity = cur_capacity + neigh_capacity
                navaids = cur_nodes + neigh_nodes


                if len(navaids) > max_number_navpoints_per_sector:
                    continue

                if np.any(demand > capacity):
                    continue
                
                sector_bag.append(neighbor)

                neighbors_tmp = list(sub_graph.neighbors(neighbor))
                neighbors_tmp = [neighbor for neighbor in neighbors_tmp if neighbor in unmarked]
                neighbors = list(set(neighbors + neighbors_tmp))

                if seed in neighbors:
                    neighbors.remove(seed)
                
                nx.contracted_nodes(sub_graph, seed, neighbor, copy=False, self_loops=False)
                nx.set_node_attributes(sub_graph, {seed:{"demand":demand,"capacity":capacity, "navaids":navaids}})
                unmarked.remove(neighbor)

                cur_capacity = capacity
                cur_demand = demand
                cur_nodes = navaids

            new_sectors[seed] = sector_bag

        for sec in new_sectors.keys():
            merged_sectors = new_sectors[sec]
            capacity = sub_graph.nodes[sec]["capacity"]
            demand = sub_graph.nodes[sec]["demand"]
            navaids = sub_graph.nodes[sec]["navaids"]

            capacity_time_matrix[merged_sectors, t_start:t_end + 1] = 0
            system_loads[merged_sectors, t_start:t_end + 1] = 0

            capacity_time_matrix[sec, t_start:t_end + 1] = capacity
            system_loads[sec, t_start:t_end + 1] = demand

            navaid_sector_time_assignment[navaids, t_start:t_end + 1] = sec


        converted_instance_matrix = OptimizeFlights.instance_computation_after_sector_change(list(range(converted_instance_matrix.shape[0])),
                            converted_navpoint_matrix, converted_instance_matrix, navaid_sector_time_assignment)
        

        return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads



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
                        help="Defines capacity of composite sectors.")
    parser.add_argument("--convex-sectors", type=int, default=int(C("convex-sectors", 0)),
                        help="--convex-sectors=0 (false), --convex-sectors=1 (true)")
    parser.add_argument("--max-number-navpoints-per-sector", type=int, default=int(C("max-number-navpoints-per-sector", -1)),
                        help="Defines the maximum number of navpoints per composite sectors (-1=initial max number).")
    parser.add_argument("--max-number-sectors", type=int, default=int(C("max-number-sectors", -1)),
                        help="Defines the maximum number of sectors that can exist at one timepoint (-1=initial number).")
    parser.add_argument("--minimize-number-sectors-enabled", type=str, default=str(C("minimize-number-sectors-enabled", "false")),
                        help="true/false: If enabled, minimized every timestep-granularity the sectors.")

    parser.add_argument("--composite-sector-function", type=str, default=str(C("composite-sector-function", "max")),
                        help="Defines the function of the composite sector - available: max, triangular, linear")

                        

    # DYNAMIC SECTORIZATION:
    parser.add_argument("--number-capacity-management-configs", type=int, default=int(C("number-capacity-management-configs", 7)), help="How many compisitions/partitions to consider (only works when cap-mgmt. is enabled.")
    parser.add_argument("--capacity-management-enabled",
        type=str,
        default=str(C("capacity-management-enabled", "true")),
        help="true/false: true when cap-mgmt. is enabled.",
    )

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
    
    parser.add_argument("--optimizer", type=str, default=C("optimizer","ASP"), help="Either ASP or Enumerate")

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
    args.minimize_number_sectors = _str2bool(args.minimize_number_sectors_enabled)

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
        "capacity_time_matrix":     getattr(app, "capacity_time_matrix", None),
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
# Top-level script wrapper
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Script entry-point compatible with both `python -m` and `poetry run`."""
    args = parse_cli(argv)
    
    if args.verbosity > 0:
        # Optional: small echo of resolved inputs
        print("[i] Using inputs:")
        print(f"    graph:           {args.graph_path}")
        print(f"    sectors:         {args.sectors_path}")
        print(f"    flights:         {args.flights_path}")
        print(f"    airports:        {args.airports_path}")
        print(f"    airplanes:       {args.airplanes_path}")
        print(f"    airplane-flight: {args.airplane_flight_path}")
        print(f"    navaid-sector:   {args.navaid_sector_path}")
        print(f"    encoding:        {args.encoding_path}")
        if args.data_dir:
            print(f"    data-dir:        {args.data_dir}")
        if args.config:
            print(f"    config:          {args.config}")
        if args.wandb_enabled:
            print(f"    wandb:           entity={args.wandb_entity or '(default)'} project={args.wandb_project} "
                  f"name={args.wandb_experiment_name_prefix}{_derive_output_name(args)}{args.wandb_experiment_name_suffix}")


    composite_sector_function = args.composite_sector_function.lower()
    if composite_sector_function not in [MAX,LINEAR,TRIANGULAR]:
        raise Exception(f"Specified composite sector function {composite_sector_function} not in {[MAX,LINEAR,TRIANGULAR]}")

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
                "number_capacity_management_configs": args.number_capacity_management_configs,
                "capacity_management_enabled": args.capacity_management_enabled,
                "composite_sector_function": composite_sector_function,
                "seed": args.seed,
                "max_time": args.max_time,
            },
        )
        wandb_log = run.log

    app = Main(args.graph_path, args.sectors_path, args.flights_path,
               args.airports_path, args.airplanes_path,
               args.airplane_flight_path, args.navaid_sector_path,
               args.encoding_path,
               args.seed, args.number_threads, args.timestep_granularity,
               args.max_explored_vertices, args.max_delay_per_iteration,
               args.max_time, args.verbosity,
               args.sector_capacity_factor,
               args.number_capacity_management_configs,
               args.capacity_management_enabled,
               composite_sector_function,
               experiment_name,
               wandb_log,
               args.optimizer, args.max_number_navpoints_per_sector, args.max_number_sectors, args.minimize_number_sectors,
               args.convex_sectors)
    app.run()

    # Save results if requested
    if args.save_results:
        _save_results(args, app)

    if run is not None:
        run.finish()

    #print(app.get_total_atfm_delay())


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()

