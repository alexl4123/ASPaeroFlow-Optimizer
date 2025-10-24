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

import numpy as np
import warnings

import pickle

import multiprocessing as mp

from pathlib import Path
from typing import Any, Final, List, Optional
from datetime import datetime, timezone

from solver import Solver, Model

from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    
from concurrent.futures import ProcessPoolExecutor
from optimize_flights import OptimizeFlights, MAX, TRIANGULAR, LINEAR
from concurrent.futures import ProcessPoolExecutor


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
        composite_sector_function: Optional[str]) -> None:

        self._graph_path: Optional[Path] = graph_path
        self._sectors_path: Optional[Path] = sectors_path
        self._flights_path: Optional[Path] = flights_path
        self._airports_path: Optional[Path] = airports_path
        self._airplanes_path: Optional[Path] = airplanes_path
        self._airplane_flight_path: Optional[Path] = airplane_flight_path
        self._navaid_sector_path: Optional[Path] = navaid_sector_path


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
        converted_instance_matrix, planned_arrival_times = OptimizeFlights.instance_to_matrix_vectorized(self.flights, self.airplane_flight, navaid_sector_time_assignment.shape[1], self._timestep_granularity, navaid_sector_time_assignment)

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

            _, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)


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

            #time_index,bucket_index = self.first_overload(capacity_overload_mask)

            max_number_processors = 1
            time_bucket_tuples = self.first_overloads(capacity_overload_mask, k=max_number_processors)

            start_time = time.time()

            all_jobs = []
            all_candidates = {}
            for time_index, sector_index in time_bucket_tuples:

                #print(capacity_demand_diff_matrix[sector_index, time_index])
                
                #_, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)
                job, candidates = self.build_job(time_index, sector_index, converted_instance_matrix, converted_navpoint_matrix,
                                capacity_time_matrix,
                                system_loads, capacity_demand_diff_matrix, additional_time_increase, fill_value, 
                                max_number_airplanes_considered_in_ASP, max_number_processors, original_max_time,
                                self.networkx_navpoint_graph, self.unit_graphs, planned_arrival_times, self.airplane_flight,
                                self.airplanes, navaid_sector_time_assignment, flight_durations,
                                iteration, self.sector_capacity_factor, self.flights,
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


            flight_ids = np.array(list(flight_ids.keys()), dtype=int)

            if len(sector_configs) > 0:
                if len(sector_configs) > 1:
                    print("[WARNING] - Multiple sector configs found which are different from the default one - taking the first one.")

                sector_config_number, sector_config_restore_dict = sector_configs[0]

                if sector_config_number != 0:

                    tmp_composition_navpoints = sector_config_restore_dict[sector_config_number]["composition_navpoints"]
                    tmp_composition_sectors = sector_config_restore_dict[sector_config_number]["composition_sectors"]
                    tmp_loads_matrix = sector_config_restore_dict[sector_config_number]["demand"]
                    tmp_capacity_time_matrix = sector_config_restore_dict[sector_config_number]["capacity"]
                    tmp_navaid_sector_time_assignment = sector_config_restore_dict[sector_config_number]["composition"]
                    tmp_time_index = sector_config_restore_dict[sector_config_number]["time_index"]
                    tmp_sector_index = sector_config_restore_dict[sector_config_number]["sector_index"]


                    capacity_time_matrix[tmp_composition_sectors,:] = tmp_capacity_time_matrix
                    navaid_sector_time_assignment[tmp_composition_navpoints,:] = tmp_navaid_sector_time_assignment

                    converted_instance_matrix[:,tmp_time_index:][np.isin(converted_instance_matrix[:,tmp_time_index:], tmp_composition_sectors)] = tmp_sector_index
                    aggregated_demand = system_loads[tmp_composition_sectors, tmp_time_index:].sum(axis=0)
                    system_loads[tmp_composition_sectors, tmp_time_index:] = 0
                    system_loads[tmp_sector_index, tmp_time_index:] = aggregated_demand

                    capacity_demand_diff_matrix = capacity_time_matrix - system_loads

            flight_ids = np.array(flight_ids)
            capacity_demand_diff_matrix = self.system_loads_computation_v2(converted_instance_matrix, fill_value, flight_ids, capacity_demand_diff_matrix)

            converted_instance_matrix[flight_ids, :] = -1
            converted_navpoint_matrix[flight_ids, :] = -1

            new_flight_durations = {}
            for flight in all_sector_flights:
                flight_id = int(str(flight.arguments[0]))
                position_id = int(str(flight.arguments[1]))
                time_id = int(str(flight.arguments[2]))

                if time_id >= converted_instance_matrix.shape[1]:
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
            capacity_demand_diff_matrix = capacity_time_matrix - system_loads


            #np.testing.assert_array_equal(capacity_demand_diff_matrix, capacity_demand_diff_matrix_cpy)

            # OLD - Just number of conflicting sectors:
            #number_of_conflicts_prev = number_of_conflicts
            #number_of_conflicts = capacity_overload_mask.sum()

            # NEW - With absolute overload:
            number_of_conflicts_prev = number_of_conflicts
            capacity_overload_mask = capacity_demand_diff_matrix < 0
            number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()

            # HEURISTIC SELECTION OF COMPLEXITY OF TASK
            # -----------------------------------------------------------------------------
            if number_of_conflicts is not None and number_of_conflicts_prev is not None:
                if number_of_conflicts >= number_of_conflicts_prev:
                    counter_equal_solutions += 1

                    #if counter_equal_solutions >= 2 and counter_equal_solutions % 2 == 0:
                    additional_time_increase += 1
                    if self.verbosity > 1:
                        print(f">>> INCREASED TIME TO:{additional_time_increase}")
                    
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

                    for flight_id in new_flight_durations.keys():
                        flight_durations[flight_id] = new_flight_durations[flight_id]["duration"]

                    if max_number_processors < 20 or max_number_airplanes_considered_in_ASP > 2:
                        if self.verbosity > 1:
                            print(f">>> RESET PROCESSOR COUNT TO:{max_number_processors}; AIRPLANES TO: {max_number_airplanes_considered_in_ASP}")

            # -----------------------------------------------------------------------------

            if self.verbosity > 1:
                t_init  = self.last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
                t_final = self.last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

                # --- 3. compute delays --------------------------------------------------------
                # Flights that disappear completely (-1 in *both* files) get a delay of 0
                delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

                # --- 4. aggregate in whichever way you need -----------------------------------
                total_delay  = delay.sum()

                print(f">>> Current total delay: {total_delay}")


            iteration += 1

            #if iteration == 372:
            #    print("DEBUG EXIT AT ITERATION 372!")
            #    quit()


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

        if self.verbosity > 1:
            np.savetxt("20250826_final_matrix.csv", converted_instance_matrix, delimiter=",", fmt="%i") 

    def get_total_atfm_delay(self):
        return self.total_atfm_delay

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

        # Create capacity demand diff matrix where only the values for the candidates have changed:
        capacity_demand_diff_matrix_cpy_2 = capacity_demand_diff_matrix.copy()
        capacity_demand_diff_matrix_cpy_2 = self.system_loads_computation_v2(converted_instance_matrix, fill_value, all_potentially_problematic_flight_indices, capacity_demand_diff_matrix_cpy_2)

        #np.testing.assert_array_equal(capacity_demand_diff_matrix_tmp, capacity_demand_diff_matrix_cpy)
        #np.testing.assert_array_equal(capacity_demand_diff_matrix_tmp, capacity_demand_diff_matrix_cpy_2)

        job = (self.encoding, self.sectors, self.graph, self.airports,
                                    problematic_flight_indices, converted_instance_matrix, converted_navpoint_matrix,
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

    def instance_to_matrix(self,
                            flights: np.ndarray,
                            airplane_flight: np.ndarray,
                            navaid_sector_time_assignment: np.ndarray,
                            max_time: int,
                            time_granularity: int,
                            *,
                            fill_value: int = -1,
                            compress: bool = False) -> np.ndarray:

        rows = airplane_flight[:,0].astype(int)

        vals = flights[:, 1]
        cols = flights[:, 2].astype(int)

        max_time = max(cols.max(), (max_time + 1) * time_granularity)

        out = np.full((rows.max() + 1, max_time),
                        fill_value,
                        dtype=vals.dtype)
        
        flight_ids = flights[:,0]
        flight_ids = np.unique(flight_ids)

        planned_arrival_times = {}

        for flight_id in flight_ids:

            airplane_id = (airplane_flight[airplane_flight[:,1] == flight_id])[0,0]

            current_flight = flights[flights[:,0] == flight_id]

            for flight_hop_index in range(current_flight.shape[0]):

                navaid = current_flight[flight_hop_index,1]
                time = current_flight[flight_hop_index,2]
                #sector = (navaid_sector[navaid_sector[:,0] == navaid])[0,1]
                sector = navaid_sector_lookup[navaid]

                if flight_hop_index == 0:
                    out[airplane_id,time] = sector
                else:

                    prev_navaid = current_flight[flight_hop_index - 1,1]
                    prev_time = current_flight[flight_hop_index - 1,2]
                    prev_sector = (navaid_sector[navaid_sector[:,0] == prev_navaid])[0,1]

                    for time_index in range(1, time-prev_time + 1):

                        if time_index <= math.floor((time - prev_time)/2):
                            out[airplane_id,prev_time + time_index] = prev_sector

                        else:
                            out[airplane_id,prev_time + time_index] = sector

                if flight_id not in planned_arrival_times:
                    planned_arrival_times[flight_id] = time
                elif planned_arrival_times[flight_id] < time:
                    planned_arrival_times[flight_id] = time

        #np.savetxt("20250826_converted_instance.csv", out, delimiter=",",fmt="%i")

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

        if max_time_dim % time_granularity != 0:
            remainder = max_time_dim % time_granularity
            max_time_dim += time_granularity - remainder

            if max_time_dim % time_granularity != 0:
                print("[ERROR] - Should never occur - failure in maths")
                raise Exception("[ERROR IN COMPUTATION]")

        sectors = navaid_sector[:, 1]                      # shape (N,)
        return np.repeat(sectors[:, None], max_time_dim, axis=1)  # shape (N, T)
           
    # TODO -> DELETE! 
    def instance_to_matrix_vectorized_TMP_DELETE(self,
                                    flights: np.ndarray,
                                    airplane_flight: np.ndarray,
                                    max_time: int,
                                    time_granularity: int,
                                    navaid_sector_time_assignment: np.ndarray,
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
        #max_time_dim = int(max(t.max(), (max_time + 1) * time_granularity))
        max_time_dim = max_time
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
   

    # ------------------------------------------------
    # Dynamic capacity matrix with composite sectors
    # ------------------------------------------------
    def capacity_time_matrix_DELETE(self,
                             cap: np.ndarray,
                             n_times: int,
                             time_granularity: int,
                             navaid_sector_time_assignment: np.ndarray,
                             z = 1) -> np.ndarray:
        """
        cap: shape (N, >=2), atomic per-block capacities in column 1 (ints)
        n_times: total number of time slots in the horizon
        time_granularity: number of slots per capacity-block (e.g., per hour)
        navaid_sector_time_assignment: (N, n_times) where entry [nav, t] is the
                                       sector-id (0..N-1) that nav belongs to at time t.
                                       Multiple navs can share the same sector-id at a given t,
                                       forming a composite sector.

        returns: (N, n_times) matrix of sector capacities per time slot.
                 Row index == sector-id.
        """

        # ---- validate / prep
        cap = np.asarray(cap)
        N = cap.shape[0]
        T = int(time_granularity)

        if n_times % T != 0:
            raise ValueError("n_times must be a multiple of time_granularity")

        if navaid_sector_time_assignment.shape != (N, n_times):
            raise ValueError(
                f"navaid_sector_time_assignment must be shape (N, n_times) = ({N}, {n_times}), "
                f"got {navaid_sector_time_assignment.shape}"
            ) 

        capacity = np.asarray(cap[:, 1], dtype=np.int64)  # per-block (e.g., per hour)
        base = capacity // T
        rem  = capacity %  T

        # template over a single block of length T
        template = np.broadcast_to(base[:, None], (N, T)).astype(np.int32, copy=True)

        max_r = int(rem.max())
        if max_r > 0:
            step = np.empty_like(rem, dtype=np.int64)
            np.floor_divide(T, rem, out=step, where=rem > 0)

            J = np.arange(max_r, dtype=np.int64)             # 0..max(rem)-1
            mask = J[None, :] < rem[:, None]                 # N × max_r
            rows2d = np.broadcast_to(np.arange(N)[:, None], (N, max_r))
            cols2d = step[:, None] * J[None, :]

            r_idx = rows2d[mask]
            c_idx = cols2d[mask]
            np.add.at(template, (r_idx, c_idx), 1)

        reps = n_times // T
        atomic_per_slot = np.tile(template, (1, reps))       # shape (N, n_times), int32

        # ---- Step 2: aggregate to current sector configuration per time
        # Vectorized linear case (sum over contributing navs) – equivalent to
        # compute_sector_capacity with a linear rule, then we apply z below.
        sector_cap = np.zeros((N, n_times), dtype=np.int64)

        # sector row indices for each (nav, t)
        sec_rows = navaid_sector_time_assignment.astype(np.int64, copy=False)
        if sec_rows.min() < 0 or sec_rows.max() >= N:
            raise ValueError("sector ids in navaid_sector_time_assignment must be in [0, N-1].")

        # time-column indices for each (nav, t)
        time_cols = np.broadcast_to(np.arange(n_times, dtype=np.int64), (N, n_times))

        # accumulate atomic capacities into their sector rows per time column
        np.add.at(sector_cap, (sec_rows, time_cols), atomic_per_slot.astype(np.int64, copy=False))

        # ---- Step 3: apply composite capacity rule (currently linear z * sum)
        # If you later change compute_sector_capacity to a non-linear rule,
        # replace the two lines below by the commented slow-but-explicit loop.
        #z = getattr(self, "z", 1.0)  # allow overriding via instance attribute
        #if z != 1.0:
        #    sector_cap = np.rint(sector_cap * float(z)).astype(np.int64, copy=False)
        #return sector_cap

        # -------------------------
        # Non-linear fallback (example):
        # -------------------------
        # If/when compute_sector_capacity becomes non-linear in the list of
        # atomic capacities, use this O(N * n_times) path instead of the
        # vectorized add.at above. Keep the atomic_per_slot computation.
        #
        sector_cap = np.zeros((N, n_times), dtype=np.int64)
        for t in range(n_times):
            # For each sector id s, collect contributing navs at time t
            for s in range(N):
                nav_mask = (navaid_sector_time_assignment[:, t] == s)
                if not np.any(nav_mask):
                    continue
                atomic_caps_here = atomic_per_slot[nav_mask, t]
                sector_cap[s, t] = self.compute_sector_capacity(atomic_caps_here, z)

        # --- DEBUG dump (optional)
        #np.savetxt("20251003_cap_mat.csv", sector_cap.astype(np.int64), delimiter=",", fmt="%i")
        #quit()

        # return sector_cap

    

    def capacity_time_matrix_TMP_DELETE(self,
                            cap: np.ndarray,
                            n_times: int,
                            time_granularity: int,
                            navaid_sector_time_assignment: np.ndarray) -> np.ndarray:
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

        # This is for debugging purposes only:
        np.savetxt("20251003_cap_mat.csv", cap_mat, delimiter=",",fmt="%i")

        return cap_mat
    
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
        T = max_time - t0 + 1  # inclusive of max_time

        out = np.full((F, T), fill_value, dtype=int)

        # keep only times that land inside [0, T-1]
        m = (t >= 0) & (t < T)
        out[f_idx[m], t[m]] = n[m]  # last write wins if duplicates

        return out, flights  # 'flights' tells you which row corresponds to which flight_id


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

    parser.add_argument("--composite-sector-function", type=str, default=str(C("composite-sector-function", "max")),
                        help="Defines the function of the composite sector - available: max, triangular, linear")


    parser.add_argument("--number-capacity-management-configs", type=int, default=int(C("number-capacity-management-configs", 7)), help="How many compisitions/partitions to consider (only works when cap-mgmt. is enabled.")
    parser.add_argument("--capacity-management-enabled",
        type=str,
        default=str(C("capacity-management-enabled", "true")),
        help="true/false: true when cap-mgmt. is enabled.",
    )

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


    composite_sector_function = args.composite_sector_function.lower()
    if composite_sector_function not in [MAX,LINEAR,TRIANGULAR]:
        raise Exception(f"Specified composite sector function {composite_sector_function} not in {[MAX,LINEAR,TRIANGULAR]}")

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
               composite_sector_function)
    app.run()

    # Save results if requested
    if args.save_results:
        _save_results(args, app)

    print(app.get_total_atfm_delay())


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()

