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
import pickle

import multiprocessing as mp

from pathlib import Path
from typing import Any, Final, List, Optional

from solver import Solver, Model

from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    
from concurrent.futures import ProcessPoolExecutor
from optimize_flights import OptimizeFlights
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

    def run(self) -> None:  
        """Run the application"""
        self.load_data()

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

        navaid_sector_lookup = {}
        for row_index in range(self.navaid_sector.shape[0]):
            navaid_sector_lookup[self.navaid_sector[row_index,0]] = self.navaid_sector[row_index, 1]


        # 1.) Create flights matrix (|F|x|T|) --> For easier matrix handling
        converted_instance_matrix, planned_arrival_times = self.instance_to_matrix_vectorized(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)
        #converted_instance_matrix, planned_arrival_times = self.instance_to_matrix(self.flights, self.airplane_flight, self.navaid_sector,  self._max_time, self._timestep_granularity, navaid_sector_lookup)

        #np.testing.assert_array_equal(converted_instance_matrix, converted_instance_matrix_2)
        #quit()

        # 2.) Create demand matrix (|R|x|T|)
        system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)

        # 3.) Create capacity matrix (|R|x|T|)
        capacity_time_matrix = self.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity)

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
                extra_col = -1 * np.ones((converted_instance_matrix.shape[0], in_units * self._timestep_granularity), dtype=int)
                converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 
                # 2.) Create demand matrix (|R|x|T|)
                system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                # 3.) Create capacity matrix (|R|x|T|)
                capacity_time_matrix = self.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity)
                # 4.) Subtract demand from capacity (|R|x|T|)
                capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                # 5.) Create capacity overload matrix
                capacity_overload_mask = capacity_demand_diff_matrix < 0

            #time_index,bucket_index = self.first_overload(capacity_overload_mask)

            max_number_processors = 1
            time_bucket_tuples = self.first_overloads(capacity_overload_mask, k=max_number_processors)

            start_time = time.time()

            all_jobs = []
            all_candidates = {}
            for time_index, bucket_index in time_bucket_tuples:

                #_, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)
                job, candidates = self.build_job(time_index, bucket_index, converted_instance_matrix, capacity_time_matrix,
                                system_loads, capacity_demand_diff_matrix, additional_time_increase, fill_value, 
                                max_number_airplanes_considered_in_ASP, max_number_processors, original_max_time,
                                self.networkx_navpoint_graph, self.unit_graphs, planned_arrival_times, self.airplane_flight, self.navaid_sector,
                                self.airplanes, navaid_sector_lookup, flight_durations,
                                iteration)
                
                all_new = True
                for candidate in candidates:
                    if candidate in all_candidates:
                        all_new = False
                    else:
                        all_candidates[candidate] = True

                if all_new:
                    all_jobs.append(job)

            models = Parallel(n_jobs=max_number_processors, backend="loky")(
                        delayed(_run)(job) for job in all_jobs)
            
            #models = [_run(job) for job in jobs]

            end_time = time.time()
            if self.verbosity > 1:
                print(f">> Elapsed solving time: {end_time - start_time}")

            #models = self.run_parallel(jobs)

            flight_ids = []
            all_flights = []
            for model in models:
                for reroute in model.get_reroutes():
                    flight_id = int(str(reroute.arguments[0]))
                    flight_ids.append(flight_id)

                for flight in model.get_flights():
                    all_flights.append(flight)

            flight_ids = np.array(flight_ids)
            capacity_demand_diff_matrix = self.system_loads_computation_v2(converted_instance_matrix, fill_value, flight_ids, capacity_demand_diff_matrix)
            converted_instance_matrix[flight_ids, :] = -1

            new_flight_durations = {}
            for flight in all_flights:
                flight_id = int(str(flight.arguments[0]))
                time_id = int(str(flight.arguments[1]))
                position_id = int(str(flight.arguments[2]))

                if time_id >= converted_instance_matrix.shape[1]:
                    #extra_col = -1 * np.ones((converted_instance_matrix.shape[0], 1), dtype=int)
                    #converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                    in_units = 1
                    extra_col = -1 * np.ones((converted_instance_matrix.shape[0], in_units * self._timestep_granularity), dtype=int)
                    converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                    # 2.) Create demand matrix (|R|x|T|)
                    system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                    # 3.) Create capacity matrix (|R|x|T|)
                    capacity_time_matrix = self.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity)
                    # 4.) Subtract demand from capacity (|R|x|T|)
                    capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                    # 5.) Create capacity overload matrix
                    capacity_overload_mask = capacity_demand_diff_matrix < 0

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



            # Rerun check if there are still things to solve:

            # TODO -> TIME DELTA EVALUTAION!
            # BETTER SYSTEM_LOADS/CAPACITY_TIME_MATRIX/etc. computation (as only 2 flights changed... never all sectors!)
            # MAYBE ALSO BETTER DURATION COMPUTATION?


            flights_affected = converted_instance_matrix[flight_ids,:]
            to_change = (flights_affected[:,1:] != fill_value) & (flights_affected[:,1:] != flights_affected[:,:-1])
            to_change_first = np.reshape(flights_affected[:,0] != fill_value, (flights_affected.shape[0],1))

            to_change = np.hstack((to_change_first, to_change)) 
            to_change_indices = np.nonzero(to_change)
            flight_affected_buckets = flights_affected[to_change_indices]

            np.add.at(capacity_demand_diff_matrix, (flight_affected_buckets, to_change_indices[1]), -1)

            #system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
            #capacity_time_matrix = self.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity)
            #capacity_demand_diff_matrix = capacity_time_matrix - system_loads

            capacity_overload_mask = capacity_demand_diff_matrix < 0

            #np.testing.assert_array_equal(capacity_demand_diff_matrix, capacity_demand_diff_matrix_cpy)

            # OLD - Just number of conflicting sectors:
            #number_of_conflicts_prev = number_of_conflicts
            #number_of_conflicts = capacity_overload_mask.sum()

            # NEW - With absolute overload:
            number_of_conflicts_prev = number_of_conflicts
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

                    if counter_equal_solutions >= 32 and counter_equal_solutions % 16 == 0:
                        self._max_explored_vertices = max(1,int(self._max_explored_vertices/2))
                        if self.verbosity > 1:
                            print(f">>> CONSIDERED VERTICES REDUCED TO:{self._max_explored_vertices}")
                    if counter_equal_solutions >= 64 and counter_equal_solutions % 32 == 0:
                        max_number_processors = max(1,int(max_number_processors / 2))
                        if self.verbosity > 1:
                            print(f">>> PARALLEL PROCESSORS REDUCED TO:{max_number_processors}")
                    elif counter_equal_solutions >= 128 and counter_equal_solutions % 64 == 0:
                        max_number_airplanes_considered_in_ASP += 1
                        if self.verbosity > 1:
                            print(f">>> INCREASED AIRPLANES CONSIDERED TO:{max_number_airplanes_considered_in_ASP}")

                    converted_instance_matrix = old_converted_instance

                    system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                    capacity_time_matrix = self.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity)
                    capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                    capacity_overload_mask = capacity_demand_diff_matrix < 0

                    number_of_conflicts_prev = number_of_conflicts
                    number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()


                else:
                    old_converted_instance = converted_instance_matrix.copy()
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

        print(total_delay)

        if self.verbosity > 1:
            np.savetxt("20250826_final_matrix.csv", converted_instance_matrix, delimiter=",", fmt="%i") 

    def build_job(self, time_index: int,
                bucket_index: int,
                converted_instance_matrix: np.ndarray,
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
                navaid_sector,
                airplanes,
                navaid_sector_lookup,
                duration,
                iteration,
                ):
        """
        Split the candidate rows into *n_proc* equally‑sized chunks
        (≤ max_rows each) and build one ``OptimizeFlights`` instance per chunk.
        """

        if time_index > 0:
            bucket_index_mask = (converted_instance_matrix[:, time_index] == bucket_index) & (converted_instance_matrix[:, time_index - 1] != bucket_index)
        else:
            bucket_index_mask = converted_instance_matrix[:, time_index] == bucket_index

        candidate = np.flatnonzero(bucket_index_mask)

        #print(f"===> Trying to solve sector:{bucket_index} for time: {time_index}, with overload: {capacity_demand_diff_matrix[bucket_index, time_index]}, candidates: {len(candidate)}")
        #bucket_index_mask = bucket_index_mask.any(axis=1)


        candidate_duration = duration[candidate]

        order = np.argsort(candidate_duration, kind="stable")
        candidate_sorted = candidate[order]
        candidate = candidate_sorted
        #rng.shuffle(candidate)

        capacity_difference = abs(capacity_demand_diff_matrix[bucket_index, time_index])
        if capacity_difference < 2:
            capacity_difference = 2

        needed   = n_proc * max_rows
        needed = min(needed, capacity_difference)

        if self.verbosity > 2:
            print(f"---> ACTUALLY INVESTIGATED AIRPLANES: <<{needed}>>")

        rows_pool = candidate[:needed]

        chunk_size = max_rows
        n_chunks   = min(n_proc, len(rows_pool) // chunk_size)
        n_chunks = max(n_chunks, 1)

        chunk_index = 0
        rows = rows_pool[chunk_index*chunk_size:(chunk_index+1)*chunk_size]

        flights = converted_instance_matrix[rows, :]
        de_facto_max_time = converted_instance_matrix.shape[1]

        #capacity_demand_diff_matrix_tmp = self.system_loads_computation_v0(converted_instance_matrix, rows, capacity_time_matrix, de_facto_max_time)
        #capacity_demand_diff_matrix_cpy = self.system_loads_computation_v1(time_index, converted_instance_matrix, capacity_time_matrix, system_loads, fill_value, rows)
        capacity_demand_diff_matrix_cpy_2 = capacity_demand_diff_matrix.copy()
        capacity_demand_diff_matrix_cpy_2 = self.system_loads_computation_v2(converted_instance_matrix, fill_value, rows, capacity_demand_diff_matrix_cpy_2)

        #np.testing.assert_array_equal(capacity_demand_diff_matrix_tmp, capacity_demand_diff_matrix_cpy)
        #np.testing.assert_array_equal(capacity_demand_diff_matrix_tmp, capacity_demand_diff_matrix_cpy_2)

        job = (self.encoding, self.sectors, self.graph, self.airports,
                                    flights, rows, converted_instance_matrix, time_index,
                                    bucket_index, capacity_time_matrix, capacity_demand_diff_matrix_cpy_2,
                                    additional_time_increase,
                                    networkx_graph, unit_graphs, planned_arrival_times,
                                    airplane_flight, airplanes, flights,
                                    navaid_sector, navaid_sector_lookup,
                                    self.nearest_neighbors_lookup,
                                    fill_value, self._timestep_granularity, self._seed,
                                    self._max_explored_vertices, self._max_delay_per_iteration, 
                                    original_max_time, iteration, self.verbosity)

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
        to_change = (flights_affected[:,1:] != fill_value) & (flights_affected[:,1:] != flights_affected[:,:-1])
        to_change_first = np.reshape(flights_affected[:,0] != fill_value, (flights_affected.shape[0],1))

        to_change = np.hstack((to_change_first, to_change)) 
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
                            navaid_sector: np.ndarray,
                            max_time: int,
                            time_granularity: int,
                            navaid_sector_lookup,
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
                            time_granularity: int) -> np.ndarray:

        template_matrix = np.zeros((cap.shape[0],time_granularity))

        for cap_index in range(cap.shape[0]):

            capacity = cap[cap_index, 1]

            per_timestep_capacity = math.floor(capacity / time_granularity)

            template_matrix[cap_index,:] = template_matrix[cap_index,:] + per_timestep_capacity

            # rem_cap < time_granularity per construction
            rem_cap = capacity - (per_timestep_capacity * time_granularity)
            step_size = math.floor(time_granularity / rem_cap)

            for time_index in range(0, time_granularity, step_size):
                
                template_matrix[cap_index, time_index] += 1
                rem_cap -= 1

                if rem_cap <= 0:
                    break

        # template_matrix: shape (N, T)
        reps = n_times // template_matrix.shape[1]  # requires n_times % T == 0
        if n_times % template_matrix.shape[1] != 0:
            raise ValueError("n_times must be a multiple of template_matrix.shape[1]")

        cap_mat = np.tile(template_matrix, (1, reps))  # shape (N, MAX-T)


        # DEBUG ONLY:
        # np.savetxt("20250819_cap_mat.csv", cap_mat, delimiter=",",fmt="%i")

        return cap_mat
    

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


 
