import types

import argparse
import sys
import time
import math
import networkx as nx
import os
import json

import base64
import zmq


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
import warnings

from ..auxiliaries.dto_helpers import convert_dto_to_global_vars, convert_global_vars_to_dto
from ..auxiliaries.computation_helpers import compute_total_number_sectors, last_valid_pos, system_loads_computation
from ..optimize_flights import OptimizeFlights
from ..auxiliaries.communication_helpers import decode_ndarray, encode_ndarray
from ..solver import Solver

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

class IterationStep:

    def __init__(self, global_var_dto):
        convert_dto_to_global_vars(self, global_var_dto)

    def optimization_step(self, optimization_dto):

        navaid_sector_time_assignment = optimization_dto["navaid_sector_time_assignment"]
        converted_instance_matrix = optimization_dto["converted_instance_matrix"]
        converted_navpoint_matrix = optimization_dto["converted_navpoint_matrix"]
        system_loads = optimization_dto["system_loads"]
        capacity_time_matrix = optimization_dto["capacity_time_matrix"]
        original_converted_instance_matrix = optimization_dto["original_converted_instance_matrix"]
        original_converted_navpoint_matrix = optimization_dto["original_converted_navpoint_matrix"]
        original_navaid_sector_time_assignment = optimization_dto["original_navaid_sector_time_assignment"]
        capacity_demand_diff_matrix = optimization_dto["capacity_demand_diff_matrix"]
        capacity_overload_mask = optimization_dto["capacity_overload_mask"]
        number_of_conflicts = optimization_dto["number_of_conflicts"]
        counter_equal_solutions = optimization_dto["counter_equal_solutions"]
        additional_time_increase = optimization_dto["additional_time_increase"]
        max_number_airplanes_considered_in_ASP = optimization_dto["max_number_airplanes_considered_in_ASP"]
        iteration = optimization_dto["iteration"]
        fill_value = optimization_dto["fill_value"]
        max_number_processors = optimization_dto["max_number_processors"]
        seed = optimization_dto["seed"]
        original_start_time = optimization_dto["original_start_time"]
        original_max_time = optimization_dto["original_max_time"]
        planned_arrival_times = optimization_dto["planned_arrival_times"]
        paused = optimization_dto["paused"]
        flight_durations = optimization_dto["flight_durations"]
        original_max_explored_vertices = optimization_dto["original_max_explored_vertices"] 
        sector_index = optimization_dto["sector_index"] 
        global_t_start = optimization_dto["global_t_start"] 
        time_bucket_updated = optimization_dto["time_bucket_updated"] 

        old_converted_instance = optimization_dto["old_converted_instance"] 
        old_converted_navpoint_matrix = optimization_dto["old_converted_navpoint_matrix"]
        old_navaid_sector_time_assignment = optimization_dto["old_navaid_sector_time_assignment"]

        solutions = optimization_dto["solutions"] 
        iteration_backup = optimization_dto["iteration_backup"] 
        controller_sector_diff_dict = optimization_dto["controller_sector_diff_dict"] 
        default_number_capacity_management_configs = optimization_dto["default_number_capacity_management_configs"]


        additional_return_parameters = []

        if self._explainability_context is None:
            if self._controller_enabled is True:

                iteration_backup = {}
                iteration_backup["NAVAID-SECTOR-TIME-ASSIGNMENT"] = encode_ndarray(navaid_sector_time_assignment.copy())
                iteration_backup["CONVERTED-INSTANCE-MATRIX"] = encode_ndarray(converted_instance_matrix.copy())
                iteration_backup["CONVERTED-NAVPOINT-MATRIX"] = encode_ndarray(converted_navpoint_matrix.copy())
                iteration_backup["SYSTEM-LOADS"] = encode_ndarray(system_loads.copy())
                iteration_backup["CAPACITY-TIME-MATRIX"] = encode_ndarray(capacity_time_matrix.copy())
                iteration_backup["SECTORS"] = encode_ndarray(self.sectors.copy())
                iteration_backup["ORIGINAL-CONVERTED-INSTANCE-MATRIX"] = encode_ndarray(original_converted_instance_matrix.copy())
                iteration_backup["ORIGINAL-NAVAID-SECTOR-TIME-ASSIGNMENT"] = encode_ndarray(original_navaid_sector_time_assignment.copy())

                # A. State Machine for Interrupts
                events = dict(self._control_poller.poll(timeout=0))

                for event_key in events.keys():
                    if self._control_ctrl_socket == event_key:
                        command = self._control_ctrl_socket.recv_string()
                        if command == "PAUSE":
                            paused = True
                            print("[CONTROL->OPTIMIZER]: PAUSE")
                            self._control_ctrl_socket.send_string("TELEMETRY: [STATUS] PAUSED")
                        elif command == "START":
                            paused = False
                            print("[CONTROL->OPTIMIZER]: START")
                            self._control_ctrl_socket.send_string("TELEMETRY: [STATUS] RESUMED")
                            print("SENT RESUMED")
                        elif command.startswith("<LOAD>"):
                            print("[CONTROL->OPTIMIZER]: LOAD")
                            global_dto = convert_global_vars_to_dto(self)
                            return optimization_dto, global_dto, ("<LOAD>", command[6:])
                        elif command.startswith("<OPTION>"):
                            command = command[8:]
                            command = json.loads(command)
                            for key in command.keys():
                                if key == "timestep_granularity":
                                    self._timestep_granularity = int(command[key])
                                elif key == "max_explored_vertices":
                                    self._max_explored_vertices = int(command[key])
                                    original_max_explored_vertices = self._max_explored_vertices
                                elif key == "max_delay_per_iteration":
                                    self._max_delay_per_iteration = int(command[key])
                                elif key == "number_capacity_management_configs":
                                    self.number_capacity_management_configs = int(command[key])
                                    default_number_capacity_management_configs = self.number_capacity_management_configs
                                else:
                                    print(f"NOT FOUND ATTR:{key}:{command[key]}")
                    else:
                        print("WEIRD BEHAVIOR IN CONNECTION - self._control_ctrl_socket not there")
                        print(events)
                        
                # B. Blocking Wait Loop (halts heuristic progression)
                if paused:
                    # poller.poll() with None blocks indefinitely until I/O occurs
                    events = dict(self._control_poller.poll(timeout=None))
                    if self._control_ctrl_socket in events:
                        command = self._control_ctrl_socket.recv_string()
                        if command == "START":
                            print("[CONTROL->OPTIMIZER]: START")
                            paused = False
                            self._control_ctrl_socket.send_string("TELEMETRY: [STATUS] RESUMED")
                        elif command.startswith("<LOAD>"):
                            print("[CONTROL->OPTIMIZER]: LOAD")
                            global_dto = convert_global_vars_to_dto(self)
                            return optimization_dto, global_dto, ("<LOAD>", command[6:])
                        elif command.startswith("<OPTION>"):
                            command = command[8:]
                            command = json.loads(command)
                            for key in command.keys():
                                if key == "timestep_granularity":
                                    self._timestep_granularity = int(command[key])
                                elif key == "max_explored_vertices":
                                    self._max_explored_vertices = int(command[key])
                                    original_max_explored_vertices = self._max_explored_vertices
                                elif key == "max_delay_per_iteration":
                                    self._max_delay_per_iteration = int(command[key])
                                elif key == "number_capacity_management_configs":
                                    self.number_capacity_management_configs = int(command[key])
                                    default_number_capacity_management_configs = self.number_capacity_management_configs
                                else:
                                    print(f"NOT FOUND ATTR:{key}:{command[key]}")

                    optimization_dto["paused"] = paused
                    global_dto = convert_global_vars_to_dto(self)
                    return optimization_dto, global_dto, "continue"

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

            if len(time_bucket_tuples) == 0:
                additional_return_parameters.append("SEQUENTIAL END")

            controller_sector_diff_dict = {}

            all_jobs = []
            all_candidates = {}
            for time_index, sector_index in time_bucket_tuples:
            
                time_bucket_updated = time_index

                #print(capacity_demand_diff_matrix[sector_index, time_index])

                self._sequential_execution_candidate_dict[(time_index,sector_index)] = True
                
                #_, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)
                job, candidates = self.build_job(time_index, sector_index, converted_instance_matrix, converted_navpoint_matrix,
                                capacity_time_matrix,
                                system_loads, capacity_demand_diff_matrix, additional_time_increase, fill_value, 
                                max_number_airplanes_considered_in_ASP, max_number_processors, original_max_time,
                                self.networkx_navpoint_graph, self.unit_graphs, planned_arrival_times, self.airplane_flight,
                                self.airplanes, navaid_sector_time_assignment, flight_durations,
                                iteration, self.sector_capacity_factor, self.flights,
                                self._sequential_execution
                                )

                controller_sector_diff_dict["time_index"] = int(time_index)
                controller_sector_diff_dict["sector_index"] = int(sector_index)
                controller_sector_diff_dict["prev_sector_config"] = {}
                controller_sector_diff_dict["prev_sector_config"][int(sector_index)] = {}
                controller_sector_diff_dict["prev_sector_config"][int(sector_index)]["vertices"] = [int(i) for i in np.where(navaid_sector_time_assignment[:, time_index] == sector_index)[0]]
                controller_sector_diff_dict["prev_sector_config"][int(sector_index)]["overload"] = int(-capacity_demand_diff_matrix[sector_index,time_index])
                
                all_new = True
                for candidate in candidates:
                    if candidate in all_candidates:
                        all_new = False
                    else:
                        all_candidates[candidate] = True

                if all_new:
                    all_jobs.append(job)

            #solutions = Parallel(n_jobs=max_number_processors, backend="loky")(
            #            delayed(_run)(job) for job in all_jobs)

            solutions = [_run(job) for job in all_jobs]

            end_time = time.time()
            if self.verbosity > 1:
                print(f">> Elapsed solving time: {end_time - start_time}")

            #models = self.run_parallel(jobs)

        else: # explainability_contex is not None:

            controller_sector_diff_dict = {}
            sector_config_restore_dict = {}
            
            for key_1 in self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"].keys():
                sector_config_restore_dict[int(key_1)] = {}
                sector_config_restore_dict[int(key_1)]["composition_navpoints"] = decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["composition_navpoints"])
                sector_config_restore_dict[int(key_1)]["composition_sectors"] = decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["composition_sectors"])
                sector_config_restore_dict[int(key_1)]["demand"] = decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["demand"])
                sector_config_restore_dict[int(key_1)]["capacity"] = decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["capacity"])
                sector_config_restore_dict[int(key_1)]["composition"] = decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["composition"])
                sector_config_restore_dict[int(key_1)]["affected_flights"] = decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["affected_flights"])
                sector_config_restore_dict[int(key_1)]["time_index"] = int(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["time_index"])
                sector_config_restore_dict[int(key_1)]["sector_index"] = int(self._explainability_context["ITERATION-BACKUP"]["SECTOR-CONFIG-RESTORE-DICT"][key_1]["sector_index"])

            instance = self._explainability_context["ITERATION-BACKUP"]["ASP-INSTANCE"]
            encoding = self.encoding

            solver: Model = Solver(encoding, instance)
            model = solver.solve()

            solutions = [(model, sector_config_restore_dict, instance)]


        optimization_dto["flight_durations"] = flight_durations
        optimization_dto["navaid_sector_time_assignment"] = navaid_sector_time_assignment
        optimization_dto["converted_instance_matrix"] = converted_instance_matrix
        optimization_dto["converted_navpoint_matrix"] = converted_navpoint_matrix
        optimization_dto["system_loads"] = system_loads
        optimization_dto["capacity_time_matrix"] = capacity_time_matrix
        optimization_dto["original_converted_instance_matrix"] = original_converted_instance_matrix
        optimization_dto["original_converted_navpoint_matrix"] = original_converted_navpoint_matrix
        optimization_dto["original_navaid_sector_time_assignment"] = original_navaid_sector_time_assignment
        optimization_dto["capacity_demand_diff_matrix"] = capacity_demand_diff_matrix
        optimization_dto["capacity_overload_mask"] = capacity_overload_mask
        optimization_dto["number_of_conflicts"] = number_of_conflicts
        optimization_dto["counter_equal_solutions"] = counter_equal_solutions
        optimization_dto["additional_time_increase"] = additional_time_increase
        optimization_dto["max_number_airplanes_considered_in_ASP"] = max_number_airplanes_considered_in_ASP
        optimization_dto["iteration"] = iteration
        optimization_dto["fill_value"] = fill_value
        optimization_dto["max_number_processors"] = max_number_processors
        optimization_dto["seed"] = seed
        optimization_dto["original_start_time"] = original_start_time
        optimization_dto["original_max_time"] = original_max_time
        optimization_dto["planned_arrival_times"] = planned_arrival_times
        optimization_dto["paused"] = paused 
        optimization_dto["global_t_start"] = global_t_start 
        optimization_dto["time_bucket_updated"] = time_bucket_updated

        optimization_dto["old_converted_instance"] = old_converted_instance 
        optimization_dto["old_converted_navpoint_matrix"] = old_converted_navpoint_matrix 
        optimization_dto["old_navaid_sector_time_assignment"] = old_navaid_sector_time_assignment 

        optimization_dto["solutions"] = solutions
        optimization_dto["iteration_backup"] = iteration_backup
        optimization_dto["controller_sector_diff_dict"] = controller_sector_diff_dict
        optimization_dto["original_max_explored_vertices"] = original_max_explored_vertices 
        optimization_dto["default_number_capacity_management_configs"] = default_number_capacity_management_configs 
        optimization_dto["sector_index"] = sector_index

        global_dto = convert_global_vars_to_dto(self)

        last_return_string = ""
        for parameter in additional_return_parameters:
            last_return_string += parameter + ","

        return optimization_dto, global_dto, last_return_string

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
                if self._sequential_execution is False or (t,b) not in self._sequential_execution_candidate_dict:
                    results.append((t, b))
                    if len(results) == k:
                        return results
        return results 

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
                sequential_execution
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
        capacity_demand_diff_matrix_cpy_2 = system_loads_computation(converted_instance_matrix, fill_value, all_potentially_problematic_flight_indices, capacity_demand_diff_matrix_cpy_2)

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
                                    sequential_execution,
                                    )

        return job, rows_pool
