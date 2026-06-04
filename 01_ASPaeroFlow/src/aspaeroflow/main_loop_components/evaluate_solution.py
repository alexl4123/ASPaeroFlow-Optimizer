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
from ..auxiliaries.computation_helpers import compute_total_number_sectors, last_valid_pos, system_loads_computation, minimize_number_of_sectors_new
from ..auxiliaries.communication_helpers import decode_ndarray, encode_ndarray

from ..optimize_flights import OptimizeFlights

class EvaluateSolution:

    def __init__(self, global_var_dto):
        convert_dto_to_global_vars(self, global_var_dto)

    def evaluate_solution(self, optimization_dto):

        navaid_sector_time_assignment = optimization_dto["navaid_sector_time_assignment"]
        time_bucket_updated = optimization_dto["time_bucket_updated"] 
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
        paused = optimization_dto["paused"]
        original_max_time = optimization_dto["original_max_time"] 
        planned_arrival_times = optimization_dto["planned_arrival_times"]
        flight_durations = optimization_dto["flight_durations"]
        original_max_explored_vertices = optimization_dto["original_max_explored_vertices"]
        default_number_capacity_management_configs = optimization_dto["default_number_capacity_management_configs"] 
        sector_index = optimization_dto["sector_index"] 
        global_t_start  = optimization_dto["global_t_start"] 

        old_converted_instance = optimization_dto["old_converted_instance"] 
        old_converted_navpoint_matrix = optimization_dto["old_converted_navpoint_matrix"] 
        old_navaid_sector_time_assignment = optimization_dto["old_navaid_sector_time_assignment"] 

        solutions = optimization_dto["solutions"] 
        iteration_backup = optimization_dto["iteration_backup"] 
        controller_sector_diff_dict = optimization_dto["controller_sector_diff_dict"] 

        flight_ids = {}
        all_sector_flights = []
        all_navpoint_flights = []
        sector_configs = []

        flight_per_navpoint_id = {}


        for model, sector_config_restore_dict, instance in solutions:
            if self._optimizer == "ASP":

                if self._controller_enabled is True:

                    config_restore_dict_tmp = {}

                    for key_1 in sector_config_restore_dict.keys():
                        config_restore_dict_tmp[key_1] = {}
                        config_restore_dict_tmp[key_1]["composition_navpoints"] = encode_ndarray(np.array(sector_config_restore_dict[key_1]["composition_navpoints"]))
                        config_restore_dict_tmp[key_1]["composition_sectors"] = encode_ndarray(sector_config_restore_dict[key_1]["composition_sectors"])
                        config_restore_dict_tmp[key_1]["demand"] = encode_ndarray(sector_config_restore_dict[key_1]["demand"])
                        config_restore_dict_tmp[key_1]["capacity"] = encode_ndarray(sector_config_restore_dict[key_1]["capacity"])
                        config_restore_dict_tmp[key_1]["composition"] = encode_ndarray(sector_config_restore_dict[key_1]["composition"])
                        config_restore_dict_tmp[key_1]["affected_flights"] = encode_ndarray(sector_config_restore_dict[key_1]["affected_flights"])
                        config_restore_dict_tmp[key_1]["time_index"] = int(sector_config_restore_dict[key_1]["time_index"])
                        config_restore_dict_tmp[key_1]["sector_index"] = int(sector_config_restore_dict[key_1]["sector_index"])

                    iteration_backup["SECTOR-CONFIG-RESTORE-DICT"] = config_restore_dict_tmp
                    iteration_backup["ASP-INSTANCE"] = base64.b64encode(instance.encode("utf-8")).decode("utf-8")

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

                    if flight_id not in flight_per_navpoint_id:
                        flight_per_navpoint_id[flight_id] = {}

                    navpoint_id = int(str(flight.arguments[1]))
                    time_id = int(str(flight.arguments[2]))

                    flight_per_navpoint_id[flight_id][time_id] = navpoint_id

                    all_navpoint_flights.append(flight)

        flight_ids = np.array(list(flight_ids.keys()), dtype=int)


        all_flights_diff_dict = {}
        if self._controller_enabled is True or self._explainability_context is not None: 

            for flight_id in flight_ids:
                navpoint_flight = converted_navpoint_matrix[flight_id,:]

                old_navpoint_flight_times = [int(i) for i in np.where(navpoint_flight != -1)[0]]
                new_navpoint_flight_times = sorted(list(flight_per_navpoint_id[flight_id].keys()))

                flights_equal = True

                if old_navpoint_flight_times == new_navpoint_flight_times:

                    for flight_time in old_navpoint_flight_times:
                        new_navpoint_id = flight_per_navpoint_id[flight_id][flight_time]
                        old_navpoint_id = int(navpoint_flight[flight_time])

                        if new_navpoint_id != old_navpoint_id:
                            flights_equal = False
                            break
                else:
                    flights_equal = False

                if flights_equal is False:
                    
                    old_flight_per_navpoint_id = {}
                    for flight_time in old_navpoint_flight_times:
                        old_navpoint_id = int(navpoint_flight[flight_time])
                        old_flight_per_navpoint_id[int(flight_time)] = old_navpoint_id

                    flight_diff_dict = {}
                    flight_diff_dict["id"] = int(flight_id)
                    flight_diff_dict["old_flight"] = old_flight_per_navpoint_id
                    flight_diff_dict["new_flight"] = flight_per_navpoint_id[flight_id]

                    all_flights_diff_dict[int(flight_id)] = flight_diff_dict

        if len(sector_configs) > 0:
            if len(sector_configs) > 1:
                print("[WARNING] - Multiple sector configs found which are different from the default one - taking the first one.")

            sector_config_number, sector_config_restore_dict = sector_configs[0]

            if self.verbosity > 2:
                print("")
                print(sector_config_number)
                print("")


            if sector_config_number != 0: # So there is a difference

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
            capacity_demand_diff_matrix = system_loads_computation(converted_instance_matrix, fill_value, flight_ids, capacity_demand_diff_matrix)

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

                    #number_new_cols = 1 * self._timestep_granularity
                    number_new_cols = int((math.ceil(float(time_id - converted_instance_matrix.shape[1]) / self._timestep_granularity) + 1) * self._timestep_granularity)

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

                #converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, t_start, t_end, self.networkx_navpoint_graph, self.airports)
                converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, global_t_start, global_t_end, self.networkx_navpoint_graph, self.airports)
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
        if number_of_conflicts is not None and number_of_conflicts_prev is not None and self._explainability_context is None:
            if number_of_conflicts >= number_of_conflicts_prev and self._sequential_execution is False:
                controller_sector_diff_dict["accepted_solution"] = False

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


            elif number_of_conflicts < number_of_conflicts_prev:
                if self.verbosity > 1:
                    print(f">> ACCEPT SOLUTION| OLD = {number_of_conflicts_prev} > {number_of_conflicts} = NEW")

                controller_sector_diff_dict["accepted_solution"] = True

                old_navaid_sector_time_assignment = navaid_sector_time_assignment.copy()
                old_converted_instance = converted_instance_matrix.copy()
                old_converted_navpoint_matrix = converted_navpoint_matrix.copy()

                counter_equal_solutions = 0
                max_number_processors = 20
                max_number_airplanes_considered_in_ASP = self.max_considered_aircraft
                additional_time_increase = 0
                self._max_explored_vertices = original_max_explored_vertices
                self.number_capacity_management_configs = default_number_capacity_management_configs

                for flight_id in new_flight_durations.keys():
                    flight_durations[flight_id] = new_flight_durations[flight_id]["duration"]

                if max_number_processors < 20 or max_number_airplanes_considered_in_ASP > 2:
                    if self.verbosity > 1:
                        print(f">>> RESET PROCESSOR COUNT TO:{max_number_processors}; AIRPLANES TO: {max_number_airplanes_considered_in_ASP}")

            else: # self._sequential_execution is True:
                controller_sector_diff_dict["accepted_solution"] = False

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

        elif self._explainability_context is not None:
            controller_sector_diff_dict["accepted_solution"] = True

        # -----------------------------------------------------------------------------
        t_init  = last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
        t_final = last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

        # --- 3. compute delays --------------------------------------------------------
        # Flights that disappear completely (-1 in *both* files) get a delay of 0
        delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

        # --- 4. aggregate in whichever way you need -----------------------------------
        total_delay  = delay.sum()

        if self.verbosity > 1:
            print(f">>> Current total delay: {total_delay}")
        
        iteration += 1

        number_sectors = compute_total_number_sectors(navaid_sector_time_assignment)
        sector_diff = np.count_nonzero(navaid_sector_time_assignment[:, 1:] != navaid_sector_time_assignment[:, :-1])

        diff_tmp = navaid_sector_time_assignment.shape[1] - original_navaid_sector_time_assignment.shape[1]
        tmp_navaid_sector_time_assignment = np.hstack([original_navaid_sector_time_assignment, np.repeat(original_navaid_sector_time_assignment[:, [-1]], diff_tmp, axis=1)])
        number_sector_reconfigurations = np.count_nonzero(navaid_sector_time_assignment != tmp_navaid_sector_time_assignment)

        original_max_time_converted = original_converted_instance_matrix.shape[1]  # original_max_time
        #rerouted_mask = np.any(converted_instance_matrix[:, :original_max_time_converted] != original_converted_instance_matrix, axis=1)     # True if flight differs anywhere
        rerouted_mask = np.any(converted_navpoint_matrix[:, :original_max_time_converted] != original_converted_navpoint_matrix, axis=1)     # True if flight differs anywhere
        number_reroutes = int(np.count_nonzero(rerouted_mask))

        if self._controller_enabled is True:
            relevant_vertices = np.array(controller_sector_diff_dict["prev_sector_config"][int(sector_index)]["vertices"])
            time_index = controller_sector_diff_dict["time_index"]
            unique_sectors = np.unique(navaid_sector_time_assignment[relevant_vertices, time_index])
            controller_sector_diff_dict["post_sector_config"] = {}
            for sector_index in unique_sectors:
                controller_sector_diff_dict["post_sector_config"][int(sector_index)] = {}
                controller_sector_diff_dict["post_sector_config"][int(sector_index)]["vertices"] = [int(i) for i in np.where(navaid_sector_time_assignment[:, time_index] == sector_index)[0]]
                controller_sector_diff_dict["post_sector_config"][int(sector_index)]["overload"] = int(-capacity_demand_diff_matrix[sector_index,time_index])

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
        output_dict["ITERATION"] = int(iteration)
        output_dict["OVERLOAD"] = int(number_of_conflicts)
        output_dict["ARRIVAL-DELAY"] = int(total_delay)
        output_dict["SECTOR-NUMBER"] = int(number_sectors)
        output_dict["SECTOR-DIFF"] = int(sector_diff)
        output_dict["REROUTE"] = int(number_reroutes)
        output_dict["RECONFIG"] = int(number_sector_reconfigurations)
        output_dict["TOTAL-TIME-TO-THIS-POINT"] =  int(current_time)
        output_dict["COMPUTATION-FINISHED"] = False
        output_string = json.dumps(output_dict)

        if self._explainability_context is None:
            print(output_string, flush=True)

        if self._controller_enabled is True or self._explainability_context is not None:

            output_dict["DIFF"] = {}
            # TODO -> STORE ALL STUFF NECESSARY FOR EXPLANATION!!!
            if self._controller_enabled is True:
                output_dict["ITERATION-BACKUP"] = iteration_backup

                output_dict["DIFF"]["ITERATION"] = int(iteration)
                output_dict["DIFF"]["ACCEPTED_SOLUTION"] = controller_sector_diff_dict["accepted_solution"]

                prev_sector_indices = list(controller_sector_diff_dict["prev_sector_config"].keys())
                post_sector_indices = list(controller_sector_diff_dict["post_sector_config"].keys())

                output_dict["DIFF"]["OPTIMIZED_SECTOR"] = list(controller_sector_diff_dict["prev_sector_config"].keys())
                if prev_sector_indices == post_sector_indices:
                    controller_sector_diff_dict = {}
                output_dict["DIFF"]["FLIGHTS"] = all_flights_diff_dict
                output_dict["DIFF"]["SECTORS"] = controller_sector_diff_dict

            if self._explainability_context is not None:
                output_dict["EXPLAIN"] = True
                output_dict["ITERATION"] = self._explainability_context["ITERATION"]
                output_dict["TYPE"] = self._explainability_context["TYPE"]
                output_dict["ID"] = self._explainability_context["ID"]

            output_string = json.dumps(output_dict)
            self._control_pub_socket.send_string(f"{output_string}")

            if self._explainability_context is not None:
                self._control_pub_socket.setsockopt(zmq.LINGER,0)
                self._control_pub_socket.close()
                quit()

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
        optimization_dto["flight_durations"] = flight_durations 
        optimization_dto["default_number_capacity_management_configs"] = default_number_capacity_management_configs 
        optimization_dto["paused"] = paused 

        optimization_dto["old_converted_instance"] = old_converted_instance 
        optimization_dto["old_converted_navpoint_matrix"] = old_converted_navpoint_matrix 
        optimization_dto["old_navaid_sector_time_assignment"] = old_navaid_sector_time_assignment 
        optimization_dto["sector_index"] = sector_index

        optimization_dto["solutions"] = solutions
        optimization_dto["iteration_backup"] = iteration_backup
        optimization_dto["controller_sector_diff_dict"] = controller_sector_diff_dict
        optimization_dto["original_max_explored_vertices"] = original_max_explored_vertices 
        optimization_dto["global_t_start"]  = global_t_start  
        optimization_dto["time_bucket_updated"] = time_bucket_updated 

        global_dto = convert_global_vars_to_dto(self)
        return optimization_dto, global_dto
