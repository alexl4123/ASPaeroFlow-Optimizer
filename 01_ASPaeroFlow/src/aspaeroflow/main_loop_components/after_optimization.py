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
from ..auxiliaries.computation_helpers import compute_total_number_sectors, last_valid_pos, minimize_number_of_sectors_new
from ..optimize_flights import OptimizeFlights


class AfterOptimization:

    def __init__(self, global_var_dto):
        convert_dto_to_global_vars(self, global_var_dto)

    def post_processing(self, optimization_dto):

        navaid_sector_time_assignment = optimization_dto["navaid_sector_time_assignment"]
        converted_instance_matrix = optimization_dto["converted_instance_matrix"]
        converted_navpoint_matrix = optimization_dto["converted_navpoint_matrix"]
        system_loads = optimization_dto["system_loads"]
        capacity_time_matrix = optimization_dto["capacity_time_matrix"]
        original_converted_instance_matrix = optimization_dto["original_converted_instance_matrix"]
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

        global_t_start = optimization_dto["global_t_start"] 
        time_bucket_updated = optimization_dto["time_bucket_updated"] 

        old_converted_instance = optimization_dto["old_converted_instance"] 
        old_converted_navpoint_matrix = optimization_dto["old_converted_navpoint_matrix"] 
        old_navaid_sector_time_assignment = optimization_dto["old_navaid_sector_time_assignment"] 

        solutions = optimization_dto["solutions"] 
        iteration_backup = optimization_dto["iteration_backup"] 
        controller_sector_diff_dict = optimization_dto["controller_sector_diff_dict"] 

        if self.minimize_number_sectors is True:
            time_bucket_updated = converted_navpoint_matrix.shape[1] - 1

            while global_t_start + self._timestep_granularity - 1 <= time_bucket_updated:
                global_t_end = global_t_start + self._timestep_granularity - 1

                if global_t_end >= converted_instance_matrix.shape[1]:
                    global_t_end = converted_instance_matrix.shape[1] - 1

                converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, global_t_start, global_t_end, self.networkx_navpoint_graph, self.airports)
                #converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = self.minimize_number_of_sectors(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, global_t_start, global_t_end, self.networkx_navpoint_graph, self.airports)
                global_t_start =  global_t_start + self._timestep_granularity

        #np.savetxt("01_final_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

        t_init  = last_valid_pos(original_converted_instance_matrix)      # last non--1 in the *initial* schedule
        t_final = last_valid_pos(converted_instance_matrix)     # last non--1 in the *final* schedule

        # --- 3. compute delays --------------------------------------------------------
        # Flights that disappear completely (-1 in *both* files) get a delay of 0
        delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)

        # --- 4. aggregate in whichever way you need -----------------------------------
        total_delay  = delay.sum()
        mean_delay   = delay.mean()
        max_delay    = delay.max()
        #per_flight   = delay.tolist()

        # Track to Weights & Biases when enabled
        number_sectors = compute_total_number_sectors(navaid_sector_time_assignment)
        sector_diff = np.count_nonzero(navaid_sector_time_assignment[:, 1:] != navaid_sector_time_assignment[:, :-1])

        diff_tmp = navaid_sector_time_assignment.shape[1] - original_navaid_sector_time_assignment.shape[1]
        tmp_navaid_sector_time_assignment = np.hstack([original_navaid_sector_time_assignment, np.repeat(original_navaid_sector_time_assignment[:, [-1]], diff_tmp, axis=1)])
        number_sector_reconfigurations = np.count_nonzero(navaid_sector_time_assignment != tmp_navaid_sector_time_assignment)

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
        output_dict["ITERATION"] = int(iteration)
        output_dict["OVERLOAD"] = int(number_of_conflicts)
        output_dict["ARRIVAL-DELAY"] = int(total_delay)
        output_dict["SECTOR-NUMBER"] = int(number_sectors)
        output_dict["SECTOR-DIFF"] = int(sector_diff)
        output_dict["REROUTE"] = int(number_reroutes)
        output_dict["RECONFIG"] = int(number_sector_reconfigurations)
        output_dict["TOTAL-TIME-TO-THIS-POINT"] =  int(current_time)
        output_dict["COMPUTATION-FINISHED"] = True
        output_dict["DIFF"] = {}
        output_dict["DIFF"]["ACCEPTED_SOLUTION"] = False
        output_string = json.dumps(output_dict)
        print(output_string, flush=True)
        if self._controller_enabled is True:
            self._control_pub_socket.send_string(f"{output_string}")

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

        global_dto = convert_global_vars_to_dto(self)
        return "fin","fin", global_dto
