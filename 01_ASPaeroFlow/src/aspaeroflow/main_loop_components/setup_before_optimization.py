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
from ..auxiliaries.computation_helpers import compute_total_number_sectors, minimize_number_of_sectors_new
from ..optimize_flights import OptimizeFlights

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

    def conv(x):
        return int(math.ceil(float(x)))

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=1, converters=conv)
    except ValueError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc



class SetupBeforeOptimization:

    def __init__(self, global_var_dto):
        convert_dto_to_global_vars(self, global_var_dto)



    def setup_before_optimization(self):

        counter_equal_solutions = 0
        additional_time_increase = 0
        max_number_airplanes_considered_in_ASP = self.max_considered_aircraft

        paused = None
        original_max_time = None
        planned_arrival_times = None
        flight_durations = None
        original_max_explored_vertices = None
        default_number_capacity_management_configs = None
        old_converted_instance = None
        old_converted_navpoints = None
        old_converted_navpoint_matrix = None
        old_navaid_sector_time_assignment = None

        iteration = 0
        fill_value = -1

        max_number_processors = self._number_threads
        seed = self._seed

        original_start_time = time.time()

        if self._max_delay_per_iteration < 0:
            #self._max_delay_per_iteration = original_max_time
            self._max_delay_per_iteration = 20

        if self._explainability_context is None:
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

            #np.savetxt("202603_converted_navpoint_matrix.csv", converted_navpoint_matrix, delimiter=",",fmt="%i")
            #np.savetxt("202603_capacity_demand_diff_matrix.csv", capacity_demand_diff_matrix, delimiter=",",fmt="%i")
            #quit()

            #number_of_conflicts = capacity_overload_mask.sum()
            number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()
            number_of_conflicts_prev = None


            if self.verbosity > 1:
                np.savetxt("20250826_initial_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

            original_converted_instance_matrix = converted_instance_matrix.copy()
            original_navaid_sector_time_assignment = navaid_sector_time_assignment.copy()

            original_max_explored_vertices = self._max_explored_vertices
            original_max_time = original_converted_instance_matrix.shape[1]
            self.original_max_time = original_max_time


            old_converted_instance = converted_instance_matrix.copy()
            old_converted_navpoint_matrix = converted_navpoint_matrix.copy()
            old_navaid_sector_time_assignment = navaid_sector_time_assignment.copy()

            if np.any(capacity_overload_mask, where=True):
                
                default_number_capacity_management_configs = self.number_capacity_management_configs

                _, _, flight_durations = self.flight_spans_contiguous(converted_instance_matrix, fill_value=-1)

            # Track to Weights & Biases when enabled
            current_time = time.time() - original_start_time

            number_sectors = compute_total_number_sectors(navaid_sector_time_assignment)

            if self._wandb_log is not None:

                self._wandb_log({
                    "iteration": int(iteration), # FIRST ONE:
                    "number_of_conflicts": int(number_of_conflicts),
                    "total_delay": int(0),
                    "number_sectors": int(number_sectors),
                    "sector_diff": int(0),
                    "number_reroutes": int(0),
                    "number_reconfigurations": int(0),
                    "current_time": int(current_time),
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
            print(output_string, flush=True)
            if self._controller_enabled is True:
                output_dict["DIFF"] = {}
                output_dict["DIFF"]["FLIGHTS"] = []
                output_dict["DIFF"]["SECTORS"] = []
                output_dict["DIFF"]["ACCEPTED_SOLUTION"] = True
                output_dict["ITERATION"] = 0
                output_string = json.dumps(output_dict)

                self._control_pub_socket.send_string(f"RESET-OBJECTIVE-VALUE")
                self._control_pub_socket.send_string(f"{output_string}")

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
            elif self.max_number_sectors == -2:
                self.max_number_sectors = navaid_sector_time_assignment.shape[0]

            if self.minimize_number_sectors is True:

                if self.verbosity > 1:
                    print("MINIMIZE NUMBER OF SECTORS INITIALIZED")

                t_start = 1
                t_end = 60

                for t_start in range(1,converted_instance_matrix.shape[1] + 1, self._timestep_granularity):
                    t_end = t_start + self._timestep_granularity - 1

                    if t_end >= converted_instance_matrix.shape[1]:
                        t_end = converted_instance_matrix.shape[1] - 1

                    converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads = minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, self.max_number_navpoints_per_sector, self.max_number_sectors, t_start, t_end, self.networkx_navpoint_graph, self.airports)

            global_t_start = 1
            time_bucket_updated = 0

            paused = True

            if self._controller_enabled is True:
                # Send navpoints, etc.
                folder_path = self._data_dir
                # Navgraph:
                folder_path_navgraph_vertex = Path(folder_path / "navgraph" / "vertices.csv")
                folder_path_navgraph_edges = Path(folder_path / "navgraph" / "edges.csv")
                folder_path_mappings = Path(folder_path / "mappings" / "vertex_map.csv")

                vertex_map = np.genfromtxt(
                    folder_path_mappings, 
                    delimiter=',', 
                    names=True, 
                    dtype=None, 
                    encoding='utf-8'
                )

                # Load vertex coordinates/attributes (Assuming numeric float/int)
                # usecols can be used to filter specific ATM parameters (e.g., Lat, Lon, Alt)
                vertices = np.genfromtxt(
                    folder_path_navgraph_vertex, 
                    delimiter=',', 
                    names=True, 
                    dtype=None, 
                    encoding='utf-8'
                )

                # Load edge list (Source, Target, Weight)
                edges = np.genfromtxt(
                    folder_path_navgraph_edges, 
                    delimiter=',', 
                    names=True, 
                    dtype=None, 
                    encoding='utf-8'
                )

                graph_dict = {}
                graph_dict["vertices"] = {}
                graph_dict["edges"] = {}
                graph_dict["sectors"] = {}

                for vertex in self.networkx_navpoint_graph.nodes():

                    vertex = int(vertex)

                    graph_dict["vertices"][vertex] = {}
                    row = vertex_map[vertex_map["VERTEX_ID"] == vertex]
                    name = row[0][0]

                    graph_dict["vertices"][vertex]["id"] = int(vertex)
                    graph_dict["vertices"][vertex]["name"] = str(name)
                    full_vertex = vertices[vertices["IDENTIFIER"] == name][0]
                    graph_dict["vertices"][vertex]["lat"] = float(full_vertex[1])
                    graph_dict["vertices"][vertex]["lon"] = float(full_vertex[2])
                    graph_dict["vertices"][vertex]["alt"] = float(full_vertex[3])
                    graph_dict["vertices"][vertex]["airport"] = int(full_vertex[4])


                    if vertex not in graph_dict["sectors"]:
                        graph_dict["sectors"][vertex] = {}
                        graph_dict["sectors"][vertex]["vertices"] = []

                    sector = int(navaid_sector_time_assignment[vertex,0])

                    if sector not in graph_dict["sectors"]:
                        graph_dict["sectors"][sector] = {}
                        graph_dict["sectors"][sector]["vertices"] = []

                    graph_dict["sectors"][sector]["vertices"].append(vertex)

                for edge in self.networkx_navpoint_graph.edges():
                    v0 = int(edge[0])
                    v1 = int(edge[1])

                    if v0 not in graph_dict["edges"]:
                        graph_dict["edges"][v0] = {}

                    if v1 not in graph_dict["edges"]:
                        graph_dict["edges"][v1] = {}

                    graph_dict["edges"][v0][v1] = int(0)
                    graph_dict["edges"][v1][v0] = int(0)

                for sector in graph_dict["sectors"].keys():

                    overload = 0
                    for _time in range(capacity_demand_diff_matrix.shape[1]):

                        if capacity_demand_diff_matrix[sector,_time] < 0:
                            overload += (-capacity_demand_diff_matrix[sector,_time])
                    
                    graph_dict["sectors"][int(sector)]["overload"] = int(overload)

                for index in range(1, self.flights.shape[0]):

                    tuple_0 = self.flights[index-1,:]
                    tuple_1 = self.flights[index,:]

                    id0 = int(tuple_0[0])
                    v0 = int(tuple_0[1])
                    id1 = int(tuple_1[0])
                    v1 = int(tuple_1[1])

                    if id0 == id1:
                        graph_dict["edges"][v0][v1] += int(1)
                        graph_dict["edges"][v1][v0] += int(1)

                json_graph_dict = json.dumps(graph_dict)
                self._control_ctrl_socket.send_string(json_graph_dict)
                print(json_graph_dict)
                print("[OPTIMIZER->CONTROL]: SENT GRAPH DICT")

        else: # explainability context not none:

            if self._encoding_path is not None:
                with open(self._encoding_path, "r") as file:
                    self.encoding = file.read()

            global_t_start = 1
            self.sectors = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SECTORS"])
            navaid_sector_time_assignment = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["NAVAID-SECTOR-TIME-ASSIGNMENT"])
            converted_instance_matrix = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["CONVERTED-INSTANCE-MATRIX"])
            converted_navpoint_matrix = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["CONVERTED-NAVPOINT-MATRIX"])
            system_loads = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["SYSTEM-LOADS"])
            capacity_time_matrix = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["CAPACITY-TIME-MATRIX"])

            original_converted_instance_matrix = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["ORIGINAL-CONVERTED-INSTANCE-MATRIX"])
            original_navaid_sector_time_assignment = self.decode_ndarray(self._explainability_context["ITERATION-BACKUP"]["ORIGINAL-NAVAID-SECTOR-TIME-ASSIGNMENT"])

            capacity_demand_diff_matrix = capacity_time_matrix - system_loads
            # 5.) Create capacity overload matrix:
            capacity_overload_mask = capacity_demand_diff_matrix < 0

            number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()

            asp_encoded = self._explainability_context["ITERATION-BACKUP"]["ASP-INSTANCE"]
            self._explainability_context["ITERATION-BACKUP"]["ASP-INSTANCE"] = base64.b64decode(asp_encoded).decode("utf-8")

            #open(f"20260501_test_instance.lp","w").write(self._explainability_context["ITERATION-BACKUP"]["ASP-INSTANCE"])

            if self._explainability_context["TYPE"] == "FLIGHT":
                explanation_fact = f"chosen_path({self._explainability_context["ID"]},0)."
            elif self._explainability_context["TYPE"] == "SECTOR":
                explanation_fact = f"chosen_config(0)."
            else:
                raise Exception(f"NOT IMPLEMENTED: {self._explainability_context["TYPE"]}")
            
            self._explainability_context["ITERATION-BACKUP"]["ASP-INSTANCE"] += f"\n {explanation_fact}"

        optimization_dto = {}

        optimization_dto["global_t_start"] = global_t_start 
        optimization_dto["time_bucket_updated"] = time_bucket_updated 
        optimization_dto["navaid_sector_time_assignment"] = navaid_sector_time_assignment
        optimization_dto["converted_instance_matrix"] = converted_instance_matrix
        optimization_dto["converted_navpoint_matrix"] = converted_navpoint_matrix
        optimization_dto["system_loads"] = system_loads
        optimization_dto["capacity_time_matrix"] = capacity_time_matrix
        optimization_dto["original_converted_instance_matrix"] = original_converted_instance_matrix
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
        optimization_dto["paused"] = paused
        optimization_dto["original_max_time"] = original_max_time
        optimization_dto["planned_arrival_times"] = planned_arrival_times
        optimization_dto["flight_durations"] = flight_durations
        optimization_dto["original_max_explored_vertices"] = original_max_explored_vertices 
        optimization_dto["default_number_capacity_management_configs"] = default_number_capacity_management_configs 

        optimization_dto["old_converted_instance"] = old_converted_instance
        optimization_dto["old_converted_navpoint_matrix"] = old_converted_navpoint_matrix
        optimization_dto["old_navaid_sector_time_assignment"] = old_navaid_sector_time_assignment

        optimization_dto["solutions"] = None
        optimization_dto["iteration_backup"] = None
        optimization_dto["controller_sector_diff_dict"] = None
        optimization_dto["sector_index"] = None

        global_dto = convert_global_vars_to_dto(self)

        return optimization_dto, global_dto

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


