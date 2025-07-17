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

import pandas as pd

import numpy as np
import pickle

import multiprocessing as mp

from pathlib import Path
from typing import Any, Final, List, Optional

from solver import Solver, Model

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    
from concurrent.futures import ProcessPoolExecutor
from optimize_flights import OptimizeFlights
from concurrent.futures import ProcessPoolExecutor

import gurobipy as gp
from gurobipy import GRB

from mip_model import MIPModel


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
        capacity_path: Optional[Path],
        instance_path: Optional[Path],
        airport_vertices_path: Optional[Path],
        encoding_path: Optional[Path],
        seed: Optional[int],
        number_threads: Optional[int],
        timestep_granularity: Optional[int],
        max_explored_vertices: Optional[int],
        max_delay_per_iteration: Optional[int],
        max_time: Optional[int],
    ) -> None:
        self._graph_path: Optional[Path] = graph_path
        self._capacity_path: Optional[Path] = capacity_path
        self._instance_path: Optional[Path] = instance_path
        self._airport_vertices_path: Optional[Path] = airport_vertices_path
        self._encoding_path: Optional[Path] = encoding_path
        self._seed: Optional[int] = seed
        self._number_threads: Optional[int] = number_threads
        self._timestep_granularity: Optional[int] = timestep_granularity

        self._max_explored_vertices: Optional[int] = max_explored_vertices
        self._max_delay_per_iteration: Optional[int] = max_delay_per_iteration
        self._max_time: Optional[int] = max_time

        # Data containers — populated by :pymeth:`load_data`.
        self.graph: Optional[np.ndarray] = None
        self.capacity: Optional[np.ndarray] = None
        self.instance: Optional[np.ndarray] = None
        self.encoding: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def load_data(self) -> None:
        """Load all CSV files provided on the command line."""
        if self._graph_path is not None:
            self.graph = _load_csv(self._graph_path)
        if self._capacity_path is not None:
            self.capacity = _load_csv(self._capacity_path)
        if self._instance_path is not None:
            self.instance = _load_csv(self._instance_path)
        if self._airport_vertices_path is not None:
            self.airport_vertices = _load_csv(self._airport_vertices_path)
        if self._encoding_path is not None:
            print(self._encoding_path)
            with open(self._encoding_path, "r") as file:
                self.encoding = file.read()

    def run(self) -> None:  # noqa: D401 – imperatives okay here
        """Run the application"""
        self.load_data()

        # --- Demonstration output — remove/replace in production ---------
        print("  graph   :", None if self.graph is None else self.graph.shape)
        print("  capacity:", None if self.capacity is None else self.capacity.shape)
        print("  instance:", None if self.instance is None else self.instance.shape)
        print("  airport-vertices:", None if self.airport_vertices is None else self.airport_vertices.shape)
        # -----------------------------------------------------------------

        edges_instance = []

        for row_index in range(self.graph.shape[0]):
            row = self.graph[row_index,: ]
            edges_instance.append(f"sectorEdge({row[0]},{row[1]}).")

        edges_instance = "\n".join(edges_instance)

        edge_distances = self.compute_distance_matrix()

        airport_instance = []

        for row_index in range(self.airport_vertices.shape[0]):
            row = self.airport_vertices[row_index]
            airport_instance.append(f"airport({row}).")

        airport_instance = "\n".join(airport_instance)

        # |T| = 24 (1h-simulation), or 24*4 (15 minute simulation)
        # |F| = number of flights
        # |R| = number of regions (self.capacity.shape[0])

        # 1.) Create flights matrix (|F|x|T|) --> For easier matrix handling
        converted_instance_matrix = self.instance_to_matrix(self.instance, self._max_time)
        # 2.) Create demand matrix (|R|x|T|)
        system_loads = OptimizeFlights.bucket_histogram(converted_instance_matrix, self.capacity.shape[0], self._timestep_granularity)
        # 3.) Create capacity matrix (|R|x|T|)
        capacity_time_matrix = self.capacity_time_matrix(self.capacity,system_loads.shape[1])
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

        np.savetxt("00_initial_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")
        original_converted_instance_matrix = converted_instance_matrix.copy()

        original_max_time = original_converted_instance_matrix.shape[1]


        start_time = time.time()

        model = self.build_MIP_model(edge_distances, converted_instance_matrix, capacity_time_matrix,
                        capacity_demand_diff_matrix, additional_time_increase, fill_value, 
                        max_number_airplanes_considered_in_ASP, max_number_processors, original_max_time)

        end_time = time.time()
        print(f">> Elapsed solving time: {end_time - start_time}")


        np.savetxt("01_final_instance.csv", converted_instance_matrix,delimiter=",",fmt="%i")

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

        print("<<<<<<<<<<<<<<<<----------------->>>>>>>>>>>>>>>>")
        print("                  FINAL RESULTS")
        print("<<<<<<<<<<<<<<<<----------------->>>>>>>>>>>>>>>>")
        print(f"Total delay (all flights): {total_delay}")
        print(f"Average delay per flight:  {mean_delay:.2f}")
        print(f"Maximum single-flight delay: {max_delay}")


    def build_MIP_model(self,
                edge_distances: np.ndarray,
                converted_instance_matrix: np.ndarray,
                capacity_time_matrix: np.ndarray,
                capacity_demand_diff_matrix: np.ndarray,
                additional_time_increase: int,
                fill_value: int,
                max_rows: int,
                n_proc: int,
                original_max_time: int,
                ):
        """
        Split the candidate rows into *n_proc* equally‑sized chunks
        (≤ max_rows each) and build one ``OptimizeFlights`` instance per chunk.
        """

        max_delay = 24

        flight_sector_instances = []
        flight_times_instance = []
        timed_capacities = []

        considered_vertices = set()


        env = gp.Env(empty=True)          
        env.setParam('OutputFlag', 1)     
        env.start()

        model = gp.Model(env=env, name="toy")

        # |F|x|D|x|T|x|V|
        # F=flights, D=possible-delays, T=time, V=vertices
        flight_variables = {}
        flight_variables_pd = pd.DataFrame(columns=["F","D","T","V","obj"])

        #max_delay = max_delay + 1

        for flight_affected_index in range(converted_instance_matrix.shape[0]):

            if flight_affected_index not in flight_variables:
                flight_variables[flight_affected_index] = {}

            flight_affected = converted_instance_matrix[flight_affected_index,:]

            start_time = np.flatnonzero(flight_affected != fill_value)
            start_time = start_time[0] if start_time.size else 0
            
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            origin_to_destination_time = edge_distances[origin,destination]

            considered_vertices = considered_vertices.union(set([origin,destination]))

            prev_vertices = None

            for possible_start_time in range(start_time, start_time+max_delay+1):

                delay = possible_start_time - start_time

                #if delay not in flight_variables[flight_affected_index]:
                #    flight_variables[flight_affected_index][delay] = {}               
                #if possible_start_time not in flight_variables[flight_affected_index][delay]:
                #    flight_variables[flight_affected_index][delay][possible_start_time] = {}

                
                origin_variable = model.addVar(vtype=GRB.BINARY, name=f"x[{flight_affected_index},{delay},{possible_start_time},{origin}]")

                #flight_variables[flight_affected_index][delay][possible_start_time][origin] = origin_variable

                entry = pd.DataFrame.from_dict({
                    "F": [flight_affected_index],
                    "D": [delay],
                    "T": [possible_start_time],
                    "V": [origin],
                    "obj": [origin_variable]
                })

                flight_variables_pd = pd.concat([flight_variables_pd,entry], ignore_index = True)


            for from_origin_time in range(1,origin_to_destination_time + 1): # The +1 means that we add the destination airport
                
                time_to_destination_diff = origin_to_destination_time - from_origin_time

                origin_vertices = edge_distances[origin,:] == from_origin_time
                destination_vertices = edge_distances[destination,:] == time_to_destination_diff

                matching_vertices = origin_vertices & destination_vertices

                vertex_ids = np.where(matching_vertices)[0]
                vertex_ids = self.restrict_max_vertices(prev_vertices, vertex_ids, matching_vertices, flight_affected, from_origin_time, edge_distances)
                prev_vertices = list(vertex_ids)

                for vertex_id in vertex_ids:

                    for possible_time in range(start_time + from_origin_time, start_time+max_delay+1+from_origin_time):
                        
                        delay = possible_time - start_time - from_origin_time

                        #if delay not in flight_variables[flight_affected_index]:
                        #    flight_variables[flight_affected_index][delay] = {}
                        #if possible_time not in flight_variables[flight_affected_index][delay]:
                        #    flight_variables[flight_affected_index][delay][possible_time] = {}

                        origin_variable = model.addVar(vtype=GRB.BINARY, name=f"x[{flight_affected_index},{delay},{possible_time},{vertex_id}]")
                        #flight_variables[flight_affected_index][delay][possible_time][vertex_id] = origin_variable

                        entry = pd.DataFrame.from_dict({
                            "F": [flight_affected_index],
                            "D": [delay],
                            "T": [possible_time],
                            "V": [vertex_id],
                            "obj": [origin_variable]
                        })

                        flight_variables_pd = pd.concat([flight_variables_pd,entry], ignore_index = True)

        print("VARIABLE GENERATION COMPLETE")
        #print(flight_variables_pd)

        model.update()

        ############################
        # CONSTRAINTS (1,2,3,4):

        for sector in range(capacity_time_matrix.shape[0]):

            if np.any(self.airport_vertices == sector, where=True):
                is_airport=True
            else:
                is_airport=False

            for time in range(self._max_time):

                considered_rows = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd["T"]==time)]

                if considered_rows.shape[0] > 0:

                    positive_capacity_variables = []
                    negative_capacity_variables = []

                    for _, row in considered_rows.iterrows():

                        flight = row["F"]
                        delay = row["D"]

                        positive_capacity_variables.append(row["obj"])

                        if is_airport is True:
                            next_considered_rows = flight_variables_pd.loc[(flight_variables_pd['F']==flight)&(flight_variables_pd['V']==sector)&(flight_variables_pd["T"]==time-1)]

                            if considered_rows.shape[0] == 0:
                                continue

                            for _,airport_row in next_considered_rows.iterrows():
                                negative_capacity_variables.append(airport_row["obj"])

                        else:

                            next_considered_rows = flight_variables_pd.loc[(flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd["T"]==time+1)]

                            if considered_rows.shape[0] == 0:
                                continue

                            vertices = list(set(next_considered_rows["V"]))

                            for vertex in vertices:
                                if edge_distances[sector,vertex] == 1:
                                    next_considered_rows = flight_variables_pd.loc[(flight_variables_pd['F']==flight)&(flight_variables_pd['V']==vertex)&(flight_variables_pd["T"]==time)]

                                    for _,vertex_row in next_considered_rows.iterrows():
                                        negative_capacity_variables.append(vertex_row["obj"])

                    #print("+".join([var.VarName for var in positive_capacity_variables]) + " - (" + "+".join([var.VarName for var in negative_capacity_variables]) + ") <= " + str(capacity_time_matrix[sector,time]))

                    model.addConstr(gp.quicksum(positive_capacity_variables) - gp.quicksum(negative_capacity_variables) <= capacity_time_matrix[sector, time]) 


        model.update()
        # ADD FLIGHT CONSTRTAINTS (4,5,6,8):

        optimization_variables = []

        for flight in range(converted_instance_matrix.shape[0]):

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)]

            flight_affected = converted_instance_matrix[flight,:]
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            for time in list(set(considered_rows["T"])):
            
                considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time))]

                for sector in list(set(considered_rows_2["V"])):

                    
                    considered_rows_constraint_8 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time)&(flight_variables_pd["V"]==sector))]
                    considered_rows_constraint_8_prim = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["V"]==sector))]

                    if considered_rows_constraint_8.shape[0] > 0 and considered_rows_constraint_8_prim.shape[0] > 0:
                        # CONSTRAINT (8)

                        for _,variable_1_row in considered_rows_constraint_8.iterrows():
                            for _,variable_2_row in considered_rows_constraint_8_prim.iterrows():
                                model.addConstr(variable_1_row["obj"] - variable_2_row["obj"] <= 0)


                    if sector != origin:
                        # Do not add constraints at origin (Constraint (4))
                    
                        considered_rows_3 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time)&(flight_variables_pd["V"]==sector))]

                        if considered_rows_3.shape[0] > 0:
                            for index,row in considered_rows_3.iterrows():

                                left_variable = row["obj"]
                                delay = row["D"]

                                considered_rows_4 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time-1)&(flight_variables_pd["D"]==delay))]

                                if considered_rows_4.shape[0] > 0:
                                    # otherwise at origin

                                    vertices = list(set(considered_rows_4["V"]))
                                    
                                    right_variables = []
                                    for vertex in vertices:
                                        if edge_distances[sector,vertex] == 1:
                                            considered_rows_5 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time-1)&(flight_variables_pd["D"]==delay)&(flight_variables_pd["V"]==vertex))]

                                            for _,vertex_row in considered_rows_5.iterrows():
                                                right_variables.append(vertex_row["obj"])

                                    if len(right_variables) > 0:
                                        # CONSTRAINT (4)
                                        model.addConstr(left_variable - gp.quicksum(right_variables) <= 0)


            # CONSTRAINTS (5)&(6): 
            considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay))]

            for sector in list(set(considered_rows_2["V"])):

                print(sector)

                if sector != destination:
                    
                    considered_rows_3 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["V"]==sector))]

                    for _,row in considered_rows_3.iterrows():

                        left_variable = row["obj"]
                        right_variables = []
                        
                        considered_rows_4 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["T"]==row["T"]+1))]

                        vertices = list(set(considered_rows_4["V"]))

                        for vertex in vertices:

                            if edge_distances[sector,vertex] == 1:
                                considered_rows_5 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["T"]==row["T"]+1)&(flight_variables_pd["V"]==vertex))]

                                for _,vertex_row in considered_rows_5.iterrows():

                                    right_variables.append(vertex_row["obj"])

                        # CONSTRAINT (5):
                        print(f"{left_variable.VarName} - ({'+'.join([var.VarName for var in right_variables])}) <= 0")
                        model.addConstr(left_variable - gp.quicksum(right_variables) <= 0)

                        # CONSTRAINT (6):
                        model.addConstr(gp.quicksum(right_variables) <= 1) 

                
            considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==origin))]
            
            variables = []
            for _,considered_row_2 in considered_rows_2.iterrows():
                variables.append(considered_row_2["obj"])
            model.addConstr(gp.quicksum(variables) >= 1)


            # OPTIMIZATION VARIABLES FILL:

            considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination))]

            arrival_times = list(set(considered_rows_2["T"]))
            latest_arrival_time = max(arrival_times)

            considered_rows_2_prim = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination)&(flight_variables_pd["T"]==latest_arrival_time))]
            for _,considered_rows_2_prim_row in considered_rows_2_prim.iterrows():
                latest_arrival_time_variable = considered_rows_2_prim_row["obj"]

            for arrival_time in arrival_times:
                considered_rows_2_prim = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination)&(flight_variables_pd["T"]==arrival_time))]
            
                for _,considered_rows_2_prim_row in considered_rows_2_prim.iterrows():
                    arrival_time_variable = considered_rows_2_prim_row["obj"]

                optimization_variables.append(latest_arrival_time_variable - arrival_time_variable)
                                    
        # OPTIMIZATION
        model.setObjective(gp.quicksum(optimization_variables), GRB.MINIMIZE)
        model.optimize()

        print(f"objective value={model.ObjVal}")

        result_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        for flight in range(converted_instance_matrix.shape[0]):

            flight_affected = converted_instance_matrix[flight,:]
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination)]

            actual_delay = max_delay

            for _,row in considered_rows.iterrows():

                if row["obj"].X >= 1 and row["D"] < actual_delay:
                    actual_delay = row["D"]

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==actual_delay)]

            for _,row in considered_rows.iterrows():

                if row["obj"].X >= 1:
                    result_matrix[flight,row["T"]] = row["V"]

            print(f"flight:{flight};delay:{actual_delay}")


        print(result_matrix)






                
            



        # CONTINUE WITH MODEL BUILDING
        # 1.) Inequalities for capacity overload
        # 2.) Inequalities for other things

        quit()

        return model
    
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

    def instance_to_matrix(self, inst: np.ndarray,
                           max_time: int,
                        *,
                        fill_value: int = -1,
                        compress: bool = False) -> np.ndarray:
        """
        Convert an *instance* array with shape (n, 3) into a 2-D matrix.

        Parameters
        ----------
        inst
            Each row is (row_id, value, col_id).
        fill_value
            Placeholder put where no entry is defined (defaults to -1).
        compress
            • False  ➜ allocate rows = max(row_id)+1, cols = max(col_id)+1  
            (fastest, but can be large if indices are sparse).

            • True   ➜ map the *unique* row/col labels to consecutive
            indices before building the matrix.  Returns the compact
            matrix plus the label mappings (see below).

        Returns
        -------
        If `compress is False`:
            2-D `np.ndarray` with dtype taken from `inst[:,1]`.
        If `compress is True`:
            (matrix, row_labels, col_labels)
        """
        rows = inst[:, 0].astype(int)
        vals = inst[:, 1]
        cols = inst[:, 2].astype(int)

        out = np.full((rows.max() + 1, cols.max() + 1),
                        fill_value,
                        dtype=vals.dtype)
        out[rows, cols] = vals

        if out.shape[1] < max_time:
            diff = max_time - out.shape[1]
            extra_col = -1 * np.ones((out.shape[0], diff), dtype=int)
            out = np.hstack((out, extra_col)) 

        return out

   
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

    def capacity_time_matrix(self, cap: np.ndarray,
                            n_times: int,
                            *,
                            sort_by_bucket: bool = False) -> np.ndarray:
        """
        Expand the per-bucket capacity vector to shape (|B|, n_times).

        Parameters
        ----------
        cap
            2-column array ``[bucket_id, capacity_value]`` with shape (|B|, 2).
        n_times
            Number of time steps (usually ``counts.shape[1]``).
        sort_by_bucket
            If *True* (default) the rows are first sorted by *bucket_id*
            so that row *i* corresponds to bucket *i* (0 … |B|–1).

        Returns
        -------
        cap_mat : np.ndarray
            Shape (|B|, n_times) where every row is the bucket’s capacity.
        """
        if sort_by_bucket:
            cap = cap[np.argsort(cap[:, 0])]

        # Extract the capacity column → shape (|B|,)
        cap_vals = cap[:, 1]

        # Broadcast to (|B|, n_times) without an explicit loop
        cap_mat = np.broadcast_to(cap_vals[:, None], (cap_vals.size, n_times))
        # cap_mat = cap_mat.copy()

        return cap_mat
    
 
    def restrict_max_vertices(self, prev_vertices, vertex_ids, matching_vertices, flight_affected, from_origin_time, edge_distances):

        max_vertices_cutoff_value = self._max_explored_vertices
        # ------------------------------------------------
        # START PATH STUFF:

        if len(vertex_ids) > max_vertices_cutoff_value:
            # GO INTO PATH MODE:
            if prev_vertices is None:
                # FALLBACK
                rng = np.random.default_rng(seed = self._seed)
                vertex_ids = rng.choice(vertex_ids, size=max_vertices_cutoff_value, replace=False)
            else:
                # prev_vertices is not None
                # Create path

                used_vertices = []

                if len(prev_vertices) >= max_vertices_cutoff_value:
                    del prev_vertices[-1]

                for prev_vertex in prev_vertices:

                    path_vertices_mask = edge_distances[prev_vertex,:] == 1 & matching_vertices
                    path_vertices_ids = np.where(path_vertices_mask)[0]

                    rng = np.random.default_rng(seed = self._seed)
                    used_vertex_id = rng.choice(path_vertices_ids, size=1, replace=False)
                    used_vertices.append(int(used_vertex_id))

                flightPathVertex = int(flight_affected[from_origin_time])
                if flightPathVertex not in used_vertices:
                    if len(used_vertices) >= max_vertices_cutoff_value:
                        del used_vertices[-1]
                    used_vertices.append(int(flightPathVertex))

                vertex_ids = used_vertices

        else:
            pass
        # -------------------------------------------------------

        return vertex_ids
        


 
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
        "--path-graph",
        type=Path,
        default=Path("instances","edges.csv"),  
        metavar="FILE",
        help="Location of the graph CSV file.",
    )
    parser.add_argument(
        "--path-capacity",
        type=Path,
        default=Path("instances","capacity.csv"), 
        metavar="FILE",
        help="Location of the capacity CSV file.",
    )
    parser.add_argument(
        "--path-instance",
        type=Path,
        default=Path("instances","instance_100.csv"), 
        metavar="FILE",
        help="Location of the instance CSV file.",
    )
    parser.add_argument(
        "--airport-vertices-path",
        type=Path,
        default=Path("instances","airport_vertices.csv"), 
        metavar="FILE",
        help="Location of the airport-vertices CSV file.",
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

    app = Main(args.path_graph, args.path_capacity, args.path_instance,
               args.airport_vertices_path, args.encoding_path,
               args.seed,args.number_threads, args.timestep_granularity,
               args.max_explored_vertices, args.max_delay_per_iteration,
               args.max_time)
    app.run()


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


 
