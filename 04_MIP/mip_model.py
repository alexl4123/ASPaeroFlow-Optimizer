
import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np
import networkx as nx
import math
import time
import json

LINEAR = "linear"
TRIANGULAR = "triangular"
MAX = "max"

from pathlib import Path
def _save_csv(path: Path, arr):
    a = np.asarray(arr)
    fmt = "%d" if a.dtype.kind in ("i","u","b") else "%g"
    np.savetxt(path, a, fmt=fmt, delimiter=",")



class MIPModel:

    def __init__(self, sectors, airport_vertices, max_time, max_explored_vertices, seed, timestep_granularity, verbosity, number_threads, navaid_sector_lookup, composite_sector_function, sector_capacity_factor, original_converted_instance_matrix, navaid_sector_time_assignment, old_navaid_sector_time_assignment, start_time):
        
        self.sectors = sectors
        self.airport_vertices = airport_vertices
        self._max_time = max_time
        self._max_explored_vertices = max_explored_vertices
        self._seed = seed
        self._timestep_granularity = timestep_granularity
        self._composite_sector_function = composite_sector_function
        self._sector_capacity_factor = sector_capacity_factor

        self.original_converted_instance_matrix = original_converted_instance_matrix
        self.navaid_sector_time_assignment = navaid_sector_time_assignment
        self.old_navaid_sector_time_assignment = old_navaid_sector_time_assignment
        self.start_time = start_time

        self._max_number_threads = number_threads
        self.navaid_sector_lookup = navaid_sector_lookup

        self.env = gp.Env(empty=True)          
        self.env.setParam('OutputFlag', 1)     
        self.env.setParam(GRB.Param.Seed, seed)

        if verbosity == 0:
            self.env.setParam("OutputFlag", 0)

        self.report_every_s = float(1)
        self.progress = [] 

        self.env.start()

    def construct_objective_function(self, flight_variables_pd, converted_instance_matrix, planned_arrival_times, fill_value = -1):

        optimization_variables = []

        for flight in range(converted_instance_matrix.shape[0]):
            considered_flight = converted_instance_matrix[flight,:]
            considered_flight = considered_flight[considered_flight != fill_value]
            if len(considered_flight) == 0:
                continue

            destination = considered_flight[-1]

            planned_arrival_time = planned_arrival_times[flight]

            destination_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==destination))]
            for delay in list(destination_rows['D']):
            
                final_destination_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==destination)&(flight_variables_pd['D']==delay))]

                actual_arrival_time = max(list(final_destination_rows['T']))

                actual_arrival_rows = flight_variables_pd[((flight_variables_pd['F']==flight)&(flight_variables_pd['V']==destination)&(flight_variables_pd['D']==delay)&(flight_variables_pd['T']==actual_arrival_time))]

                for _, actual_arrival_row in actual_arrival_rows.iterrows():

                    arrival_variable = actual_arrival_row['obj']

                    real_delay = actual_arrival_time - planned_arrival_time

                    optimization_variables.append(real_delay * arrival_variable)

        return optimization_variables

    def create_model(self, converted_instance_matrix, capacity_time_matrix, unit_graphs, airplanes, max_delay, planned_arrival_times, airplane_flight, filed_flights, navaid_sector_time_assignment, converted_navpoint_matrix):
        
        solution = None
        self._max_time += max_delay

        """
        if converted_instance_matrix.shape[1] < self._max_time:
            diff = self._max_time - converted_instance_matrix.shape[1]
            extra_col = -1 * np.ones((converted_instance_matrix.shape[0], diff), dtype=int)
            converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 
        """

        delay_time_window = 2

        while solution is None:

            model = gp.Model(env=self.env, name="toy")
            model.Params.Threads = self._max_number_threads

            flight_variables_pd, sector_variables_pd, next_sectors, previous_sectors, max_effective_delay, all_flights_navpoints = self.add_variables(model, converted_instance_matrix, capacity_time_matrix, unit_graphs, airplanes, delay_time_window, filed_flights, airplane_flight)

            model.update()

            self.add_capacity_constraint(model, flight_variables_pd, sector_variables_pd, capacity_time_matrix, unit_graphs, converted_instance_matrix)


            model.update()

            self.add_flight_constraints_get_optimization_variables(model, flight_variables_pd, converted_instance_matrix, unit_graphs, max_delay, next_sectors, previous_sectors)

            model.update()

            self.add_consecutive_flight_constraints(model, flight_variables_pd, airplanes, airplane_flight, converted_instance_matrix)

            model.update()

            #self.add_valid_inequalities(model, flight_variables_pd, converted_instance_matrix, edge_distances, max_delay)

            optimization_variables = self.construct_objective_function(flight_variables_pd, converted_instance_matrix, planned_arrival_times)

            self.optimize(model, optimization_variables, flight_variables_pd, sector_variables_pd, converted_instance_matrix, max_delay, all_flights_navpoints)

            converted_instance_matrix_tmp, converted_navpoint_matrix_tmp = self.reconstruct_solution(model, flight_variables_pd, sector_variables_pd,
                                                                                             converted_instance_matrix, max_delay, all_flights_navpoints)


            mask = converted_instance_matrix_tmp != -1                                        # same shape as arr

            new_round = False
            if np.any(mask, where=True):
                # -----------------------------------------------------------------------------
                t_init  = self.last_valid_pos(self.original_converted_instance_matrix)      # last non--1 in the *initial* schedule
                t_final = self.last_valid_pos(converted_instance_matrix_tmp)     # last non--1 in the *final* schedule
                
                diff_tmp = converted_instance_matrix_tmp.shape[1] - self.old_navaid_sector_time_assignment.shape[1]
                if diff_tmp > 0:
                    self.old_navaid_sector_time_assignment = np.hstack([self.old_navaid_sector_time_assignment, np.repeat(self.old_navaid_sector_time_assignment[:, [-1]], diff_tmp, axis=1)])

                diff_tmp = converted_instance_matrix_tmp.shape[1] - self.navaid_sector_time_assignment.shape[1]
                if diff_tmp > 0:
                    self.navaid_sector_time_assignment = np.hstack([self.navaid_sector_time_assignment, np.repeat(self.navaid_sector_time_assignment[:, [-1]], diff_tmp, axis=1)])

                delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)
                system_loads = MIPModel.bucket_histogram(converted_instance_matrix_tmp, self.sectors, self.sectors.shape[0], converted_instance_matrix_tmp.shape[1], self._timestep_granularity)
                capacity_time_matrix = MIPModel.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, self.navaid_sector_time_assignment, z = self._sector_capacity_factor, composite_sector_function=self._composite_sector_function)
                capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                capacity_overload_mask = capacity_demand_diff_matrix < 0
                number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()
                total_delay  = delay.sum()
                number_sectors = self.compute_total_number_sectors(self.navaid_sector_time_assignment)
                sector_diff = np.count_nonzero(self.navaid_sector_time_assignment[:, 1:] != self.navaid_sector_time_assignment[:, :-1])
                original_max_time_converted = self.original_converted_instance_matrix.shape[1]  # original_max_time
                rerouted_mask = np.any(converted_instance_matrix_tmp[:, :original_max_time_converted] != self.original_converted_instance_matrix, axis=1)     # True if flight differs anywhere
                number_reroutes = int(np.count_nonzero(rerouted_mask))
                number_sector_reconfigurations = np.count_nonzero(self.navaid_sector_time_assignment != self.old_navaid_sector_time_assignment)

                current_time = time.time() - self.start_time
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

                if number_of_conflicts > 0:
                    new_round = True
            else:
                new_round = True

            if new_round is True:
                solution = None
                #max_delay = max_delay + 1
                self._max_time = converted_instance_matrix_tmp.shape[1]
                delay_time_window += self._timestep_granularity

                #navaid_sector_time_assignment = self.navaid_sector_time_assignment

                #navaid_sector_time_assignment = self.navaid_sector_lookup
                #converted_instance_matrix = converted_instance_matrix_tmp
                #converted_navpoint_matrix = converted_navpoint_matrix_tmp
                #system_loads = system_loads
                #capacity_time_matrix = capacity_time_matrix

                in_units = 1
                #number_new_cols = in_units * self._timestep_granularity
                number_new_cols = self._max_time - converted_instance_matrix.shape[1] + in_units * self._timestep_granularity

                # 0.) Handle Sector Assignments:
                new_cols = np.repeat(navaid_sector_time_assignment[:,[-1]], number_new_cols, axis=1)  # shape (N,k)
                navaid_sector_time_assignment = np.concatenate([navaid_sector_time_assignment, new_cols], axis=1)
                #navaid_sector_time_assignment = np.hstack([navaid_sector_time_assignment, np.repeat(navaid_sector_time_assignment[:, [-1]], number_new_cols, axis=1)])
                # 1.) Handle Instance Matrix:
                extra_col = -1 * np.ones((converted_instance_matrix.shape[0], number_new_cols), dtype=int)
                converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                extra_col = -1 * np.ones((converted_navpoint_matrix.shape[0], number_new_cols), dtype=int)
                converted_navpoint_matrix = np.hstack((converted_navpoint_matrix, extra_col)) 

                # 2.) Create demand matrix (|R|x|T|):
                system_loads = MIPModel.bucket_histogram(converted_instance_matrix, self.sectors, self.sectors.shape[0], converted_instance_matrix.shape[1], self._timestep_granularity)
                # 3.) Create capacity matrix (|R|x|T|):
                capacity_time_matrix = MIPModel.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, navaid_sector_time_assignment, z = self._sector_capacity_factor, composite_sector_function=self._composite_sector_function)

                # 4.) Subtract demand from capacity (|R|x|T|):
                capacity_demand_diff_matrix = capacity_time_matrix - system_loads
            else:
                solution = model

        converted_instance_matrix = converted_instance_matrix_tmp
        converted_navpoint_matrix = converted_navpoint_matrix_tmp

        return converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix
    

    def add_consecutive_flight_constraints(self, model, flight_variables_pd, airplanes, airplane_flight, converted_instance_matrix):
        
        airplanes = list(set(airplane_flight[:,0]))

        for airplane in airplanes:

            flights = airplane_flight[airplane_flight[:,0] == airplane, 1]
            
            flight_order = []
            for flight in flights:
                starting_time = min(list(np.nonzero(converted_instance_matrix[flight,:] != -1)[0]))
                flight_order.append((starting_time,flight))

            flight_order.sort()

            for index in range(1,len(flight_order)):
                flight_index_0 = flight_order[index-1][1]
                flight_index_1 = flight_order[index][1]

                arrival_time_flight_0 = max(list(np.nonzero(converted_instance_matrix[flight_index_0,:] != -1)[0]))
                departure_time_flight_1 = min(list(np.nonzero(converted_instance_matrix[flight_index_1,:] != -1)[0]))


                considered_flight_rows_0 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight_index_0))]
                considered_flight_rows_1 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight_index_1))]


                for delay_flight_0 in list(set(considered_flight_rows_0['D'])):

                    considered_flight_rows_0_tmp = flight_variables_pd.loc[((flight_variables_pd["F"]==flight_index_0)&(flight_variables_pd["D"]==delay_flight_0))]

                    actual_arrival_time = max(considered_flight_rows_0_tmp["T"])

                    for delay_flight_1 in list(set(considered_flight_rows_1['D'])):

                        considered_flight_rows_1_tmp = flight_variables_pd.loc[((flight_variables_pd["F"]==flight_index_1)&(flight_variables_pd["D"]==delay_flight_1))]
                        actual_departure_time = min(considered_flight_rows_1_tmp["T"])

                        if actual_arrival_time >= actual_departure_time:

                            considered_0 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight_index_0)&(flight_variables_pd["D"]==delay_flight_0)&(flight_variables_pd["T"]==actual_arrival_time))]
                            considered_1 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight_index_1)&(flight_variables_pd["D"]==delay_flight_1)&(flight_variables_pd["T"]==actual_departure_time))]

                            for _, considered_0_row in considered_0.iterrows():
                                for _, considered_1_row in considered_1.iterrows():
                                    model.addConstr((1 - considered_0_row["obj"]) >= considered_1_row["obj"])
    
    def k_diverse_near_shortest_paths(
        self,
        G, s, t, nearest_neighbors_lookup, k=5, eps=0.10, jaccard_max=0.6,
        penalty_scale=0.5, max_tries=200, weight_key="weight",
        filed_path = []
    ):
        
        s_t_length, _ = nx.bidirectional_dijkstra(G, s, t, weight=weight_key)

        allowed = (1.0 + eps) * s_t_length

        # 1) shortest length & prune to a small corridor: ds[u]+dt[u] ≤ (1+eps)*L0
        #
        if s not in nearest_neighbors_lookup:
            ds = nx.single_source_dijkstra_path_length(G, s, weight=weight_key)
            nearest_neighbors_lookup[s] = ds
        else:
            ds = nearest_neighbors_lookup[s]

        if t not in nearest_neighbors_lookup:
            dt = nx.single_source_dijkstra_path_length(G, t, weight=weight_key)
            nearest_neighbors_lookup[t] = dt
        else:
            dt = nearest_neighbors_lookup[t]

        keep = {u for u in G if u in ds and u in dt and ds[u] + dt[u] <= allowed}

        H = G.subgraph(keep).copy()

        # Edge-penalties (undirected key)
        def ekey(u, v):
            return (u, v) if u <= v else (v, u)
        penalties = {}

        # Penalized weight function
        def w(u, v, d):
            base = d.get(weight_key, 1.0)
            pen = penalties.get(ekey(u, v), 0.0)
            return base + pen

        paths, edge_sets, tries = [], [], 0

        if len(filed_path) > 0:
            E = {ekey(u, v) for u, v in zip(filed_path, filed_path[1:])}
            paths.append(filed_path)
            if len(E) > 0:
                edge_sets.append(E)

        while len(paths) < k and tries < max_tries:
            tries += 1
            try:
                p = nx.shortest_path(H, s, t, weight=w)
            except nx.NetworkXNoPath:
                break

            # Evaluate real (unpenalized) length
            L = nx.path_weight(G, p, weight=weight_key)
            if L > allowed:
                break  # can’t find more within slack

            # Edge-set and diversity check (Jaccard on edges)
            E = {ekey(u, v) for u, v in zip(p, p[1:])}
            similar = any(len(E & Es) / len(E | Es) > jaccard_max for Es in edge_sets)

            # Always penalize current path to push the next one away
            avg_edge = L / max(1, len(E))
            for e in E:
                penalties[e] = penalties.get(e, 0.0) + penalty_scale * avg_edge

            if similar:
                continue  # reject, keep searching

            paths.append(p)
            if len(E) > 0:
                edge_sets.append(E)

        return paths    
    
    def get_flight_navpoint_trajectory(self, flights_affected, networkx_graph, flight_index, start_time, airplane_speed_kts, path, timestep_granularity):

        traj = []
        current_time = start_time
        for hop, vertex in enumerate(path):
            if hop == 0:
                # Origin
                t_slot = current_time

                if t_slot >= flights_affected.shape[1]:
                    raise Exception("In optimize_flights max time exceeded current allowed time.")

            else:
                # En-route/destination
                prev_vertex = path[hop -1]
                duration_in_unit_standards = networkx_graph[prev_vertex][vertex]["weight"]

                current_time = current_time + duration_in_unit_standards

                t_slot=current_time

                if t_slot >= flights_affected.shape[1]:
                    raise Exception("In optimize_flights max time exceeded current allowed time.")

            traj.append((flight_index, vertex, t_slot))

        return traj
 

    def add_variables(self, model, converted_instance_matrix, capacity_time_matrix,  unit_graphs, airplanes, time_window, filed_flights, airplane_flight, fill_value = -1):

        # |F|x|D|x|T|x|V|
        # F=flights, D=possible-delays, T=time, V=vertices
        flight_variables_pd = pd.DataFrame(columns=["F","D","T","V","obj"])
        sector_variables_pd = pd.DataFrame(columns=["T","S","obj"])
        considered_vertices = set()

        max_effective_delay = 0

        next_sectors = {}
        previous_sectors = {}

        #max_delay = max_delay + 1
        self.nearest_neighbors_lookup = {}
        k = self._max_explored_vertices

        all_flight_navpoints_dict = {}

        for flight_affected_index in range(converted_instance_matrix.shape[0]):

            all_flight_navpoints_dict[flight_affected_index] = {}

            next_sectors[flight_affected_index] = {}
            previous_sectors[flight_affected_index] = {}


            flight_affected = converted_instance_matrix[flight_affected_index,:]
            flights_affected = converted_instance_matrix[np.array([flight_affected_index]),:]

            airplane_index = airplane_flight[airplane_flight[:,1] == flight_affected_index][0][0]
            airplane_speed = airplanes[airplanes[:,0]==airplane_index]
            airplane_speed = airplane_speed[0][1]

            filed_flight_path = filed_flights[filed_flights[:,0] == flight_affected_index,:]

            if airplane_speed not in self.nearest_neighbors_lookup:
                self.nearest_neighbors_lookup[airplane_speed] = {}

            unit_graph = unit_graphs[airplane_speed]


            start_time = np.flatnonzero(flight_affected != fill_value)
            start_time = start_time[0] if start_time.size else 0
            
            flight_affected = flight_affected[flight_affected != fill_value]

            if len(flight_affected) == 0:
                continue

            origin = flight_affected[0]
            destination = flight_affected[-1]


            paths = self.k_diverse_near_shortest_paths(unit_graph, origin, destination, self.nearest_neighbors_lookup[airplane_speed],
                                                k=k, eps=0.1, jaccard_max=0.6, penalty_scale=0.1, max_tries=50, weight_key="weight",
                                                filed_path=list(filed_flight_path[:,1]))

            graph_instance = []
            flight_sector_instances = []
            flight_times_instance = []
            sector_instance = [] 
            actual_arrival_time_instance = []
            actual_delay = 0
            path_number = 0

            all_variables_dict = {}

            max_time_generation = 0


            for path in paths:

                navpoint_trajectory = self.get_flight_navpoint_trajectory(flights_affected, unit_graph, flight_affected_index, start_time, airplane_speed, path, self._timestep_granularity)

                for delay in range(time_window):

                    current_time = start_time
                    max_delay_tmp = time_window 
                    delay_number = (delay) + path_number * max_delay_tmp 

                    if max_delay_tmp > max_effective_delay:
                        # FOR INFERRING DELAY LATER
                        max_effective_delay = max_delay_tmp

                    for flight_hop_index in range(len(navpoint_trajectory)):

                        navaid = navpoint_trajectory[flight_hop_index][1]
                        time = navpoint_trajectory[flight_hop_index][2]
                        #sector = (self.navaid_sector[self.navaid_sector[:,0] == navaid])[0,1]
                        sector = self.navaid_sector_lookup[navaid]

                        if sector not in next_sectors[flight_affected_index]:
                            next_sectors[flight_affected_index][sector] = {}

                        if len(navpoint_trajectory) > flight_hop_index + 1:
                            next_navaid = navpoint_trajectory[flight_hop_index+1][1]
                            next_sector = self.navaid_sector_lookup[next_navaid]
                            next_sectors[flight_affected_index][sector][next_sector] = True
                        
                        if sector not in previous_sectors[flight_affected_index]:
                            previous_sectors[flight_affected_index][sector] = {}

                        if flight_hop_index > 0:
                            prev_navaid = navpoint_trajectory[flight_hop_index-1][1]
                            prev_sector = self.navaid_sector_lookup[prev_navaid]
                            previous_sectors[flight_affected_index][sector][prev_sector] = True

                        if flight_hop_index == 0:

                            for time_index in range(actual_delay + delay + 1):

                                if time_index == actual_delay + delay:
                                    if f"x[{flight_affected_index},{delay_number},{current_time},{origin}]" not in all_variables_dict:
                                        all_variables_dict[f"x[{flight_affected_index},{delay_number},{current_time},{origin}]"] = True

                                        origin_variable = model.addVar(vtype=GRB.BINARY, name=f"x[{flight_affected_index},{delay_number},{current_time},{origin}]")

                                        entry = pd.DataFrame.from_dict({
                                            "F": [flight_affected_index],
                                            "D": [delay_number],
                                            "T": [current_time],
                                            "V": [origin],
                                            "obj": [origin_variable]
                                        })

                                        flight_variables_pd = pd.concat([flight_variables_pd,entry], ignore_index = True)

                                current_time += 1

                        else:

                            prev_navaid = navpoint_trajectory[flight_hop_index-1][1]
                            prev_time = navpoint_trajectory[flight_hop_index-1][2]
                            prev_sector = self.navaid_sector_lookup[prev_navaid]

                            if delay_number not in all_flight_navpoints_dict[flight_affected_index]:
                                all_flight_navpoints_dict[flight_affected_index][delay_number] = {}

                            if prev_navaid not in all_flight_navpoints_dict[flight_affected_index][delay_number]:
                                all_flight_navpoints_dict[flight_affected_index][delay_number][prev_navaid] = current_time - 1

                            #graph_instance[f"sectorEdge({prev_sector},{sector})."] = True

                            for time_index in range(1, time-prev_time + 1):
                                
                                #flight_times_instance.append(f"flightTime({flight_index},{flight_time}).")

                                if time_index <= math.floor((time - prev_time)/2):

                                    if f"x[{flight_affected_index},{delay_number},{current_time},{origin}]" not in all_variables_dict:
                                        all_variables_dict[f"x[{flight_affected_index},{delay_number},{current_time},{origin}]"] = True

                                        origin_variable = model.addVar(vtype=GRB.BINARY, name=f"x[{flight_affected_index},{delay_number},{current_time},{prev_sector}]")

                                        entry = pd.DataFrame.from_dict({
                                            "F": [flight_affected_index],
                                            "D": [delay_number],
                                            "T": [current_time],
                                            "V": [prev_sector],
                                            "obj": [origin_variable]
                                        })

                                        flight_variables_pd = pd.concat([flight_variables_pd,entry], ignore_index = True)

                                else:

                                    if f"x[{flight_affected_index},{delay_number},{current_time},{origin}]" not in all_variables_dict:
                                        all_variables_dict[f"x[{flight_affected_index},{delay_number},{current_time},{origin}]"] = True
                                        origin_variable = model.addVar(vtype=GRB.BINARY, name=f"x[{flight_affected_index},{delay_number},{current_time},{sector}]")

                                        entry = pd.DataFrame.from_dict({
                                            "F": [flight_affected_index],
                                            "D": [delay_number],
                                            "T": [current_time],
                                            "V": [sector],
                                            "obj": [origin_variable]
                                        })

                                        flight_variables_pd = pd.concat([flight_variables_pd,entry], ignore_index = True)

                                current_time += 1

                            if navaid not in all_flight_navpoints_dict[flight_affected_index][delay_number]:
                                all_flight_navpoints_dict[flight_affected_index][delay_number][navaid] = current_time - 1

                        if max_time_generation < current_time:
                            max_time_generation = current_time

                #actual_arrival_time_instance.append(f"actualArrivalTime({airplane_id},{current_time - 1},{path_number}).")
                # path_numbers = #PATHS * #DELAYS
                path_number += 1

        #considered_variables = flight_variables_pd[(flight_variables_pd['F']==2)&(flight_variables_pd['D']==2)]
        #print(considered_variables)
        #quit()

        for time in range(max_time_generation):
            for sector in range(capacity_time_matrix.shape[0]):
                capacity_variable = model.addVar(vtype=GRB.INTEGER, name=f"y[{time},{sector}]")

                entry = pd.DataFrame.from_dict({
                    "T": [time],
                    "S": [sector],
                    "obj": [capacity_variable]
                })
                sector_variables_pd = pd.concat([sector_variables_pd, entry], ignore_index = True)


        return flight_variables_pd, sector_variables_pd, next_sector, previous_sectors, max_effective_delay, all_flight_navpoints_dict


    def add_capacity_constraint(self, model: gp.Model, flight_variables_pd, sector_variables_pd, capacity_time_matrix, unit_graphs, converted_instance_matrix, fill_value = -1 , relaxed_version = True):
        """
        Add capacity constraint 
        """
        all_sector_vars = []

        capacity_computed_first_reached = False

        for sector in range(capacity_time_matrix.shape[0]):

            for time in range(capacity_time_matrix.shape[1]):

                sector_variables = []
                considered_rows = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd["T"]==time)]

                for flight in list(set(considered_rows['F'])):
                    for delay in list(set(considered_rows['D'])):

                        if capacity_computed_first_reached is False:
                            considered_rows_2 = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd['D']==delay)&(flight_variables_pd['F']==flight)&(flight_variables_pd['T']==time)]

                            if considered_rows_2.shape[0] > 0:
                                for _, considered_row_2 in considered_rows_2.iterrows():
                                    sector_variables.append(considered_row_2['obj'])

                        else:
                            considered_rows_2 = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd['D']==delay)&(flight_variables_pd['F']==flight)]

                            if considered_rows_2.shape[0] > 0:
                                min_time = min(list(considered_rows_2['T']))

                                if min_time != time:
                                    continue

                                considered_rows_3 = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd['D']==delay)&(flight_variables_pd['F']==flight)&(flight_variables_pd['T']==min_time)]

                                if considered_rows_3.shape[0] > 0:
                                    for _, considered_row_3 in considered_rows_3.iterrows():
                                        sector_variables.append(considered_row_3['obj'])

                if len(sector_variables) > 0:
                    #print(f"{['+'.join([var.VarName for var in sector_variables])]} <= {capacity_time_matrix[sector, time]}") 

                    if time < capacity_time_matrix.shape[1]:
                        cap_value = capacity_time_matrix[sector,time]
                    else:
                        cap_value = capacity_time_matrix[sector,0]

                    if relaxed_version is True:
                        considered_sector_rows = sector_variables_pd.loc[(sector_variables_pd["S"]==sector)&(sector_variables_pd['T']==time)]
                        if considered_sector_rows.shape[0] > 0:
                            for _, considered_sector_row in considered_sector_rows.iterrows():
                                model.addConstr(gp.quicksum(sector_variables) - cap_value <= considered_sector_row['obj'])
                                model.addConstr(0 <= considered_sector_row['obj'])

                                all_sector_vars.append(considered_sector_row['obj'])
                    else: 
                        model.addConstr(gp.quicksum(sector_variables) <= cap_value)

        if relaxed_version is True:  
           model.setObjectiveN(gp.quicksum(all_sector_vars),index=0,priority = 20)


    def add_flight_constraints_get_optimization_variables(self, model, flight_variables_pd, converted_instance_matrix,
                                                          edge_distances, max_delay, next_sectors, previous_sectors,
                                                          fill_value = -1):


        for flight in range(converted_instance_matrix.shape[0]):
            considered_flight = converted_instance_matrix[flight,:]

            considered_flight_time_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight))]

            min_flight_time = min(considered_flight_time_rows["T"])
            max_flight_time = max(considered_flight_time_rows["T"])

            considered_flight = considered_flight[considered_flight != fill_value]
            if len(considered_flight) == 0:
                continue

            #for time in range(converted_instance_matrix.shape[1]):
            for time in range(min_flight_time, max_flight_time + 1):

                considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time))]

                if considered_rows.shape[0] > 0:
                    ## ADD UNIQUE DELAY CONSTRAINT AND SUBSEQUENT SINGLE SECTOR CONSTRAINT
                    for sector in list(set(considered_rows["V"])):

                        delay_variables = []

                        delay_matrix = considered_rows.loc[((considered_rows["V"]==sector))]

                        for _, delay_row in delay_matrix.iterrows():
                            delay_variables.append(delay_row['obj'])
                            delay = delay_row["D"]
                            #subsequent_variables = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["V"]==sector)&(flight_variables_pd["D"]==delay))]
                            subsequent_variables = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["D"]==delay))]

                            if subsequent_variables.shape[0] == 1:
                                for _, subsequent_delay_variable in subsequent_variables.iterrows():
                                    # SUBSEQUENT CONSTRAINT
                                    model.addConstr(delay_row["obj"] <= subsequent_delay_variable["obj"])

                        # SINGLE PATH (DELAY) CONSTRAINT
                        #print(f"{['+'.join([var.VarName for var in delay_variables])]} <= 1") 
                        model.addConstr(gp.quicksum(delay_variables) <= 1)

                    ## ADD UNIQUE PATH CONSTRAINT:
                    variables_list = []

                    for _, flight_time_row in considered_rows.iterrows():
                        variables_list.append(flight_time_row['obj'])

                    # ADD UNIQUE PATH CONSTRAINT:
                    #print(f"{['+'.join([var.VarName for var in variables_list])]} <= 1") 
                    model.addConstr(gp.quicksum(variables_list) <= 1)

            ## START CONSTRAINT:
            start_variables = []
            considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight))]
            for delay_number in list(set(considered_rows['D'])):
                considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==delay_number))]

                min_time = min(list(set(considered_rows['T'])))

                start_variables_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]== delay_number)&(flight_variables_pd['T']==min_time))]

                for _, start_variable_row in start_variables_rows.iterrows():
                    start_variables.append(start_variable_row['obj'])
                        
            model.addConstr(1 <= gp.quicksum(start_variables))
            model.addConstr(gp.quicksum(start_variables) <= 1)




    def add_valid_inequalities(self, model: gp.Model, flight_variables_pd, converted_instance_matrix,
                                                          edge_distances, max_delay, fill_value = -1):

        optimization_variables = []

        for flight in range(converted_instance_matrix.shape[0]):

            flight_affected = converted_instance_matrix[flight,:]
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)]
            for sector in list(set(considered_rows["V"])):

                # VALID INEQUALITY 1)
                considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==sector))]
                for time in list(set(considered_rows_2["T"])):

                    considered_rows_constraint_8 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time)&(flight_variables_pd["V"]==sector))]

                    for _,variable_1_row in considered_rows_constraint_8.iterrows():
                        sector_variable = variable_1_row["obj"]
                        delay = variable_1_row["D"]

                    considered_rows_constraint_8_prim = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["D"]==delay))]
                    considered_rows_constraint_8_prim_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time)&(flight_variables_pd["D"]==delay))]

                    next_vertices = list(set(considered_rows_constraint_8_prim["V"]))
                    prev_vertices = list(set(considered_rows_constraint_8_prim_2["V"]))

                    sector_is_fork = True

                    fork_variables = []

                    for vertex in next_vertices:

                        if sector_is_fork is False:
                            break

                        if edge_distances[sector, vertex] == 1:

                            for prev_vertex in prev_vertices:
                                if edge_distances[prev_vertex,vertex] == 1 and prev_vertex != sector:
                                    sector_is_fork = False
                                    break

                            considered_rows_temp = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["D"]==delay)&(flight_variables_pd["V"]==vertex))]

                            for _,row in considered_rows_temp.iterrows():
                                fork_variables.append(row["obj"])

                    if sector_is_fork is True:
                        # Valid inequality 1)
                        model.addConstr(sector_variable - gp.quicksum(fork_variables) >= 0)

                # VALID INEQUALITY 2)
                sector_max_delay_pd = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["V"]==sector))]

                for _,row in sector_max_delay_pd.iterrows():
                    sector_variable = row["obj"]
                    time = row["T"]

                prev_sectors_pd = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["T"]==time-1))]
                cur_sectors_pd = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["T"]==time))]

                prev_vertices = list(set(prev_sectors_pd["V"]))
                current_vertices = list(set(cur_sectors_pd["V"]))

                sector_is_joint = True
                joint_variables = []

                for prev_vertex in prev_vertices:

                    if sector_is_joint is False:
                        break

                    if edge_distances[prev_vertex,sector] == 1:
                        for cur_vertex in current_vertices:
                            if edge_distances[prev_vertex, cur_vertex] == 1 and cur_vertex != sector:
                                sector_is_joint = False

                        considered_rows_temp = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time-1)&(flight_variables_pd["D"]==max_delay)&(flight_variables_pd["V"]==prev_vertex))]

                        for _,row in considered_rows_temp.iterrows():
                            joint_variables.append(row["obj"])

                if sector_is_joint is True:
                    #print(f"{['+'.join([var.VarName for var in joint_variables])]} - {sector_variable.VarName} <= 0") 
                    model.addConstr(gp.quicksum(joint_variables) - sector_variable <= 0)



            # VALID INEQUALITY 3):
            for delay in range(max_delay):
                considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==delay))]

                for time in list(set(considered_rows["T"])):
                    considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==delay)&(flight_variables_pd["T"]==time))]

                    vertices_anti_chain = list(set(considered_rows_2["V"]))

                    vertices_anti_chain_variables = []

                    for vertex in vertices_anti_chain:
                        considered_rows_3 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==delay)&(flight_variables_pd["T"]==time)&(flight_variables_pd["V"]==vertex))]

                        for _,row in considered_rows_3.iterrows():

                            vertices_anti_chain_variables.append(row["obj"])

                    # VALID INEQUALITY 3):
                    #print(f"{['+'.join([var.VarName for var in vertices_anti_chain_variables])]} <= 1")
                    model.addConstr(gp.quicksum(vertices_anti_chain_variables) <= 1)
                               

    """
    def optimize(self, model, optimization_variables):
            
        model.setObjective(gp.quicksum(optimization_variables), GRB.MINIMIZE)
        model.optimize()
    """

    def optimize(self, model: gp.Model, arrival_delay_optimization_variables, flight_variables_pd, sector_variables_pd, converted_instance_matrix, max_delay, all_flight_navpoint_dict):


        model.setObjectiveN(gp.quicksum(arrival_delay_optimization_variables), index=1, priority=10)



        last_t = {"t": -1e100}


        def cb(m: gp.Model, where: int):
            if where == GRB.Callback.MIP:
                t = m.cbGet(GRB.Callback.RUNTIME)
                if t - last_t["t"] < self.report_every_s:
                    return
                last_t["t"] = t


                best  = m.cbGet(GRB.Callback.MIP_OBJBST)  
                bound = m.cbGet(GRB.Callback.MIP_OBJBND)  
                nodes = m.cbGet(GRB.Callback.MIP_NODCNT)
                sols  = m.cbGet(GRB.Callback.MIP_SOLCNT)

                # Compute a robust relative gap (if incumbent exists)
                if best >= GRB.INFINITY or best <= -GRB.INFINITY:
                    gap = None  # no incumbent yet
                else:
                    denom = abs(best) + 1e-10   # avoids divide-by-zero if best == 0
                    gap = abs(best - bound) / denom

                gap_str = "n/a" if gap is None else f"{gap:.4%}"
                #print(f"[{t:8.1f}s] best={best:g}  bound={bound:g}  gap={gap_str}  nodes={nodes:.0f}  sols={sols}")

            elif where == GRB.Callback.MIPSOL:
                # Optional: called when a new solution is found :contentReference[oaicite:4]{index=4}
                t = m.cbGet(GRB.Callback.RUNTIME)
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                converted_instance_matrix_tmp, converted_navpoint_matrix_tmp = self.reconstruct_solution_cb(m, flight_variables_pd, sector_variables_pd, converted_instance_matrix, max_delay, all_flight_navpoint_dict)

                # -----------------------------------------------------------------------------
                t_init  = self.last_valid_pos(self.original_converted_instance_matrix)      # last non--1 in the *initial* schedule
                t_final = self.last_valid_pos(converted_instance_matrix_tmp)     # last non--1 in the *final* schedule
                
                diff_tmp = converted_instance_matrix_tmp.shape[1] - self.old_navaid_sector_time_assignment.shape[1]
                if diff_tmp > 0:
                    self.old_navaid_sector_time_assignment = np.hstack([self.old_navaid_sector_time_assignment, np.repeat(self.old_navaid_sector_time_assignment[:, [-1]], diff_tmp, axis=1)])

                diff_tmp = converted_instance_matrix_tmp.shape[1] - self.navaid_sector_time_assignment.shape[1]
                if diff_tmp > 0:
                    self.navaid_sector_time_assignment = np.hstack([self.navaid_sector_time_assignment, np.repeat(self.navaid_sector_time_assignment[:, [-1]], diff_tmp, axis=1)])

                delay = np.where(t_init >= 0, t_final - t_init, 0)          # shape (|I|,)
                system_loads = MIPModel.bucket_histogram(converted_instance_matrix_tmp, self.sectors, self.sectors.shape[0], converted_instance_matrix_tmp.shape[1], self._timestep_granularity)
                capacity_time_matrix = MIPModel.capacity_time_matrix(self.sectors, system_loads.shape[1], self._timestep_granularity, self.navaid_sector_time_assignment, z = self._sector_capacity_factor, composite_sector_function=self._composite_sector_function)
                capacity_demand_diff_matrix = capacity_time_matrix - system_loads
                capacity_overload_mask = capacity_demand_diff_matrix < 0
                number_of_conflicts = np.abs(capacity_demand_diff_matrix[capacity_overload_mask]).sum()
                total_delay  = delay.sum()
                number_sectors = self.compute_total_number_sectors(self.navaid_sector_time_assignment)
                sector_diff = np.count_nonzero(self.navaid_sector_time_assignment[:, 1:] != self.navaid_sector_time_assignment[:, :-1])
                original_max_time_converted = self.original_converted_instance_matrix.shape[1]  # original_max_time
                rerouted_mask = np.any(converted_instance_matrix_tmp[:, :original_max_time_converted] != self.original_converted_instance_matrix, axis=1)     # True if flight differs anywhere
                number_reroutes = int(np.count_nonzero(rerouted_mask))
                number_sector_reconfigurations = np.count_nonzero(self.navaid_sector_time_assignment != self.old_navaid_sector_time_assignment)

                current_time = time.time() - self.start_time
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


                #print(converted_instance_matrix_tmp)
                #print(converted_navpoint_matrix_tmp)


        model.optimize(cb)
        return self.progress
 
    def compute_total_number_sectors(self, navaid_sector_time_assignment):

        if navaid_sector_time_assignment.shape[0] == 0:
            number_sectors = 0
        else:
            S = np.sort(navaid_sector_time_assignment, axis=0)                       # sort within each column
            changes = (S[1:, :] != S[:-1, :])            # True where a new value starts
            uniq_per_col = 1 + changes.sum(axis=0)       # unique count per column
        number_sectors = int(uniq_per_col.sum())     # sum over all columns

        return number_sectors

    def last_valid_pos(self, arr: np.ndarray) -> np.ndarray:
        """
        Return a 1-D array with, for every row in `arr`, the **last** column index
        whose value is not -1.  If a row is all -1, we return -1 for that flight.
        """
        # True where value ≠ -1
        mask = arr != -1                                        # same shape as arr

        if not np.any(mask, where=True):
            return None

        # Reverse columns so that the *first* True along axis=1 is really the last
        # in the original orientation
        reversed_first = np.argmax(mask[:, ::-1], axis=1)

        # If the whole row was False, argmax returns 0.  Detect that case:
        no_valid = ~mask.any(axis=1)                            # shape (|I|,)

        # Convert “position in reversed array” back to real column index
        last_pos = arr.shape[1] - 1 - reversed_first            # shape (|I|,)
        last_pos[no_valid] = -1                                 # sentinel value

        return last_pos.astype(np.int64)





    def reconstruct_solution_cb(self, model, flight_variables_pd, sector_variables_pd, converted_instance_matrix,
                             max_delay, all_flight_navpoint_dict, fill_value = -1, tmp_output = False):

        result_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        converted_navpoint_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        for flight in range(converted_instance_matrix.shape[0]):

            flight_affected = converted_instance_matrix[flight,:]
            flight_affected = flight_affected[flight_affected != fill_value]

            if len(flight_affected) == 0:
                continue

            origin = flight_affected[0]
            destination = flight_affected[-1]

            #considered_rows_tmp = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==origin)]
            #start_time = min(list(considered_rows_tmp['T']))
            #considered_rows_tmp = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==origin)&(flight_variables_pd['T']==start_time)]
            #print(considered_rows_tmp)

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination)]
            actual_delay = max_delay

            for _,row in considered_rows.iterrows():

                #print(row["obj"])
                #print(model.cbGetSolution(row["obj"]))

                if model.cbGetSolution(row["obj"]) >= 1:
                    actual_delay = row["D"]

            for navaid, current_time in all_flight_navpoint_dict[flight][actual_delay].items():
                converted_navpoint_matrix[flight,current_time] = navaid
        
            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==actual_delay)]

            for _,row in considered_rows.iterrows():

                if model.cbGetSolution(row["obj"]) >= 1:
                    result_matrix[flight,row["T"]] = row["V"]

            """
            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==origin)&(flight_variables_pd["D"]<actual_delay)]

            for _,row in considered_rows.iterrows():
                if row["obj"].X >= 1:
                    result_matrix[flight,row["T"]] = row["V"]
            """

        return result_matrix, converted_navpoint_matrix



    def reconstruct_solution(self, model, flight_variables_pd, sector_variables_pd, converted_instance_matrix,
                             max_delay, all_flight_navpoint_dict, fill_value = -1, tmp_output = False):

        result_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        converted_navpoint_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        if tmp_output is False:
            solution_count = model.getAttr("SolCount")

            if solution_count == 0:
                return result_matrix, None


        for flight in range(converted_instance_matrix.shape[0]):

            flight_affected = converted_instance_matrix[flight,:]
            flight_affected = flight_affected[flight_affected != fill_value]

            if len(flight_affected) == 0:
                continue

            origin = flight_affected[0]
            destination = flight_affected[-1]

            #considered_rows_tmp = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==origin)]
            #start_time = min(list(considered_rows_tmp['T']))
            #considered_rows_tmp = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==origin)&(flight_variables_pd['T']==start_time)]
            #print(considered_rows_tmp)

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination)]
            actual_delay = max_delay

            for _,row in considered_rows.iterrows():

                if row["obj"].X >= 1:
                    actual_delay = row["D"]

            for navaid, current_time in all_flight_navpoint_dict[flight][actual_delay].items():
                converted_navpoint_matrix[flight,current_time] = navaid
        
            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==actual_delay)]

            for _,row in considered_rows.iterrows():

                if row["obj"].X >= 1:
                    result_matrix[flight,row["T"]] = row["V"]

            """
            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==origin)&(flight_variables_pd["D"]<actual_delay)]

            for _,row in considered_rows.iterrows():
                if row["obj"].X >= 1:
                    result_matrix[flight,row["T"]] = row["V"]
            """

        return result_matrix, converted_navpoint_matrix

 
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
        
    @classmethod
    def bucket_histogram(cls, instance_matrix: np.ndarray,
                         sectors: np.ndarray,                # unused, kept for signature compat
                         num_buckets: int,
                         n_times: int,
                         timestep_granularity: int,          # unused, kept for signature compat
                         *,
                         fill_value: int = -1) -> np.ndarray:

        inst = np.asarray(instance_matrix)
        if inst.ndim != 2:
            raise ValueError("instance_matrix must be 2D (flights x time)")
        F, T = inst.shape
        if T != n_times:
            raise ValueError(f"n_times ({n_times}) != instance_matrix.shape[1] ({T})")

        # Early exit if everything is fill_value
        valid = inst != fill_value
        if not valid.any():
            return np.zeros((num_buckets, T), dtype=np.int32)

        # Mark entries (new sector occurrences) at each time:
        # - t = 0: any valid value
        # - t > 0: valid and changed vs previous time
        """
        # Code for entry/diff demand measure:
        change = np.zeros_like(valid, dtype=bool)
        change[:, 0] = valid[:, 0]
        if T > 1:
            change[:, 1:] = valid[:, 1:] & (inst[:, 1:] != inst[:, :-1])
        """
        change = valid

        # Gather (sector_id, time_idx) pairs where an entry happens
        sectors_at_entries = inst[change]
        time_idx = np.nonzero(change)[1]  # column indices where change==True

        # Optional safety: ensure sector ids are in [0, num_buckets)
        if sectors_at_entries.size:
            mn = int(sectors_at_entries.min())
            mx = int(sectors_at_entries.max())
            if mn < 0 or mx >= num_buckets:
                raise ValueError(
                    f"sector id(s) out of range [0, {num_buckets}): found min={mn}, max={mx}"
                )

        # Scatter-add 1 for each (sector, time) event
        hist = np.zeros((num_buckets, T), dtype=np.int32)
        np.add.at(hist, (sectors_at_entries, time_idx), 1)

        #np.savetxt("20251003_histogram.csv", hist, delimiter=",",fmt="%i")

        return hist
    

    @classmethod
    def capacity_time_matrix(cls,
                            cap: np.ndarray,
                            n_times: int,
                            time_granularity: int,
                            navaid_sector_time_assignment: np.ndarray,
                            z=1,
                            composite_sector_function = MAX
                            ) -> np.ndarray:
        """
        Same I/O and validations as before. Now we:
        1) scatter-add to get per-(sector,time) SUM and COUNT,
        2) call cls.compute_sector_capacity(avg, count, z)  <-- explicit rule, vectorized,
        3) distribute remainder over T slots.
        """
        N = cap.shape[0]
        T = int(time_granularity)

        if n_times % T != 0:
            raise ValueError("n_times must be a multiple of time_granularity (T).")

        if navaid_sector_time_assignment.shape != (N, n_times):
            raise ValueError(
                f"navaid_sector_time_assignment must be shape (N, n_times) = ({N}, {n_times})"
            )

        S = navaid_sector_time_assignment.astype(np.int64, copy=False)
        if S.min() < 0 or S.max() >= N:
            print(S)
            raise ValueError("Sector ids in navaid_sector_time_assignment must be in [0, N-1].")

        # ---- 1) Sum atomic per-block caps and contributor counts per (sector,time)
        atomic_block_cap = np.asarray(cap[:, 1], dtype=np.int64)  # length N

        total_atomic_sum = np.zeros((N, n_times), dtype=np.int64)
        contrib_count    = np.zeros((N, n_times), dtype=np.int64)

        S_flat      = S.ravel(order="C")
        t_idx_flat  = np.tile(np.arange(n_times, dtype=np.int64), N)
        cap_rep_flat= np.repeat(atomic_block_cap, n_times)

        np.add.at(total_atomic_sum, (S_flat, t_idx_flat), cap_rep_flat)
        np.add.at(contrib_count,    (S_flat, t_idx_flat), 1)


        max_atomic = None
        avg = None

        if composite_sector_function == MAX:
            # Also compute per-(sector,time) MAX for the "max" rule
            max_atomic = np.full((N, n_times), np.iinfo(np.int64).min, dtype=np.int64)
            np.maximum.at(max_atomic, (S_flat, t_idx_flat), cap_rep_flat)
            max_atomic = np.where(contrib_count > 0, max_atomic, 0)

        if composite_sector_function == TRIANGULAR:
            # Average with safe denom (still useful for triangular & linear rules)
            avg = total_atomic_sum.astype(np.float64) / np.maximum(1, contrib_count)

        # ---- 2) Explicit, *vectorized* capacity rule
        # Returns per-block integer capacities
        total_capacity = cls.compute_sector_capacity(contrib_count, float(z), avg_ = avg,
                                                            max_ = max_atomic, sum_=total_atomic_sum, function = composite_sector_function)

        # ---- 3) Distribute per-block capacity across T slots (unchanged)
        base = total_capacity // T
        rem  = total_capacity - base * T

        rem_table = cls._remainder_distribution_table(T)   # (T+1, T)
        k_mod     = np.arange(n_times, dtype=np.int64) % T

        extra      = rem_table[rem, k_mod[None, :]]
        sector_cap = (base + extra).astype(np.int64, copy=False)

        return sector_cap

    @classmethod
    def create_initial_navpoint_sector_assignment(cls,
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
     

    @classmethod
    def compute_sector_capacity(cls,
                                counts: np.ndarray,
                                z: float,
                                avg_ = None,
                                max_ = None,
                                sum_ = None,
                                function = "triangular"
                                ) -> np.ndarray:
        """
        Vectorized capacity rule (edit here to change behavior).
        Inputs:
            avg    : (N, n_times) float64, mean atomic per-block capacity for each (sector,time)
            counts : (N, n_times) int64, number of contributors k per (sector,time)
            z      : float, rule parameter
        Returns:
            int64 (N, n_times) per-block capacities BEFORE remainder distribution.
        Current rule (matches your earlier piecewise):
            if k > z:   cap = round(((z+1)/2) * avg)
            else:       cap = round(avg * triangular_weight_sum(k, z))
        """
        
        empty_mask = (counts == 0)

        if function == LINEAR:
            if sum_ is None:
                raise ValueError("compute_sector_capacity(rule='linear') requires sum_.")
            out = sum_.astype(np.int64, copy=False)
            out = np.where(empty_mask, 0, out)
            return out
        
        if function == MAX:

            if max_ is None:
                raise ValueError("compute_sector_capacity(rule='max') requires max_.")
            out = max_.astype(np.int64, copy=False)
            out = np.where(empty_mask, 0, out)
            return out
        
        if function == TRIANGULAR:

            counts = counts.astype(np.int64, copy=False)
            avg    = avg_.astype(np.float64, copy=False)

            tri    = cls._triangular_weight_sum_counts(counts, z)

            out = np.where(
                counts > z,
                ((z + 1.0) / 2.0) * avg,
                avg * tri
            )

            # Ensure empty groups yield 0 exactly
            out = np.where(counts == 0, 0.0, out)

            return np.rint(out).astype(np.int64)
 
    @classmethod 
    def _triangular_weight_sum_counts(cls, counts: np.ndarray, denom: float) -> np.ndarray:
        """
        Vectorized: for each integer count k, compute
        sum_{i=0}^{m-1} (1 - i/denom),
        where m = min(k, number of positive weights), see compute_sector_capacity.
        Returns an array of same shape as counts (float64).
        """
        counts = counts.astype(np.int64, copy=False)
        d = float(denom)
        if d <= 0:
            return counts.astype(np.float64)  # no diminishing

        d_floor = np.floor(d)
        # m_pos = floor(d) if d integer else floor(d)+1
        m_pos = np.where(np.isclose(d, d_floor), d_floor, d_floor + 1.0)
        m = np.minimum(counts.astype(np.float64), np.maximum(0.0, m_pos))

        # tri = m - (m-1)m/(2*d)
        tri = m - (m - 1.0) * m / (2.0 * d)
        return tri

    @classmethod
    def _remainder_distribution_table(slc, T: int) -> np.ndarray:
        """
        Build a (T+1, T) table where row r gives, for remainder r,
        the number of extra +1 drops that land at each index k∈[0..T-1]
        when stepping by ceil(T/r) and wrapping mod T, for r steps.

        Row 0 is all zeros (no remainder to distribute).
        """
        table = np.zeros((T + 1, T), dtype=np.int64)
        for r in range(1, T):  # r = remainder, strictly < T
            step = (T + r - 1) // r  # ceil(T / r)
            hits = (np.arange(r, dtype=np.int64) * step) % T
            # count duplicates (they matter!)
            cnt = np.bincount(hits, minlength=T)
            table[r, :cnt.size] = cnt
        return table








