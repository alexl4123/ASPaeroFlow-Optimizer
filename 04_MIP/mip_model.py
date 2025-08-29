
import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np
import networkx as nx
import math


class MIPModel:

    def __init__(self, airport_vertices, max_time, max_explored_vertices, seed, timestep_granularity, verbosity, number_threads, navaid_sector_lookup):

        self.airport_vertices = airport_vertices
        self._max_time = max_time
        self._max_explored_vertices = max_explored_vertices
        self._seed = seed
        self._timestep_granularity = timestep_granularity

        self._max_number_threads = number_threads
        self.navaid_sector_lookup = navaid_sector_lookup

        self.env = gp.Env(empty=True)          
        self.env.setParam('OutputFlag', 1)     
        self.env.setParam(GRB.Param.Seed, seed)

        if verbosity == 0:
            self.env.setParam("OutputFlag", 0)


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

    def create_model(self, converted_instance_matrix, capacity_time_matrix, unit_graphs, airplanes, max_delay, planned_arrival_times):

        solution = None
        self._max_time += max_delay

        if converted_instance_matrix.shape[1] < self._max_time:
            diff = self._max_time - converted_instance_matrix.shape[1]
            extra_col = -1 * np.ones((converted_instance_matrix.shape[0], diff), dtype=int)
            converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

        while solution is None:

            model = gp.Model(env=self.env, name="toy")
            model.Params.Threads = self._max_number_threads

            flight_variables_pd, sector_variables_pd, next_sectors, previous_sectors, max_effective_delay = self.add_variables(model, converted_instance_matrix, unit_graphs, airplanes, max_delay)

            model.update()

            self.add_capacity_constraint(model, flight_variables_pd, sector_variables_pd, capacity_time_matrix, unit_graphs, converted_instance_matrix)


            model.update()

            optimization_variables = self.add_flight_constraints_get_optimization_variables(model, flight_variables_pd, converted_instance_matrix, unit_graphs, max_delay, next_sectors, previous_sectors)

            model.update()

            #self.add_valid_inequalities(model, flight_variables_pd, converted_instance_matrix, edge_distances, max_delay)

            optimization_variables = self.construct_objective_function(flight_variables_pd, converted_instance_matrix, planned_arrival_times)

            self.optimize(model, optimization_variables)

            converted_instance_matrix = self.reconstruct_solution(model, flight_variables_pd, sector_variables_pd, converted_instance_matrix, max_delay)

            mask = converted_instance_matrix != -1                                        # same shape as arr

            if not np.any(mask, where=True):
                solution = None
                max_delay = max_delay + 1
                self._max_time = + 1

                if converted_instance_matrix.shape[1] < self._max_time:
                    diff = self._max_time - converted_instance_matrix.shape[1]
                    extra_col = -1 * np.ones((converted_instance_matrix.shape[0], diff), dtype=int)
                    converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

                system_loads = self.bucket_histogram(converted_instance_matrix, capacity_time_matrix.shape[0], self._timestep_granularity)
                capacity_time_matrix = self.capacity_time_matrix(self.capacity,system_loads.shape[1])


            else:
                solution = converted_instance_matrix

        return converted_instance_matrix
    
    def k_diverse_near_shortest_paths(
        self,
        G, s, t, nearest_neighbors_lookup, k=5, eps=0.10, jaccard_max=0.6,
        penalty_scale=0.5, max_tries=200, weight_key="weight"
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
 

    def add_variables(self, model, converted_instance_matrix, unit_graphs, airplanes, max_delay, fill_value = -1):

        # |F|x|D|x|T|x|V|
        # F=flights, D=possible-delays, T=time, V=vertices
        flight_variables_pd = pd.DataFrame(columns=["F","D","T","V","obj"])
        sector_variables_pd = pd.DataFrame(columns=["F","D","T","V","obj"])
        considered_vertices = set()

        max_effective_delay = 0

        next_sectors = {}
        previous_sectors = {}

        #max_delay = max_delay + 1
        self.nearest_neighbors_lookup = {}
        k = 6

        for flight_affected_index in range(converted_instance_matrix.shape[0]):

            next_sectors[flight_affected_index] = {}
            previous_sectors[flight_affected_index] = {}


            flight_affected = converted_instance_matrix[flight_affected_index,:]
            flights_affected = converted_instance_matrix[np.array([flight_affected_index]),:]

            airplane_speed = airplanes[airplanes[:,0]==flight_affected_index][0,1]
            airplane_id = airplanes[airplanes[:,0]==flight_affected_index][0,0]

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
                                                k=k, eps=0.5, jaccard_max=0.6, penalty_scale=0.2, max_tries=200, weight_key="weight")

            graph_instance = []
            flight_sector_instances = []
            flight_times_instance = []
            sector_instance = [] 
            actual_arrival_time_instance = []
            additional_time_increase = 0
            time_window = 10
            actual_delay = 0
            path_number = 0

            all_variables_dict = {}

            for path in paths:

                navpoint_trajectory = self.get_flight_navpoint_trajectory(flights_affected, unit_graph, flight_affected_index, start_time, airplane_speed, path, self._timestep_granularity)

                for delay in range(additional_time_increase * time_window,time_window * (additional_time_increase + 1)):

                    flight_time = 0
                    current_time = start_time
                    max_delay_tmp = time_window * (additional_time_increase + 1) 
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

                            for _ in range(actual_delay + delay + 1):


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

                                flight_time += 1
                                current_time += 1

                        else:

                            prev_navaid = navpoint_trajectory[flight_hop_index-1][1]
                            prev_time = navpoint_trajectory[flight_hop_index-1][2]
                            prev_sector = self.navaid_sector_lookup[prev_navaid]

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
                                flight_time += 1

                #actual_arrival_time_instance.append(f"actualArrivalTime({airplane_id},{current_time - 1},{path_number}).")
                # path_numbers = #PATHS * #DELAYS
                path_number += 1

        #considered_variables = flight_variables_pd[(flight_variables_pd['F']==0)&(flight_variables_pd['D']==30)]
        #print(considered_variables)
        #quit()

        return flight_variables_pd, sector_variables_pd, next_sector, previous_sectors, max_effective_delay


    def add_capacity_constraint(self, model: gp.Model, flight_variables_pd, sector_variables_pd, capacity_time_matrix, unit_graphs, converted_instance_matrix, fill_value = -1 ):
        """
        Add capacity constraint 
        """

        for sector in range(capacity_time_matrix.shape[0]):

            for time in range(capacity_time_matrix.shape[1]):

                sector_variables = []
                considered_rows = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd["T"]==time)]

                for flight in list(set(considered_rows['F'])):
                    for delay in list(set(considered_rows['D'])):

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
                    model.addConstr(gp.quicksum(sector_variables) <= capacity_time_matrix[sector, time])

    def add_flight_constraints_get_optimization_variables(self, model, flight_variables_pd, converted_instance_matrix,
                                                          edge_distances, max_delay, next_sectors, previous_sectors,
                                                          fill_value = -1):


        optimization_variables = []

        for flight in range(converted_instance_matrix.shape[0]):
            considered_flight = converted_instance_matrix[flight,:]
            considered_flight = considered_flight[considered_flight != fill_value]
            if len(considered_flight) == 0:
                continue

            for time in range(converted_instance_matrix.shape[1]):

                considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time))]

                if considered_rows.shape[0] > 0:
                    ## ADD UNIQUE DELAY CONSTRAINT AND SUBSEQUENT SINGLE SECTOR CONSTRAINT
                    for sector in list(set(considered_rows["V"])):

                        delay_variables = []

                        delay_matrix = considered_rows.loc[((considered_rows["V"]==sector))]

                        for _, delay_row in delay_matrix.iterrows():
                            delay_variables.append(delay_row['obj'])
                            delay = delay_row["D"]
                            subsequent_variables = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["V"]==sector)&(flight_variables_pd["D"]==delay))]

                            if subsequent_variables.shape[0] == 1:
                                for _, subsequent_delay_variable in subsequent_variables.iterrows():
                                    # SUBSEQUENT CONSTRAINT
                                    model.addConstr(delay_row["obj"] <= subsequent_delay_variable["obj"])

                        # SINGLE DELAY CONSTRAINT
                        #print(f"{['+'.join([var.VarName for var in delay_variables])]} <= 1") 
                        model.addConstr(gp.quicksum(delay_variables) <= 1)

                    ## ADD UNIQUE PATH CONSTRAINT:
                    variables_list = []

                    for _, flight_time_row in considered_rows.iterrows():
                        variables_list.append(flight_time_row['obj'])

                    # ADD UNIQUE PATH CONSTRAINT:

                    #print(f"{['+'.join([var.VarName for var in variables_list])]} <= 1") 
                    model.addConstr(gp.quicksum(variables_list) <= 1)

            #####################################################################################
            #####################################################################################
            # GRAPH CONSTRAINTS:

            considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight))]

            for delay in list(set(considered_rows['D'])):

                considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==delay))]

                for sector in list(set(considered_rows_2['V'])):
                    
                    considered_rows_3 = flight_variables_pd.loc[((flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd['V']==sector))]

                    # NEXT CONSTRAINT:
                    max_time = max(list(set(considered_rows_3['T'])))
                    variable_rows = flight_variables_pd.loc[((flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd['V']==sector)&(flight_variables_pd['T']==max_time))]

                    for _, variable_row in variable_rows.iterrows():

                        variable = variable_row['obj']
                        variable_rows = flight_variables_pd.loc[((flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd['T']==max_time+1))]

                        next_variables = []
                        for _,next_row in variable_rows.iterrows():

                            next_variables.append(next_row['obj'])

                        if len(next_variables) > 0:
                            #print(f"{variable.VarName} - {['+'.join([var.VarName for var in next_variables])]} <= 0") 
                            model.addConstr(variable - gp.quicksum(next_variables) <= 0)


                    # PREV CONSTRAINT:
                    min_time = min(list(set(considered_rows_3['T'])))
                    variable_rows = flight_variables_pd.loc[((flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd['V']==sector)&(flight_variables_pd['T']==min_time))]

                    for _, variable_row in variable_rows.iterrows():

                        variable = variable_row['obj']
                        variable_rows = flight_variables_pd.loc[((flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd['T']==min_time-1))]

                        prev_variables = []
                        for _,next_row in variable_rows.iterrows():

                            prev_variables.append(next_row['obj'])

                        if len(prev_variables) > 0:

                            model.addConstr(gp.quicksum(prev_variables) - variable >= 0)


            ## START CONSTRAINT:
            considered_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight))]

            min_time = min(list(set(considered_rows['T'])))
            start_variables_rows = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd['T']==min_time))]

            start_variables = []
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
                               

    def optimize(self, model, optimization_variables):
            
        model.setObjective(gp.quicksum(optimization_variables), GRB.MINIMIZE)
        model.optimize()

    def reconstruct_solution(self, model, flight_variables_pd, sector_variables_pd, converted_instance_matrix, max_delay, fill_value = -1):

        result_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        solution_count = model.getAttr("SolCount")

        if solution_count == 0:
            return result_matrix


        for flight in range(converted_instance_matrix.shape[0]):

            flight_affected = converted_instance_matrix[flight,:]
            flight_affected = flight_affected[flight_affected != fill_value]

            if len(flight_affected) == 0:
                continue

            origin = flight_affected[0]
            destination = flight_affected[-1]

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==destination)]
            
            considered_rows_tmp = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==origin)]
            start_time = min(list(considered_rows_tmp['T']))
            considered_rows_tmp = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd['V']==origin)&(flight_variables_pd['T']==start_time)]
            #print(considered_rows_tmp)

            actual_delay = max_delay

            for _,row in considered_rows_tmp.iterrows():

                if row["obj"].X >= 1:
                    actual_delay = row["D"]

            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==actual_delay)]

            for _,row in considered_rows.iterrows():

                if row["obj"].X >= 1:
                    result_matrix[flight,row["T"]] = row["V"]


            considered_rows = flight_variables_pd.loc[(flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==origin)&(flight_variables_pd["D"]<actual_delay)]

            for _,row in considered_rows.iterrows():
                if row["obj"].X >= 1:
                    result_matrix[flight,row["T"]] = row["V"]

        return result_matrix

 
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
        

    def bucket_histogram(self, instance_matrix: np.ndarray,
                        num_buckets: int,
                        timestep_granularity: int,
                        *,
                        fill_value: int = -1) -> np.ndarray:
        """
        Count how many elements occupy each *bucket* at each *time step*.

        Parameters
        ----------
        instance_matrix
            2-D array produced by `instance_to_matrix`  
            shape = (num_elements, num_time_steps)
        num_buckets
            Equals `self.capacity.shape[0]`
        fill_value
            The placeholder used for “no assignment” (default −1)

        Returns
        -------
        counts : np.ndarray
            shape = (num_buckets, num_time_steps)  
            `counts[bucket, t]` is the occupancy of *bucket* at time *t*.
        """
        n_elems, n_times = instance_matrix.shape

        # ------------------------------------------------------------------
        # 1.  Mask out empty cells (if any)
        # ------------------------------------------------------------------

        # Generate mask of values that differ from fill_value = -1
        valid_mask = instance_matrix != fill_value
        if not np.any(valid_mask):                       
            # Shortcut if all cells empty
            return np.zeros((num_buckets, n_times), dtype=int)

        # ------------------------------------------------------------------
        # 2.  Gather bucket IDs and their time indices
        # ------------------------------------------------------------------
        buckets = instance_matrix[valid_mask]                        # (K,) bucket id
        # 

        # np.nonzero --> Get indices of non-zero elements of valid_mask (non false)
        # --> with np.nonzero(valid_mask)[1] we take the time indices

        times   = np.nonzero(valid_mask)[1]              # (K,) time index

        # ------------------------------------------------------------------
        # 3.  Vectorised scatter using `np.add.at`
        # ------------------------------------------------------------------
        counts = np.zeros((num_buckets, n_times), dtype=int)

        # Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’.
        # ufunc.at(a, indices, b=None, /)
        # --> Buckets and times specify the indices
        np.add.at(counts, (buckets, times), 1)

        # Performs a sliding window aggregation according to timestep-granularity
        # To account for hour/minute/etc. computation
        axis = 1

        pad_width = [(0, 0)] * counts.ndim
        pad_width[axis] = (0, timestep_granularity - 1)
        padded = np.pad(counts, pad_width, mode="constant")

        windows = np.lib.stride_tricks.sliding_window_view(padded,
                                                    window_shape=timestep_granularity,
                                                    axis=axis)
        windows = windows.sum(axis=2)

        return windows

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
    







