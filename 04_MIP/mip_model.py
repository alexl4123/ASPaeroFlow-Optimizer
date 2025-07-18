
import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np


class MIPModel:

    def __init__(self, airport_vertices, max_time, max_explored_vertices, seed, timestep_granularity, max_threads, capacity, verbosity):

        self.airport_vertices = airport_vertices
        self._max_time = max_time
        self._max_explored_vertices = max_explored_vertices
        self._seed = seed
        self._timestep_granularity = timestep_granularity
        self._max_number_threads = max_threads
        self.capacity = capacity

        self.env = gp.Env(empty=True)          
        self.env.setParam('OutputFlag', 1)     
        self.env.setParam(GRB.Param.Seed, seed)

        if verbosity == 0:
            self.env.setParam("OutputFlag", 0)


        self.env.start()

    def create_model(self, converted_instance_matrix, capacity_time_matrix, edge_distances, max_delay):

        solution = None
        self._max_time += max_delay


        if converted_instance_matrix.shape[1] < self._max_time:
            diff = self._max_time - converted_instance_matrix.shape[1]
            extra_col = -1 * np.ones((converted_instance_matrix.shape[0], diff), dtype=int)
            converted_instance_matrix = np.hstack((converted_instance_matrix, extra_col)) 

        system_loads = self.bucket_histogram(converted_instance_matrix, capacity_time_matrix.shape[0], self._timestep_granularity)
        capacity_time_matrix = self.capacity_time_matrix(self.capacity,system_loads.shape[1])

        while solution is None:

            model = gp.Model(env=self.env, name="toy")
            model.Params.Threads = self._max_number_threads

            flight_variables_pd = self.add_variables(model, converted_instance_matrix, edge_distances, max_delay)

            model.update()

            self.add_capacity_constraint(model, flight_variables_pd, capacity_time_matrix, edge_distances, converted_instance_matrix)

            model.update()

            optimization_variables = self.add_flight_constraints_get_optimization_variables(model, flight_variables_pd, converted_instance_matrix, edge_distances, max_delay)

            model.update()

            self.add_valid_inequalities(model, flight_variables_pd, converted_instance_matrix, edge_distances, max_delay)

            self.optimize(model, optimization_variables)

            converted_instance_matrix = self.reconstruct_solution(model, flight_variables_pd, converted_instance_matrix, max_delay)

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


    def add_variables(self, model, converted_instance_matrix, edge_distances, max_delay, fill_value = -1):

        # |F|x|D|x|T|x|V|
        # F=flights, D=possible-delays, T=time, V=vertices
        flight_variables_pd = pd.DataFrame(columns=["F","D","T","V","obj"])
        considered_vertices = set()

        #max_delay = max_delay + 1

        for flight_affected_index in range(converted_instance_matrix.shape[0]):

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

                        origin_variable = model.addVar(vtype=GRB.BINARY, name=f"x[{flight_affected_index},{delay},{possible_time},{vertex_id}]")

                        entry = pd.DataFrame.from_dict({
                            "F": [flight_affected_index],
                            "D": [delay],
                            "T": [possible_time],
                            "V": [vertex_id],
                            "obj": [origin_variable]
                        })

                        flight_variables_pd = pd.concat([flight_variables_pd,entry], ignore_index = True)

        return flight_variables_pd


    def add_capacity_constraint(self, model, flight_variables_pd, capacity_time_matrix, edge_distances, converted_instance_matrix, fill_value = -1 ):

        ############################
        # CONSTRAINTS (1,2,3,4):
        for sector in range(capacity_time_matrix.shape[0]):

            timestep_granularity_window = []

            for time in range(self._max_time):

                considered_rows = flight_variables_pd.loc[(flight_variables_pd["V"]==sector)&(flight_variables_pd["T"]==time)]

                if considered_rows.shape[0] > 0:

                    positive_capacity_variables = []
                    negative_capacity_variables = []

                    for _, row in considered_rows.iterrows():

                        flight = row["F"]
                        delay = row["D"]

                        positive_capacity_variables.append(row["obj"])

                        next_considered_rows = flight_variables_pd.loc[(flight_variables_pd['F']==flight)&(flight_variables_pd['D']==delay)&(flight_variables_pd["T"]==time+1)]

                        if considered_rows.shape[0] == 0:
                            continue

                        vertices = list(set(next_considered_rows["V"]))

                        for vertex in vertices:
                            if edge_distances[sector,vertex] == 1:
                                # VARIABLES AT SAME TIME (==time):
                                next_considered_rows = flight_variables_pd.loc[(flight_variables_pd['F']==flight)&(flight_variables_pd['V']==vertex)&(flight_variables_pd["T"]==time)]

                                for _,vertex_row in next_considered_rows.iterrows():
                                    negative_capacity_variables.append(vertex_row["obj"])

                    timestep_granularity_window.append((positive_capacity_variables, negative_capacity_variables))

                    all_pos_variables = []
                    all_neg_variables = []
                    for pos_vars, neg_vars in timestep_granularity_window:
                        all_pos_variables += pos_vars
                        all_neg_variables += neg_vars

                    model.addConstr(gp.quicksum(all_pos_variables) - gp.quicksum(all_neg_variables) <= capacity_time_matrix[sector, time]) 


                    if len(timestep_granularity_window) >= self._timestep_granularity:
                        timestep_granularity_window.pop(0)



    def add_flight_constraints_get_optimization_variables(self, model, flight_variables_pd, converted_instance_matrix,
                                                          edge_distances, max_delay, fill_value = -1):


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

                # CONSTRAINTS (SELF -> ENFORCE NO EN-ROUTE HOLDING DELAY): 
                considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time))]

                for sector in list(set(considered_rows_2["V"])):
                    if sector != origin and sector != destination:
                        
                        considered_rows_3 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["T"]==time)&(flight_variables_pd["V"]==sector))]

                        for _,row in considered_rows_3.iterrows():

                            left_variable = row["obj"]
                            right_variables = []
                            
                            considered_rows_4 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==row["D"])&(flight_variables_pd["T"]==time+1))]

                            vertices = list(set(considered_rows_4["V"]))

                            for vertex in vertices:

                                if edge_distances[sector,vertex] == 1:
                                    considered_rows_5 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==row["D"])&(flight_variables_pd["T"]==time+1)&(flight_variables_pd["V"]==vertex))]

                                    for _,vertex_row in considered_rows_5.iterrows():

                                        right_variables.append(vertex_row["obj"])

                            # CONSTRAINT (NO EN-ROUTE HOLDING DELAY):
                            model.addConstr(left_variable - gp.quicksum(right_variables) <= 0)
    


            # CONSTRAINTS (5)&(6): 
            considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["D"]==max_delay))]

            for sector in list(set(considered_rows_2["V"])):

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
                        model.addConstr(left_variable - gp.quicksum(right_variables) <= 0)

                        # CONSTRAINT (6):
                        model.addConstr(gp.quicksum(right_variables) <= 1) 

                
            considered_rows_2 = flight_variables_pd.loc[((flight_variables_pd["F"]==flight)&(flight_variables_pd["V"]==origin)&(flight_variables_pd["D"]==0))]
            
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
 
        return optimization_variables


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

    def reconstruct_solution(self, model, flight_variables_pd,  converted_instance_matrix, max_delay, fill_value = -1):
 

        result_matrix = -1 * np.ones((converted_instance_matrix.shape[0], converted_instance_matrix.shape[1] + max_delay), dtype=int)

        solution_count = model.getAttr("SolCount")

        if solution_count == 0:
            return result_matrix


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
    







