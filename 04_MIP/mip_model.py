
import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np


class MIPModel:

    def __init__(self, airport_vertices, max_time, max_explored_vertices, seed):

        self.airport_vertices = airport_vertices
        self._max_time = max_time
        self._max_explored_vertices = max_explored_vertices
        self._seed = seed

        self.env = gp.Env(empty=True)          
        self.env.setParam('OutputFlag', 1)     
        self.env.start()



    def create_model(self, converted_instance_matrix, capacity_time_matrix, edge_distances, max_delay):

        model = gp.Model(env=self.env, name="toy")

        flight_variables_pd = self.add_variables(model, converted_instance_matrix, edge_distances, max_delay)

        model.update()

        self.add_capacity_constraint(model, flight_variables_pd, capacity_time_matrix, edge_distances)

        model.update()

        optimization_variables = self.add_flight_constraints_get_optimization_variables(model, flight_variables_pd, converted_instance_matrix, edge_distances, max_delay)

        model.update()

        self.optimize(model, optimization_variables)

        converted_instance_matrix = self.reconstruct_solution(model, flight_variables_pd, converted_instance_matrix, max_delay)

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


    def add_capacity_constraint(self, model, flight_variables_pd, capacity_time_matrix, edge_distances, ):

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

                    model.addConstr(gp.quicksum(positive_capacity_variables) - gp.quicksum(negative_capacity_variables) <= capacity_time_matrix[sector, time]) 


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
 
        return optimization_variables


    def add_valid_inequalities():
        pass


    def optimize(self, model, optimization_variables):
            
        model.setObjective(gp.quicksum(optimization_variables), GRB.MINIMIZE)
        model.optimize()

    def reconstruct_solution(self, model, flight_variables_pd,  converted_instance_matrix, max_delay, fill_value = -1):
 

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
        





