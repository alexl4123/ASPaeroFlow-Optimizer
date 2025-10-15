from __future__ import annotations

import time
import math

import numpy as np
import networkx as nx
from itertools import islice

from solver import Solver, Model

from networkx.algorithms.boundary import node_boundary
from sympy import bell

# pip install more-itertools
from more_itertools import set_partitions
from itertools import islice
    
class OptimizeFlights:

    def __init__(self,
                 encoding, capacity, graph, airport_vertices,
                 problematic_flight_indices, 
                 converted_instance_matrix, converted_navpoint_matrix,
                 time_index, sector_index,
                 capacity_time_matrix, capacity_demand_diff_matrix,
                 additional_time_increase,
                 networkx_graph,
                 unit_graphs,
                 planned_arrival_times,
                 airplane_flight,
                 airplanes,
                 problematic_flights,
                 navaid_sector_time_assignment,
                 nearest_neighbors_lookup,
                 sector_capacity_factor,
                 filed_flights,
                 problematic_airplane_flight_map,
                 fill_value = -1,
                 timestep_granularity = 1,
                 seed = 11904657,
                 max_vertices_cutoff_value = 6,
                 max_delay_parameter = 4,
                 original_max_time = 24,
                 iteration = 0,
                 verbosity = 0,
                 ):

        self.encoding = encoding
        self.capacity = capacity
        self.graph = graph
        self.airport_vertices = airport_vertices
        self.networkx_graph = networkx_graph
        self.unit_graphs = unit_graphs
        self.planned_arrival_times = planned_arrival_times
        self.airplane_flight = airplane_flight
        self.problematic_flights = problematic_flights
        self.navaid_sector_time_assignment = navaid_sector_time_assignment
        self.airplanes = airplanes
        self.nearest_neighbors_lookup = nearest_neighbors_lookup

        self.filed_flights = filed_flights
        self.sector_capacity_factor = sector_capacity_factor
        
        self.problematic_airplane_flight_map = problematic_airplane_flight_map
        self.problematic_flight_indices = problematic_flight_indices

        self.converted_instance_matrix = converted_instance_matrix
        self.converted_navpoint_matrix = converted_navpoint_matrix

        self.time_index = time_index
        self.sector_index = sector_index
        self.capacity_time_matrix = capacity_time_matrix
        self.capacity_demand_diff_matrix = capacity_demand_diff_matrix
        self.additional_time_increase = additional_time_increase
        self.fill_value = fill_value
        self.timestep_granularity = timestep_granularity
        self.seed = seed

        self.verbosity = verbosity
        self.iteration = iteration

        self.max_vertices_cutoff_value = max_vertices_cutoff_value
        self.max_delay_parameter = max_delay_parameter
        self.original_max_time = original_max_time
 
    def start(self):

        timestep_granularity = self.timestep_granularity
        problematic_flights = self.problematic_flight_indices
        converted_instance_matrix = self.converted_instance_matrix
        time_index = self.time_index
        sector_index = self.sector_index
        capacity_time_matrix = self.capacity_time_matrix
        capacity_demand_diff_matrix = self.capacity_demand_diff_matrix
        additional_time_increase = self.additional_time_increase
        fill_value = self.fill_value
        networkx_graph = self.networkx_graph
        unit_graphs = self.unit_graphs
        flights_affected = self.problematic_flights

        max_vertices_cutoff_value = self.max_vertices_cutoff_value
        max_delay_parameter = self.max_delay_parameter

        flight_navpoint_instance = []
        flight_times_instance = []
        
        needed_capacities_for_navpoint = {}

        sector_instance = {}
        path_fact_instances = []

        planned_departure_time_instance = []
        actual_departure_time_instance = []
        planned_arrival_time_instance = []
        actual_arrival_time_instance = []

        time_window = max_delay_parameter
        k = max_vertices_cutoff_value

        for flight_affected_index in range(flights_affected.shape[0]):

            flight_index = problematic_flights[flight_affected_index]

            airplane_id = (self.airplane_flight[self.airplane_flight[:,1] == flight_index])[0,0]
            airplane_speed_kts = self.airplanes[airplane_id,1]
            #current_flight = self.flights[self.flights[:,0] == flight_index]

            flight_affected = flights_affected[flight_affected_index,:]

            filed_flight_path = self.filed_flights[self.filed_flights[:,0] == flight_index,:]

            potentially_affected_flights = self.problematic_airplane_flight_map[airplane_id]

            potentially_affected_flights_tmp = []
            for flight_affected_index in potentially_affected_flights:
                if flight_index != flight_affected_index:
                    potentially_affected_flights_tmp.append(flight_affected_index)

            potentially_affected_flights = potentially_affected_flights_tmp



            #actual_arrival_time = (flight_affected >= 0).argmax() - 1
            actual_arrival_time = np.flatnonzero(flight_affected >= 0)[-1] 

            planned_arrival_time = self.planned_arrival_times[flight_index]
            actual_delay = actual_arrival_time - planned_arrival_time

            # TURNAROUND TIME:
            # If a flight needs more than 1 timestep to prepare for departure they coincide; otherwise different
            actual_flight_operations_start_time = np.flatnonzero(flight_affected != fill_value)[0]
            actual_flight_departure_time = np.flatnonzero(flight_affected != fill_value)[1] - 1
            if actual_flight_departure_time < 0:
                raise Exception("planned_flight_departure_time < 0 => Must never happen")

            #actual_flight_departure_time = planned_flight_departure_time + actual_delay
            
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            # WITH A FILED FLIGHT PATH WE GET PATH NUMBER = 0
            paths = self.k_diverse_near_shortest_paths(unit_graphs[airplane_speed_kts], origin, destination, self.nearest_neighbors_lookup[airplane_speed_kts],
                                                k=k, eps=0.1, jaccard_max=0.6, penalty_scale=0.1, max_tries=50, weight_key="weight",
                                                filed_path=list(filed_flight_path[:,1]))
            

            path_number = 0

            planned_departure_time_instance.append(f"actualFlightOperationsStartTime({flight_index},{actual_flight_operations_start_time}).")
            #actual_departure_time_instance.append(f"actualDepartureTime({airplane_id},{start_time}).")
            planned_arrival_time_instance.append(f"plannedArrivalTime({flight_index},{planned_arrival_time}).")

            for path in paths:

                navpoint_trajectory = self.get_flight_navpoint_trajectory(flights_affected, networkx_graph, flight_index, actual_flight_departure_time, airplane_speed_kts, path, timestep_granularity)

                for delay in range(additional_time_increase * time_window,time_window * (additional_time_increase + 1)):

                    flight_time = actual_flight_departure_time - actual_flight_operations_start_time
                    current_time = actual_flight_operations_start_time

                    navaid = navpoint_trajectory[0][1]
                    flight_navpoint_instance.append(f"next_pos({flight_index},{path_number},{navaid},{0},{navaid},{flight_time}).")

                    for flight_hop_index in range(len(navpoint_trajectory)):

                        navaid = navpoint_trajectory[flight_hop_index][1]
                        time = navpoint_trajectory[flight_hop_index][2]
                        #sector = (self.navaid_sector[self.navaid_sector[:,0] == navaid])[0,1]
                        #sector = self.navaid_sector_lookup[navaid]

                        if flight_hop_index == 0:

                            flight_navpoint_instance.append(f"next_pos({flight_index},{path_number},{navaid},{flight_time},{navaid},{flight_time+delay}).")

                            if navaid not in needed_capacities_for_navpoint:
                                needed_capacities_for_navpoint[navaid] = [current_time, current_time+delay]
                            if current_time < needed_capacities_for_navpoint[navaid][0]:
                                needed_capacities_for_navpoint[navaid][0] = current_time
                            if current_time + delay > needed_capacities_for_navpoint[navaid][1]:
                                needed_capacities_for_navpoint[navaid][1] = current_time + delay

                            current_time += delay
                            flight_time += delay                            

                        else:

                            prev_navaid = navpoint_trajectory[flight_hop_index-1][1]
                            prev_time = navpoint_trajectory[flight_hop_index-1][2]
                            #prev_sector = (self.navaid_sector[self.navaid_sector[:,0] == prev_navaid])[0,1]
                            #prev_sector = self.navaid_sector_lookup[prev_navaid]


                            time_delta = time - prev_time
                            flight_navpoint_instance.append(f"next_pos({flight_index},{path_number},{prev_navaid},{flight_time},{navaid},{flight_time+time_delta}).")

                            if navaid not in needed_capacities_for_navpoint:
                                # Approximate time_delta/2 to be on the safe side:
                                needed_capacities_for_navpoint[navaid] = [max(current_time-time_delta,0), current_time+time_delta]

                            if current_time-time_delta < needed_capacities_for_navpoint[navaid][0]:
                                needed_capacities_for_navpoint[navaid][0] = max(current_time-time_delta,0)

                            if current_time + time_delta > needed_capacities_for_navpoint[navaid][1]:
                                needed_capacities_for_navpoint[navaid][1] = current_time + time_delta

                            if current_time-time_delta < needed_capacities_for_navpoint[prev_navaid][0]:
                                # This should not happen, but if it does it is no problem:
                                needed_capacities_for_navpoint[prev_navaid][0] = max(current_time-time_delta,0)
                            if current_time + time_delta > needed_capacities_for_navpoint[prev_navaid][1]:
                                needed_capacities_for_navpoint[prev_navaid][1] = current_time + time_delta
                            

                            current_time += time_delta
                            flight_time += time_delta

                    actual_arrival_time_instance.append(f"actualArrivalTime({flight_index},{current_time},{path_number}).")


                    landing_time_previous_lag = current_time

                    #print(f"LANDING_TIME:{landing_time_previous_lag}")
                    #print(potentially_affected_flights)
                    #print(flight_index)

                    for potentially_affected_flight in potentially_affected_flights:

                            
                        
                        #print(f"Potentially Affected Flight: {potentially_affected_flight}")

                        potentially_actual_flight_operations_start_time = np.flatnonzero(converted_instance_matrix[potentially_affected_flight,:] != fill_value)[0]

                        if potentially_actual_flight_operations_start_time <= landing_time_previous_lag:
                            #print(f"--> IS ACTUALLY AFFECTED: {potentially_affected_flight}")
                            # NEED TO COMPUTE MINIMUM DELAY OF AFFECTED FLIGHTS
                            #minimum_induced_rotary_delay = landing_time_previous_lag - actual_flight_operations_start_time + 1
                            potentially_actual_flight_operations_start_time = landing_time_previous_lag + 1

                        current_time = potentially_actual_flight_operations_start_time


                        planned_departure_time_instance.append(f"actualFlightOperationsStartTime({potentially_affected_flight},{potentially_actual_flight_operations_start_time}).")

                        potentially_planned_arrival_time = self.planned_arrival_times[potentially_affected_flight]

                        planned_arrival_time_instance.append(f"plannedArrivalTime({potentially_affected_flight},{potentially_planned_arrival_time}).")


                        potentially_affected_flight_path_indices = np.flatnonzero(self.converted_navpoint_matrix[potentially_affected_flight,:] != fill_value)

                        #if potentially_affected_flight == 83:
                        #    print("<<<>>>>")
                        #    print(np.flatnonzero(self.converted_navpoint_matrix[potentially_affected_flight,:] != fill_value))
                        #    print(potentially_affected_flight_path_indices)

                        flight_time = 0


                        if len(potentially_affected_flight_path_indices) == 1:
                            hop_index = 0

                            cur_navpoint = self.converted_navpoint_matrix[potentially_affected_flight,potentially_affected_flight_path_indices[hop_index]]


                            flight_navpoint_instance.append(f"single_pos({potentially_affected_flight},{path_number},{cur_navpoint},{flight_time}).")
                            
                            if cur_navpoint not in needed_capacities_for_navpoint:
                                # Approximate time_delta/2 to be on the safe side:
                                needed_capacities_for_navpoint[cur_navpoint] = [max(current_time-1,0), current_time+1]
                            if current_time-1 < needed_capacities_for_navpoint[cur_navpoint][0]:
                                needed_capacities_for_navpoint[cur_navpoint][0] = max(current_time-1,0)
                            if current_time + 1 > needed_capacities_for_navpoint[cur_navpoint][1]:
                                needed_capacities_for_navpoint[cur_navpoint][1] = current_time + 1

                            flight_time = 1


                        for hop_index in range(1,len(potentially_affected_flight_path_indices)):

                            prev_navpoint = self.converted_navpoint_matrix[potentially_affected_flight,potentially_affected_flight_path_indices[hop_index - 1]]
                            cur_navpoint = self.converted_navpoint_matrix[potentially_affected_flight,potentially_affected_flight_path_indices[hop_index]]

                            time_delta = potentially_affected_flight_path_indices[hop_index] - potentially_affected_flight_path_indices[hop_index - 1]
                            current_time += time_delta

                            flight_navpoint_instance.append(f"next_pos({potentially_affected_flight},{path_number},{prev_navpoint},{flight_time},{cur_navpoint},{flight_time+time_delta}).")
                            
                            flight_time += time_delta

                            if cur_navpoint not in needed_capacities_for_navpoint:
                                # Approximate time_delta/2 to be on the safe side:
                                needed_capacities_for_navpoint[cur_navpoint] = [max(current_time-time_delta,0), current_time+time_delta]
                            if current_time-time_delta < needed_capacities_for_navpoint[cur_navpoint][0]:
                                needed_capacities_for_navpoint[cur_navpoint][0] = max(current_time-time_delta,0)
                            if current_time + time_delta > needed_capacities_for_navpoint[cur_navpoint][1]:
                                needed_capacities_for_navpoint[cur_navpoint][1] = current_time + time_delta

                            if prev_navpoint not in needed_capacities_for_navpoint:
                                # Approximate time_delta/2 to be on the safe side:
                                needed_capacities_for_navpoint[prev_navpoint] = [max(current_time-time_delta,0), current_time+time_delta]
                            if current_time-time_delta < needed_capacities_for_navpoint[prev_navpoint][0]:
                                needed_capacities_for_navpoint[prev_navpoint][0] = max(current_time-time_delta,0)
                            if current_time + time_delta > needed_capacities_for_navpoint[prev_navpoint][1]:
                                needed_capacities_for_navpoint[prev_navpoint][1] = current_time + time_delta
                            


                        current_time = potentially_actual_flight_operations_start_time + flight_time
                        landing_time_previous_lag = current_time

                        actual_arrival_time_instance.append(f"actualArrivalTime({potentially_affected_flight},{current_time},{path_number}).")
                        path_fact_instances.append(f"chosen_path({potentially_affected_flight},{path_number}) :- chosen_path({flight_index},{path_number}).")

                    # path_numbers = #PATHS * #DELAYS
                    path_number += 1

            path_fact_instances.append(f"paths({flight_index},0..{path_number - 1}).")


        # 1.) Get config for sector sector_index and time_index
        #   a.) Check out what we can do with this sector
        #   b.) Create alternatives!

        number_configs = 7
        config_restore_dict = {}

        sector_config_instance = []
        sector_capacity_instance = []
        navpoint_sector_assignment_instance = []

        # CONFIG = 0
        current_config = 0
        sector_config_instance.append(f"config({current_config}).")

        for navpoint in needed_capacities_for_navpoint.keys():
            from_time = needed_capacities_for_navpoint[navpoint][0]
            until_time = needed_capacities_for_navpoint[navpoint][1]

            for current_time in range(from_time, until_time + 1):
                #
                current_sector = self.navaid_sector_time_assignment[navpoint, current_time]
                navpoint_sector_assignment_instance.append(f"possible_assignment({navpoint},{current_sector},{current_time},{current_config}).")

                current_capacity = self.capacity_demand_diff_matrix[current_sector, current_time]
                sector_capacity_instance.append(f"possible_sector_capacity({current_sector},{current_capacity},{current_time},{current_config}).")
        # END CONFIG = 0
        current_config += 1

        if False:

            navpoints_in_sector = np.nonzero(self.navaid_sector_time_assignment[:,time_index] == sector_index)[0]
            navpoints_in_sector = list(set(navpoints_in_sector) & set(self.networkx_graph))   # keep only nodes that are actually in G
            neighbors = set(node_boundary(self.networkx_graph, navpoints_in_sector))  

            if len(navpoints_in_sector) == 0:
                print(time_index)
                print(sector_index)
                print("FOUND 0 NAVPOINTS IN SECTOR -> SHOULD NEVER HAPPEN")
                quit()
            else:
                print(time_index)
                print(sector_index)

            total_partitions = int(bell(len(navpoints_in_sector)))  # all set partitions
            # exclude trivial one: {whole set}
            nontrivial_count = max(total_partitions - 1, 0)

            demand_matrix = self.capacity_time_matrix - self.capacity_demand_diff_matrix
            #self.airport_vertices
            #self.capacity
            #self.navaid_sector_time_assignment

            if nontrivial_count > 0:

                composition_navpoints = navpoints_in_sector
                composition_sectors = self.navaid_sector_time_assignment[composition_navpoints, time_index]

                original_demand = demand_matrix[composition_sectors,:].copy()
                original_capacity = capacity_time_matrix[composition_sectors,:].copy()
                original_composition = self.navaid_sector_time_assignment[composition_navpoints,:].copy()

                number_partitions = (number_configs-1) / 2
                number_partitions = min(number_partitions, nontrivial_count)

                number_compositions = (number_configs - 1) - number_partitions

                parts = self.first_l_nontrivial_partitions(navpoints_in_sector, number_partitions)

                for partition in parts:

                    
                    capacity_time_matrix[composition_sectors,time_index:] = 0
                    partition_sectors = []

                    for partition_navpoints in partition:

                        if sector_index in partition_navpoints:
                            cur_sector_index = sector_index
                        else:
                            cur_sector_index = partition_navpoints[0]

                        partition_sectors.append(cur_sector_index)

                        partition_navpoints = np.array(partition_navpoints)

                        self.navaid_sector_time_assignment[partition_navpoints, time_index :] = cur_sector_index
                        tmp_navaid_sector_time_assignment = np.zeros((len(partition_navpoints),self.navaid_sector_time_assignment.shape[1]))


                        tmp_atomic_capacities = []
                        index = 0
                        for navaid in partition_navpoints:
                            tmp_atomic_capacities.append([index,int(self.capacity[navaid,1])])
                            index += 1

                        tmp_atomic_capacities = np.array(tmp_atomic_capacities)
                        composite_capacity_time_matrix = OptimizeFlights.capacity_time_matrix(tmp_atomic_capacities, self.navaid_sector_time_assignment.shape[1], self.timestep_granularity, tmp_navaid_sector_time_assignment, z = self.sector_capacity_factor)
                        capacity_time_matrix[cur_sector_index,time_index:] = composite_capacity_time_matrix[0,time_index:]

                    # All flights that pass through any navpoint in the composite sector after time_index
                    tmp_flights = np.nonzero(np.isin(self.converted_navpoint_matrix[:,time_index:], navpoints_in_sector))[0]
                    triplets = self.time_matrix_to_triplets(self.converted_navpoint_matrix[tmp_flights,time_index:])

                    # FIX INDICES
                    triplets[:,2] = triplets[:,2] + time_index

                    """
                    triplets_indices = []
                    tmp_flights_index = 0
                    for triplets_index in range(1,triplets.shape[0]):

                        if triplets[triplets_index,0] != triplets[triplets_index-1,0]:
                            print(f"{triplets[triplets_index,0]} != {triplets[triplets_index-1,0]}")
                            tmp_flights_index += 1

                        if triplets_index == 1:

                            triplets_indices.append(tmp_flights[0])
                        triplets_indices.append(tmp_flights[tmp_flights_index])

                    triplets[:,0] = np.array(triplets_indices)                    
                    """

                    #tmp_airplane_flight = self.airplane_flight[np.isin(self.airplane_flight[:,1],tmp_flights)]

                    airplane_flight_mockup = []
                    for flight_index in range(len(tmp_flights)):
                        airplane_flight_mockup.append([flight_index,flight_index])

                    airplane_flight_mockup = np.array(airplane_flight_mockup)

                    converted_instance_matrix, _ = OptimizeFlights.instance_to_matrix_vectorized(triplets, airplane_flight_mockup, self.navaid_sector_time_assignment.shape[1], self.timestep_granularity, self.navaid_sector_time_assignment)


                    system_loads_tmp = OptimizeFlights.bucket_histogram(converted_instance_matrix, None, self.capacity_time_matrix.shape[0], converted_instance_matrix.shape[1], self.timestep_granularity)

                    partition_sectors = np.array(partition_sectors)

                    demand_matrix[partition_sectors,time_index:] = system_loads_tmp[partition_sectors,time_index:]

                    capacity_demand_diff_matrix = capacity_time_matrix - demand_matrix

                    # COMPOSITION CONFIG 
                    sector_config_instance.append(f"config({current_config}).")

                    for navpoint in needed_capacities_for_navpoint.keys():
                        from_time = needed_capacities_for_navpoint[navpoint][0]
                        until_time = needed_capacities_for_navpoint[navpoint][1]

                        for current_time in range(from_time, until_time + 1):
                            #
                            current_sector = self.navaid_sector_time_assignment[navpoint, current_time]
                            navpoint_sector_assignment_instance.append(f"possible_assignment({navpoint},{current_sector},{current_time},{current_config}).")

                            current_capacity = capacity_demand_diff_matrix[current_sector, current_time]
                            sector_capacity_instance.append(f"possible_sector_capacity({current_sector},{current_capacity},{current_time},{current_config}).")
                    # COMPOSITION CONFIG

                    config_restore_dict[current_config] = {}
                    config_restore_dict[current_config]["composition_navpoints"] = composition_navpoints.copy()
                    config_restore_dict[current_config]["composition_sectors"] = composition_sectors.copy()
                    config_restore_dict[current_config]["demand"] = demand_matrix[composition_sectors,:].copy()
                    config_restore_dict[current_config]["capacity"] = capacity_time_matrix[composition_sectors,:].copy()
                    config_restore_dict[current_config]["composition"] = self.navaid_sector_time_assignment[composition_navpoints,:].copy()
                    config_restore_dict[current_config]["time_index"] = time_index
                    config_restore_dict[current_config]["sector_index"] = sector_index

                    # RESTORE ORIGINAL CONFIG:
                    demand_matrix[composition_sectors,:] = original_demand
                    capacity_time_matrix[composition_sectors,:] = original_capacity
                    self.navaid_sector_time_assignment[composition_navpoints, :] = original_composition

                    current_config += 1


            else:
                number_compositions = number_configs


            composition_number = 0

            for neighbor in neighbors:

                if neighbor in self.airport_vertices:
                    # No Composition with Airports
                    continue

                composition_navpoints = [neighbor] + navpoints_in_sector
                composition_sectors = self.navaid_sector_time_assignment[composition_navpoints, time_index]

                original_demand = demand_matrix[composition_sectors,:].copy()
                original_capacity = capacity_time_matrix[composition_sectors,:].copy()
                original_composition = self.navaid_sector_time_assignment[composition_navpoints,:].copy()
                #original_sector_navpoints = navpoints_in_sector


                self.navaid_sector_time_assignment[composition_navpoints, time_index :] = sector_index
                tmp_navaid_sector_time_assignment = np.zeros((len(composition_navpoints),self.navaid_sector_time_assignment.shape[1]))

                tmp_atomic_capacities = []
                index = 0
                for navaid in composition_navpoints:
                    tmp_atomic_capacities.append([index,int(self.capacity[navaid,1])])
                    index += 1

                tmp_atomic_capacities = np.array(tmp_atomic_capacities)

                composite_capacity_time_matrix = OptimizeFlights.capacity_time_matrix(tmp_atomic_capacities, self.navaid_sector_time_assignment.shape[1], self.timestep_granularity, tmp_navaid_sector_time_assignment, z = self.sector_capacity_factor)

                capacity_time_matrix[composition_sectors,time_index:] = 0
                capacity_time_matrix[sector_index,time_index:] = composite_capacity_time_matrix[0,time_index:]

                aggregated_demand = demand_matrix[composition_sectors, time_index:].sum(axis=0)
                demand_matrix[composition_sectors, time_index:] = 0
                demand_matrix[sector_index, time_index:] = aggregated_demand

                capacity_demand_diff_matrix = capacity_time_matrix - demand_matrix

                # COMPOSITION CONFIG 
                sector_config_instance.append(f"config({current_config}).")

                for navpoint in needed_capacities_for_navpoint.keys():
                    from_time = needed_capacities_for_navpoint[navpoint][0]
                    until_time = needed_capacities_for_navpoint[navpoint][1]

                    for current_time in range(from_time, until_time + 1):
                        #
                        current_sector = self.navaid_sector_time_assignment[navpoint, current_time]
                        navpoint_sector_assignment_instance.append(f"possible_assignment({navpoint},{current_sector},{current_time},{current_config}).")

                        current_capacity = capacity_demand_diff_matrix[current_sector, current_time]
                        sector_capacity_instance.append(f"possible_sector_capacity({current_sector},{current_capacity},{current_time},{current_config}).")
                # COMPOSITION CONFIG

                config_restore_dict[current_config] = {}
                config_restore_dict[current_config]["composition_navpoints"] = composition_navpoints.copy()
                config_restore_dict[current_config]["composition_sectors"] = composition_sectors.copy()
                config_restore_dict[current_config]["demand"] = demand_matrix[composition_sectors,:].copy()
                config_restore_dict[current_config]["capacity"] = capacity_time_matrix[composition_sectors,:].copy()
                config_restore_dict[current_config]["composition"] = self.navaid_sector_time_assignment[composition_navpoints,:].copy()
                config_restore_dict[current_config]["time_index"] = time_index
                config_restore_dict[current_config]["sector_index"] = sector_index

                # RESTORE ORIGINAL CONFIG:
                demand_matrix[composition_sectors,:] = original_demand
                capacity_time_matrix[composition_sectors,:] = original_capacity
                self.navaid_sector_time_assignment[composition_navpoints, :] = original_composition

                composition_number += 1
                current_config += 1
                if composition_number >= number_compositions:
                    # MORE THAN MAX COMPOSITIONS!
                    break


        # -----------------------------------------------------------

        planned_arrival_time_instance = list(set(planned_arrival_time_instance))

        flight_navpoint_instance = "\n".join(flight_navpoint_instance)
        flight_times_instance = "\n".join(flight_times_instance)

        path_fact_instances = "\n".join(path_fact_instances)

        flight_plan_instance = self.flight_plan_strings(problematic_flights, flights_affected)
        flight_plan_instance = "\n".join(flight_plan_instance)


        sector_config_instance = "\n".join(sector_config_instance)
        sector_capacity_instance = "\n".join(sector_capacity_instance)
        navpoint_sector_assignment_instance = "\n".join(navpoint_sector_assignment_instance)


        planned_departure_time_instance = "\n".join(planned_departure_time_instance)
        actual_departure_time_instance = "\n".join(actual_departure_time_instance)
        planned_arrival_time_instance = "\n".join(planned_arrival_time_instance)
        actual_arrival_time_instance = "\n".join(actual_arrival_time_instance)

        instance = f"""
{flight_plan_instance}
{flight_navpoint_instance}
{path_fact_instances}
{planned_departure_time_instance}
{actual_departure_time_instance}
{planned_arrival_time_instance}
{actual_arrival_time_instance}
{sector_config_instance}
{sector_capacity_instance}
{navpoint_sector_assignment_instance}
        """

        #if time_index == 727 and sector_index == 6 and additional_time_increase > 0:
        #    quit()

        encoding = self.encoding
        
        if self.verbosity > 0:
            open(f"20251009_test_instance_{additional_time_increase}.lp","w").write(instance)
            #if len(navpoints_in_sector) > 1:
            #    quit()
            #quit()

        solver: Model = Solver(encoding, instance)
        model = solver.solve()

        return model, config_restore_dict
    
    def first_l_nontrivial_partitions(self, items, l):
        l = int(l)
        xs = list(items)
        n = len(xs)

        # yields partitions of sizes 2..n-1, skipping {all} and {singletons}
        it = (tuple(map(tuple, p)) for p in set_partitions(xs) if 1 < len(p) <= n)
        return list(islice(it, l))



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
                distance = networkx_graph[prev_vertex][vertex]["weight"]

                # CONVERT SPEED TO m/s
                airplane_speed_ms = airplane_speed_kts * 0.51444

                # Compute duration from prev to vertex in unit time:
                duration_in_seconds = distance/airplane_speed_ms
                factor_to_unit_standard = 3600.00 / float(timestep_granularity)
                duration_in_unit_standards = math.ceil(duration_in_seconds / factor_to_unit_standard)

                current_time = current_time + duration_in_unit_standards

                t_slot=current_time

                if t_slot >= flights_affected.shape[1]:
                    raise Exception("In optimize_flights max time exceeded current allowed time.")

            traj.append((flight_index, vertex, t_slot))

        return traj
    
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
            edge_sets.append(E)

        return paths


    def handle_sectors_instance_generation(self, capacity_demand_diff_matrix, additional_time_increase, max_delay_parameter, start_time, delay, from_origin_time, vertex_ids, flight_index, time_window):

        timed_capacities = []

        for additional_time in range(-time_window,time_window * (additional_time_increase + max_delay_parameter + delay)):
            current_time = start_time + additional_time + from_origin_time
            if current_time >= self.timestep_granularity * (self.original_max_time + additional_time_increase):
                break
            elif current_time < 0:
                continue

            if current_time >= capacity_demand_diff_matrix.shape[1]:
                current_time = capacity_demand_diff_matrix.shape[1] - 1

            sector_times = [f"sector({vertex_id},{current_time},{capacity_demand_diff_matrix[vertex_id,current_time]})." for vertex_id in vertex_ids]

            if from_origin_time > 0:
                if additional_time < -self.timestep_granularity+self.timestep_granularity * additional_time_increase:
                    sector_times = [f":- flight({flight_index},{current_time},{vertex_id})." for vertex_id in vertex_ids]

            timed_capacities += sector_times
        return timed_capacities

    def handle_origin_sector_instance_generation(self, capacity_demand_diff_matrix, additional_time_increase, max_delay_parameter, start_time, origin, delay, time_window):

        timed_capacities = []

        for additional_time in range(-time_window,time_window * (additional_time_increase + max_delay_parameter) + delay):
            current_time = start_time + additional_time
            if current_time >= self.timestep_granularity * (self.original_max_time + additional_time_increase):
                break
            elif current_time < 0:
                continue

            if current_time >= capacity_demand_diff_matrix.shape[1]:
                current_time = capacity_demand_diff_matrix.shape[1] - 1

            sector_time = f"sector({origin},{current_time},{capacity_demand_diff_matrix[origin,current_time]})."
            timed_capacities.append(sector_time)

            if current_time == capacity_demand_diff_matrix.shape[1] - 1:
                break

        return timed_capacities

    def create_filed_flight_plan_atoms(self, flight_sector_instance, flight_times_instance, sector_instance, graph_instance, flight_index, flight_affected, capacity_demand_diff_matrix, start_time, default_filed_path_number = 0):

        for tmp_index in range(0,flight_affected.shape[0]):

            current_sector = flight_affected[tmp_index]

            flight_sector_instance.append(f"sectorFlight({flight_index},{tmp_index},{current_sector},{default_filed_path_number}).")
            flight_times_instance.append(f"flightTime({flight_index},{tmp_index}).")

            current_time = tmp_index + start_time

            sector_instance.append(f"sector({current_sector},{current_time},{capacity_demand_diff_matrix[current_sector,current_time]}).")

            if tmp_index > 0:
                graph_instance.append(f"sectorEdge({flight_affected[tmp_index - 1]},{current_sector}).")
            else:
                graph_instance.append(f"sectorEdge({current_sector},{current_sector}).")


        return flight_sector_instance, flight_times_instance, sector_instance, graph_instance
    
    def restrict_max_vertices(self, prev_vertices, vertex_ids, matching_vertices, flight_affected, from_origin_time, delay):

        max_vertices_cutoff_value = self.max_vertices_cutoff_value
        edge_distances = self.edge_distances
        # ------------------------------------------------
        # START PATH STUFF:

        if len(vertex_ids) > max_vertices_cutoff_value:
            # GO INTO PATH MODE:
            if prev_vertices is None:
                # FALLBACK
                rng = np.random.default_rng(seed = self.seed)
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

                    rng = np.random.default_rng(seed = self.seed)
                    used_vertex_id = rng.choice(path_vertices_ids, size=1, replace=False)
                    used_vertices.append(int(used_vertex_id))

                flightPathVertex = int(flight_affected[from_origin_time + delay])
                if flightPathVertex not in used_vertices:
                    if len(used_vertices) >= max_vertices_cutoff_value:
                        del used_vertices[-1]
                    used_vertices.append(int(flightPathVertex))

                vertex_ids = used_vertices

        else:
            pass
        # -------------------------------------------------------

        return vertex_ids
        


    def flight_plan_strings(self, rows_original_indices, mat: np.ndarray,
                        *,
                        fill_value: int = -1) -> np.ndarray:
        """
        Build a 1-D array of strings ``"instance(r,c,v)."``
        for every cell whose value ≥ 0 (i.e., `!= fill_value`).

        Parameters
        ----------
        mat
            Your (possibly sliced) `converted_instance_matrix`
            – shape = (n_rows, n_cols)
        fill_value
            Placeholder that marks “empty” entries (default −1)

        Returns
        -------
        out : np.ndarray
            1-D array (dtype=object) where each element is an
            `"instance(r,c,v)."` string.
        """
        # ------------------------------------------------------------------
        # 1.  Locate valid positions
        # ------------------------------------------------------------------
        mask = mat != fill_value                # Boolean matrix
        if not np.any(mask):
            return np.empty(0, dtype=object)    # nothing to report

        r, c = np.nonzero(mask)                 # row/col indices (1-D)

        rows_original_indices = np.asarray(rows_original_indices)
        r = rows_original_indices[r]
        v = mat[mask]                           # corresponding values

        # ------------------------------------------------------------------
        # 2.  Element-wise concatenation (pure NumPy, no Python loop)
        # ------------------------------------------------------------------
        s = np.char.add('flightPlan(', r.astype(str))
        s = np.char.add(np.char.add(s, ','), c.astype(str))
        s = np.char.add(np.char.add(s, ','), v.astype(str))
        s = np.char.add(s, ').')                # final dot

        return s            # shape (K,)  —  K = number of non-empty cells

    def system_loads_restricted(self, mat: np.ndarray,
                                num_buckets: int,
                                *,
                                time_idx: int,
                                bucket_idx: int,
                                fill_value: int = -1) -> np.ndarray:
        """
        Build a |B| × |T| bucket-occupancy matrix *excluding*:

        • every element sitting in `bucket_idx` at `time_idx`
        • every element whose first scheduled time step is > `time_idx`

        Parameters
        ----------
        mat
            converted_instance_matrix  (shape = |F| × |T|)
        num_buckets
            Equals `self.capacity.shape[0]`
        time_idx
            The reference time step
        bucket_idx
            The reference bucket
        fill_value
            Placeholder that means “empty slot” (default –1)

        Returns
        -------
        loads : np.ndarray
            Bucket-by-time step counts with the requested rows removed.
        """
        n_elems, n_times = mat.shape

        # ---------------------------------------------------------------
        # 1.  Identify *all* rows to drop
        # ---------------------------------------------------------------
        #
        # 1a. Elements located in (time_idx, bucket_idx)
        offending_mask = mat[:, time_idx] == bucket_idx       # shape (|F|,)

        # 1b. Elements that *start* after `time_idx`
        #
        #     First non-fill_value column for each row.
        valid_mask   = mat != fill_value                      # True where real data
        has_any      = valid_mask.any(axis=1)                 # rows that are not empty
        # np.where is ternary operator (np.where(condition,x,y)): result[i] = x[i] if condition[i] else y[i]
        first_nonneg = np.where(has_any,
                                valid_mask.argmax(axis=1),    # first True along row
                                n_times)                      # unused rows ⇒ > time_idx

        late_start_mask = first_nonneg > time_idx

        # Consolidate the exclusion rule
        keep_mask = ~(offending_mask | late_start_mask)       # rows we *keep*

        # Quick exit if nothing survives
        if not np.any(keep_mask):
            return np.zeros((num_buckets, n_times), dtype=int)

        mat_keep = mat[keep_mask]

        # ---------------------------------------------------------------
        # 2.  Re-use the earlier bucket-histogram
        # ---------------------------------------------------------------
        return OptimizeFlights.bucket_histogram(mat_keep,
                                num_buckets,
                                mat_keep.shape[1],
                                self.timestep_granularity,
                                fill_value=fill_value)
    
    def sector_string_matrix(self, values: np.ndarray) -> np.ndarray:
        """
        Return a `dtype=object` array of the same shape as *values*
        whose entries are the strings ``"bucket(r,c,v)"``.

        Parameters
        ----------
        values : np.ndarray
            Your numeric matrix (e.g. `capacity_demand_diff_matrix_restricted`).

        Notes
        -----
        *Pure* NumPy — no Python‐level loops — by chaining `np.char.add`.
        """
        rows, cols = np.indices(values.shape)       # same shape as `values`

        # Flatten once to keep the broadcasting simple
        r, c, v = rows.ravel(), cols.ravel(), values.ravel()

        s = np.char.add('sector(', r.astype(str))
        s = np.char.add(np.char.add(s, ','), c.astype(str))
        s = np.char.add(np.char.add(s, ','), v.astype(str))
        s = np.char.add(s, ').')

        #return s.reshape(values.shape)              # back to 2-D
        return s


    def not_used_airports_removal_from_instance(self, fill_value, flights_affected, timed_capacities,capacity_demand_diff_matrix_restricted, considered_vertices, all_edges):

        affected_flights_1D = flights_affected[flights_affected != fill_value] 
        affected_flights_unique = np.unique(affected_flights_1D)                        
        remaining_airport_vertices = np.setdiff1d(self.airport_vertices, affected_flights_unique, assume_unique=True)

        all_vertices = set(np.unique(all_edges))

        to_delete_vertices = all_vertices.difference(set(considered_vertices))
        to_delete_vertices = to_delete_vertices.union(set(remaining_airport_vertices))
        
        to_delete_vertices = np.array(list(to_delete_vertices))

        #timed_capacities = timed_capacities.reshape(capacity_demand_diff_matrix_restricted.shape)
        #timed_capacities = np.delete(timed_capacities, to_delete_vertices, axis=0)
        #timed_capacities = timed_capacities.flatten()
        
        remaining_airport_vertices_lookup_table = {}
        for airport_vertex in remaining_airport_vertices:
            remaining_airport_vertices_lookup_table[str(airport_vertex)] = False

        edges_instance = []
        for row_index in range(self.graph.shape[0]):
            row = self.graph[row_index,: ]

            if str(row[0]) in remaining_airport_vertices_lookup_table or str(row[1])  in remaining_airport_vertices_lookup_table:
                continue

            if row[0] not in considered_vertices or row[1] not in considered_vertices:
                continue

            edges_instance.append(f"sectorEdge({row[0]},{row[1]}).")

        edges_instance = "\n".join(edges_instance)

        airport_instance = []
        for row_index in range(self.airport_vertices.shape[0]):
            row = self.airport_vertices[row_index]

            if str(row) in remaining_airport_vertices_lookup_table:
                continue

            airport_instance.append(f"airport({row}).")

        airport_instance = "\n".join(airport_instance)
        return edges_instance,airport_instance,timed_capacities

        # 1.) Create matrix for capacities
        # 2.) Subtract: capacities-system_loads
        # 3.) If anything below 0 -> Capacity problem!
        # 4.) Then iterate over time t:
        #   a.) If at time t, there is a problem with sector x
        #   b.) Consider all flights that have already landed, prior (and including?) to t, as fixed
        #   c.) Ignore all flights that have not been started yet (starting time > t) (STRICTLY GREATER!)
        #   d.) From this, create an ASP instance with a handful of flights that can be re-scheduled accordingly

        #print(system_loads)
        #print(system_loads.shape)
        #np.savetxt("test.csv", converted_instance_matrix,delimiter=",",fmt="%i")
        #np.savetxt("test_loads.csv", capacity_demand_diff_matrix,delimiter=",",fmt="%i")

    def time_matrix_to_triplets(self, M, *, row_flights=None,
                                fill_value=-1, t0=0, sort=True):
        """
        M:               (|F| x |T|) matrix with navpoint IDs or fill_value
        interesting_flights: list/array of flight IDs to extract
        row_flights:     length-|F| array mapping row index -> flight_id.
                        If None, assumes row i corresponds to flight_id i.
        fill_value:      value used for 'no navpoint' in M
        t0:              time offset (add to column index to reconstruct original time)
        sort:            sort output by (flight_id, time)

        Returns: (N x 3) int array of [flight_id, navpoint_id, time]
        """
        M = np.asarray(M)
        if row_flights is None:
            row_flights = np.arange(M.shape[0], dtype=int)
        row_flights = np.asarray(row_flights)

        """
        interesting_flights = np.asarray(interesting_flights)
        row_mask = np.isin(row_flights, interesting_flights)
        rows_idx = np.nonzero(row_mask)[0]
        if rows_idx.size == 0:
            return np.empty((0, 3), dtype=int)
        """

        sub = M
        r_local, t = np.where(sub != fill_value)
        nvals = sub[r_local, t]
        fids = row_flights[:][r_local]
        times = t + t0

        triplets = np.column_stack((fids, nvals, times)).astype(int)
        if sort and triplets.size:
            order = np.lexsort((triplets[:, 2], triplets[:, 0]))  # sort by (flight, time)
            triplets = triplets[order]
        return triplets


    @classmethod
    def bucket_histogram_reference(cls, instance_matrix: np.ndarray,
                        sectors: np.ndarray,
                        num_buckets: int,
                        n_times: int,
                        timestep_granularity: int,
                        *,
                        fill_value: int = -1) -> np.ndarray:

        valid_mask = instance_matrix != fill_value

        bucket_histogram = np.zeros((num_buckets, n_times), dtype=int)
        if not np.any(valid_mask):                       
            # Shortcut if all cells empty
            return bucket_histogram
        
        for flight_id in range(instance_matrix.shape[0]):


            for time in range(instance_matrix.shape[1]):

                if instance_matrix[flight_id, time] != fill_value:

                    sector = instance_matrix[flight_id, time]

                    if time == 0:
                        bucket_histogram[sector, time] += 1
                    else:

                        prev_sector = instance_matrix[flight_id, time-1]

                        if prev_sector != sector:
                            bucket_histogram[sector, time] += 1

        # ONLY FOR DEBUGGING:
        #np.savetxt("20250819_bucket_histogram_hist.csv", bucket_histogram, delimiter=",",fmt="%i")

        return bucket_histogram
    
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
    

    # ---------------------------------------------------------
    # Vectorized, composite-aware capacity time matrix
    # ---------------------------------------------------------
    @classmethod
    def capacity_time_matrix(cls,
                             cap: np.ndarray,
                             n_times: int,
                             time_granularity: int,
                             navaid_sector_time_assignment: np.ndarray,
                             z = 1) -> np.ndarray:
        """
        cap: shape (N, >=2), atomic capacities in column 1 (per *block* of length T).
        n_times: total number of time slots.
        time_granularity (T): slots per block.
        navaid_sector_time_assignment: (N, n_times), entry [nav, t] = sector-id (0..N-1)
                                       that nav belongs to at time t (composite sectors allowed).

        Returns:
            sector_cap: (N, n_times) with capacity for sector-id row at each time.
                        Semantics match the reference implementation.
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

        # ---- 1) Sum atomic capacities into current sector ids per time (composite-aware)
        atomic_block_cap = np.asarray(cap[:, 1], dtype=np.int64)  # length N

        # We want: total_atomic_sum[sector, t] = sum_{nav | S[nav,t]==sector} atomic_block_cap[nav]
        total_atomic_sum = np.zeros((N, n_times), dtype=np.int64)

        # Scatter-add in one go:
        # indices over flattened (nav, t) grid
        S_flat = S.ravel(order="C")
        t_idx_flat = np.tile(np.arange(n_times, dtype=np.int64), N)
        w_flat = np.repeat(atomic_block_cap, n_times)

        # Sum of atomic per-block caps per (sector,time)
        total_atomic_sum = np.zeros((N, n_times), dtype=np.int64)
        np.add.at(total_atomic_sum, (S_flat, t_idx_flat), np.repeat(cap[:, 1].astype(np.int64), n_times))

        # Count of contributors per (sector,time)
        contrib_count = np.zeros((N, n_times), dtype=np.int64)
        np.add.at(contrib_count, (S_flat, t_idx_flat), 1)


        """
        avg = np.divide(
            total_atomic_sum.astype(np.float64),
            np.maximum(1, contrib_count),  # avoid division by zero
            where=contrib_count > 0
        )
        """
        avg = total_atomic_sum.astype(np.float64) / np.maximum(1, contrib_count)


        # piecewise: if z < k -> ((z+1)/2)*avg  else -> avg * triangular_weight_sum(k, denom)
        denom = z

        tri = cls._triangular_weight_sum_counts(contrib_count, denom)  # float64 matrix

        total_capacity = np.where(
            contrib_count > z,
            np.rint(((z + 1.0) / 2.0) * avg),          # z < k
            np.rint(avg * tri)                          # z >= k
        ).astype(np.int64)


        """
        np.add.at(total_atomic_sum, (S_flat, t_idx_flat), w_flat)

        # ---- 2) Apply composite capacity rule (linear for now): total_capacity = int(sum * z)
        # Grab z from self if available; default 1.0
        z = float(getattr(self, "z", 1.0))
        # Vectorized equivalent to compute_sector_capacity(..., z) in the linear case:
        total_capacity = (total_atomic_sum.astype(np.float64) * z).astype(np.int64)
        """

        # ---- 3) Distribute capacity across T slots using your exact remainder scheme
        base = total_capacity // T          # (N, n_times)
        rem  = total_capacity - base * T    # (N, n_times), 0..T-1

        # Precompute remainder-hit counts table once per call
        rem_table = cls._remainder_distribution_table(T)  # (T+1, T)

        # For each time column, we need index k = t % T
        k_mod = np.arange(n_times, dtype=np.int64) % T     # (n_times,)

        # Lookup “extra” increments: duplicates are honored via counts
        # shapes: rem -> (N, n_times), k_mod[None, :] -> (1, n_times)  ==> (N, n_times)
        extra = rem_table[rem, k_mod[None, :]]

        sector_cap = (base + extra).astype(np.int64, copy=False)

        # Optional debug dump
        #np.savetxt("20251004_cap_mat_fast.csv", sector_cap, delimiter=",", fmt="%i")

        return sector_cap
  
    # ---------------------------------------------------------
    # Fast remainder distribution lookup (matches reference)
    # ---------------------------------------------------------
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
    def compute_sector_capacity(cls, atomic_capacities, z: float) -> int:
        """
        atomic_capacities: 1D iterable of per-slot atomic capacities contributing to the composite
        z: scalar multiplier (can be tuned/changed later)
        returns int capacity for the composite sector at this time slot
        """


        # For now: linear combination
        #return int(np.rint(np.sum(atomic_capacities) * float(z)))
    

        if len(atomic_capacities) == 0:
            return 0

        avg_cap = sum(atomic_capacities) / len(atomic_capacities)

        if z < len(atomic_capacities):
            return ((z+1)*avg_cap) / 2
        else: # z >= len(atomic_capacities)
            sum_ = 0
            for i in range(len(atomic_capacities)):
                sum_ += (1 - (i / z)) * avg_cap

        return math.ceil(sum_)

    @classmethod 
    def instance_to_matrix_vectorized(cls,
                                    flights: np.ndarray,
                                    airplane_flight: np.ndarray,
                                    max_time: int,
                                    time_granularity: int,
                                    navaid_sector_time_assignment: np.ndarray,
                                    *,
                                    fill_value: int = -1,
                                    compress: bool = False):
        """
        Vectorized/semi-vectorized rewrite.
        Dynamic sector assignment via navaid_sector_time_assignment (shape N x T):
        sector_at_event = navaid_sector_time_assignment[navaid_id, time]
        Assumes IDs are non-negative ints (reasonably dense).
        """

        # --- ensure integer views without copies where possible
        flights = flights.astype(np.int64, copy=False)
        airplane_flight = airplane_flight.astype(np.int64, copy=False)

        # --- build flight -> airplane mapping (array is fastest if IDs are dense)
        fid_map_max = int(max(flights[:, 0].max(), airplane_flight[:, 1].max()))
        flight_to_airplane = np.full(fid_map_max + 1, -1, dtype=np.int64)
        flight_to_airplane[airplane_flight[:, 1]] = airplane_flight[:, 0]

        # --- sort by flight, then time (stable contiguous blocks per flight)
        order = np.lexsort((flights[:, 2], flights[:, 0]))
        f_sorted = flights[order]

        fid = f_sorted[:, 0]
        nav = f_sorted[:, 1]
        t   = f_sorted[:, 2]

        # --- dynamic navaid -> sector from (N x T) matrix using pairwise advanced indexing
        N, T = navaid_sector_time_assignment.shape
        if nav.size:
            nav_min, nav_max = int(nav.min()), int(nav.max())
            t_min, t_max     = int(t.min()),   int(t.max())
            if nav_min < 0 or nav_max >= N:
                raise ValueError(
                    f"Navpoint id out of bounds: got range [{nav_min}, {nav_max}], matrix has N={N}."
                )
            if t_min < 0 or t_max >= T:
                raise ValueError(
                    f"Time index out of bounds: got range [{t_min}, {t_max}], matrix has T={T}."
                )

        sec = navaid_sector_time_assignment[nav, t]  # 1D array aligned with f_sorted

        # --- output matrix shape (airplane_id rows, time columns)
        n_rows = int(airplane_flight[:, 1].max()) + 1
        out = np.full((n_rows, int(max_time)), fill_value, dtype=sec.dtype if sec.size else np.int64)

        # --- group boundaries per flight (contiguous in sorted array)
        if fid.size == 0:
            return out, {}

        u, idx_first, counts = np.unique(fid, return_index=True, return_counts=True)

        # planned arrival times = last time per group (vectorized)
        last_idx = idx_first + counts - 1
        planned_arrival_times = dict(zip(u.tolist(), t[last_idx].tolist()))

        # --- fill per-flight via slices (no per-timestep inner loops)
        for g, start in enumerate(idx_first):
            end = start + counts[g]

            flight_index = g

            if flight_index < 0:
                # flight has no airplane mapping; skip defensively
                continue

            times = t[start:end]
            secs  = sec[start:end]
            if times.size == 0:
                continue

            # set the exact event time
            out[flight_index, times[0]] = secs[0]

            if times.size >= 2:
                prev_times = times[:-1]
                next_times = times[1:]
                prev_secs  = secs[:-1]
                next_secs  = secs[1:]

                L = next_times - prev_times
                mids = prev_times + (L // 2)

                # slice-assign per segment
                for i in range(prev_times.size):
                    s0 = prev_times[i] + 1       # start (exclusive)
                    m1 = mids[i] + 1             # first-half end (inclusive) -> slice stop
                    e1 = next_times[i] + 1       # segment end (inclusive) -> slice stop

                    # first half [prev_time+1, mid]
                    if m1 > s0:
                        out[flight_index, s0:m1] = prev_secs[i]
                    # second half [mid+1, next_time]
                    if e1 > m1:
                        out[flight_index, m1:e1] = next_secs[i]

        # Optional: compress width to actually-used time if requested
        if compress and out.shape[1] > 0:
            used_cols = np.any(out != fill_value, axis=0)
            if used_cols.any():
                last_used = np.flatnonzero(used_cols)[-1] + 1
                out = out[:, :last_used]
            else:
                out = out[:, :1]  # keep at least one column

        return out, planned_arrival_times




