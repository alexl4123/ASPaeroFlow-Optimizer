from __future__ import annotations

import time
import math

import numpy as np
import networkx as nx
from itertools import islice

from solver import Solver, Model

    
class OptimizeFlights:

    def __init__(self,
                 encoding, capacity, graph, airport_vertices,
                 flights_affected, rows, 
                 converted_instance_matrix, time_index, bucket_index,
                 capacity_time_matrix, capacity_demand_diff_matrix,
                 additional_time_increase,
                 networkx_graph,
                 planned_arrival_times,
                 airplane_flight,
                 airplanes,
                 flights,
                 navaid_sector,
                 navaid_sector_lookup,
                 nearest_neighbors_lookup,
                 fill_value = -1,
                 timestep_granularity = 1,
                 seed = 11904657,
                 max_vertices_cutoff_value = 6,
                 max_delay_parameter = 4,
                 original_max_time = 24,
                 ):

        self.encoding = encoding
        self.capacity = capacity
        self.graph = graph
        self.airport_vertices = airport_vertices
        self.networkx_graph = networkx_graph
        self.planned_arrival_times = planned_arrival_times
        self.airplane_flight = airplane_flight
        self.flights = flights
        self.navaid_sector = navaid_sector
        self.airplanes = airplanes
        self.navaid_sector_lookup = navaid_sector_lookup
        self.nearest_neighbors_lookup = nearest_neighbors_lookup

        self.flights_affected = flights_affected
        self.rows = rows
        self.converted_instance_matrix = converted_instance_matrix
        self.time_index = time_index
        self.bucket_index = bucket_index
        self.capacity_time_matrix = capacity_time_matrix
        self.capacity_demand_diff_matrix = capacity_demand_diff_matrix
        self.additional_time_increase = additional_time_increase
        self.fill_value = fill_value
        self.timestep_granularity = timestep_granularity
        self.seed = seed

        self.max_vertices_cutoff_value = max_vertices_cutoff_value
        self.max_delay_parameter = max_delay_parameter
        self.original_max_time = original_max_time
 
    def start(self):

        timestep_granularity = self.timestep_granularity
        flights_affected = self.flights_affected
        rows = self.rows
        converted_instance_matrix = self.converted_instance_matrix
        time_index = self.time_index
        bucket_index = self.bucket_index
        capacity_time_matrix = self.capacity_time_matrix
        capacity_demand_diff_matrix = self.capacity_demand_diff_matrix
        additional_time_increase = self.additional_time_increase
        fill_value = self.fill_value
        networkx_graph = self.networkx_graph

        max_vertices_cutoff_value = self.max_vertices_cutoff_value
        max_delay_parameter = self.max_delay_parameter

        flight_sector_instances = []
        flight_times_instance = []

        sector_instance = {}
        graph_instance = {}
        path_fact_instances = []
        planned_departure_time_instance = []
        actual_departure_time_instance = []
        planned_arrival_time_instance = []
        actual_arrival_time_instance = []

        time_window = max_delay_parameter
        k = max_vertices_cutoff_value

        for flight_affected_index in range(flights_affected.shape[0]):

            flight_index = rows[flight_affected_index]

            flight_affected = flights_affected[flight_affected_index,:]

            #actual_arrival_time = (flight_affected >= 0).argmax() - 1
            actual_arrival_time = np.flatnonzero(flight_affected >= 0)[-1] 

            planned_arrival_time = self.planned_arrival_times[flight_index]
            actual_delay = actual_arrival_time - planned_arrival_time

            original_start_time = np.flatnonzero(flight_affected != fill_value)
            original_start_time = original_start_time[0] if original_start_time.size else 0
            start_time = original_start_time + actual_delay
            
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            # THIS IS INCLUDED IN THE SHORTEST PATH:
            #flight_sector_instances, flight_times_instance, sector_instance, graph_instance = self.create_filed_flight_plan_atoms(flight_sector_instances, flight_times_instance, sector_instance, graph_instance, flight_index, flight_affected, capacity_demand_diff_matrix, start_time)

            paths = self.k_diverse_near_shortest_paths(networkx_graph, origin, destination, k=k, eps=0.1, jaccard_max=0.6,
                                               penalty_scale=0.1, max_tries=50, weight_key="weight")
            
            airplane_id = (self.airplane_flight[self.airplane_flight[:,1] == flight_index])[0,0]
            airplane_speed_kts = self.airplanes[airplane_id,1]
            current_flight = self.flights[self.flights[:,0] == flight_index]

            path_number = 0

            planned_departure_time_instance.append(f"plannedDepartureTime({airplane_id},{original_start_time}).")
            actual_departure_time_instance.append(f"actualDepartureTime({airplane_id},{start_time}).")
            planned_arrival_time_instance.append(f"plannedArrivalTime({airplane_id},{planned_arrival_time}).")

            for path in paths:

                navpoint_trajectory = self.get_flight_navpoint_trajectory(flights_affected, networkx_graph, flight_index, start_time, airplane_speed_kts, path, timestep_granularity)

                for delay in range(additional_time_increase * time_window,time_window * (additional_time_increase + 1)):

                    flight_time = 0
                    current_time = original_start_time

                    for flight_hop_index in range(len(navpoint_trajectory)):

                        navaid = navpoint_trajectory[flight_hop_index][1]
                        time = navpoint_trajectory[flight_hop_index][2]
                        #sector = (self.navaid_sector[self.navaid_sector[:,0] == navaid])[0,1]
                        sector = self.navaid_sector_lookup[navaid]


                        graph_instance[f"sectorEdge({sector},{sector})."] = True

                        if flight_hop_index == 0:

                            for _ in range(actual_delay + delay + 1):

                                flight_sector_instances.append(f"sectorFlight({flight_index},{flight_time},{sector},{path_number}).")
                                flight_times_instance.append(f"flightTime({flight_index},{flight_time}).")
                                sector_instance[f"sector({sector},{current_time},{capacity_demand_diff_matrix[sector,current_time]})."] = True

                                flight_time += 1
                                current_time += 1

                        else:

                            prev_navaid = navpoint_trajectory[flight_hop_index-1][1]
                            prev_time = navpoint_trajectory[flight_hop_index-1][2]
                            #prev_sector = (self.navaid_sector[self.navaid_sector[:,0] == prev_navaid])[0,1]
                            prev_sector = self.navaid_sector_lookup[prev_navaid]

                            graph_instance[f"sectorEdge({prev_sector},{sector})."] = True

                            for time_index in range(1, time-prev_time + 1):
                                
                                flight_times_instance.append(f"flightTime({flight_index},{flight_time}).")

                                if time_index <= math.floor((time - prev_time)/2):

                                    flight_sector_instances.append(f"sectorFlight({flight_index},{flight_time},{prev_sector},{path_number}).")
                                    sector_instance[f"sector({prev_sector},{current_time},{capacity_demand_diff_matrix[prev_sector,current_time]})."] = True

                                else:

                                    flight_sector_instances.append(f"sectorFlight({flight_index},{flight_time},{sector},{path_number}).")
                                    sector_instance[f"sector({sector},{current_time},{capacity_demand_diff_matrix[sector,current_time]})."] = True

                                current_time += 1
                                flight_time += 1

                    actual_arrival_time_instance.append(f"actualArrivalTime({airplane_id},{current_time - 1},{path_number}).")

                    # path_numbers = #PATHS * #DELAYS
                    path_number += 1

            path_fact_instances.append(f"paths({airplane_id},0..{path_number - 1}).")

        flight_sector_instances = "\n".join(flight_sector_instances)
        flight_times_instance = "\n".join(flight_times_instance)
        sector_instance = "\n".join(sector_instance.keys())

        path_fact_instances = "\n".join(path_fact_instances)

        flight_plan_instance = self.flight_plan_strings(rows, flights_affected)
        flight_plan_instance = "\n".join(flight_plan_instance)

        graph_instance = "\n".join(graph_instance.keys())

        planned_departure_time_instance = "\n".join(planned_departure_time_instance)
        actual_departure_time_instance = "\n".join(actual_departure_time_instance)
        planned_arrival_time_instance = "\n".join(planned_arrival_time_instance)
        actual_arrival_time_instance = "\n".join(actual_arrival_time_instance)

        instance = f"""
{graph_instance}
{flight_plan_instance}
{flight_sector_instances}
{sector_instance}
{path_fact_instances}
{planned_departure_time_instance}
{actual_departure_time_instance}
{planned_arrival_time_instance}
{actual_arrival_time_instance}
        """
        #open(f"test_instance_4_{additional_time_increase}.lp","w").write(instance)

        encoding = self.encoding

        solver: Model = Solver(encoding, instance)
        model = solver.solve()

        return model

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
        G, s, t, k=5, eps=0.10, jaccard_max=0.6,
        penalty_scale=0.5, max_tries=200, weight_key="weight"
    ):
        
        s_t_length, _ = nx.bidirectional_dijkstra(G, s, t, weight=weight_key)

        allowed = (1.0 + eps) * s_t_length

        # 1) shortest length & prune to a small corridor: ds[u]+dt[u] ≤ (1+eps)*L0
        #
        if s not in self.nearest_neighbors_lookup:
            ds = nx.single_source_dijkstra_path_length(G, s, weight=weight_key)
            self.nearest_neighbors_lookup[s] = ds
        else:
            ds = self.nearest_neighbors_lookup[s]

        if t not in self.nearest_neighbors_lookup:
            dt = nx.single_source_dijkstra_path_length(G, t, weight=weight_key)
            self.nearest_neighbors_lookup[t] = dt
        else:
            dt = self.nearest_neighbors_lookup[t]

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
        change = np.zeros_like(valid, dtype=bool)
        change[:, 0] = valid[:, 0]
        if T > 1:
            change[:, 1:] = valid[:, 1:] & (inst[:, 1:] != inst[:, :-1])

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

        #np.savetxt("20250819_histogram.csv", hist, delimiter=",",fmt="%i")

        return hist


