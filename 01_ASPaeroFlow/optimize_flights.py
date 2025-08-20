from __future__ import annotations

import time

import numpy as np

from solver import Solver, Model

    
class OptimizeFlights:

    def __init__(self,
                 encoding, capacity, graph, airport_vertices,
                 flights_affected, rows, edge_distances,
                 converted_instance_matrix, time_index, bucket_index,
                 capacity_time_matrix, capacity_demand_diff_matrix,
                 additional_time_increase,
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

        self.flights_affected = flights_affected
        self.rows = rows
        self.edge_distances = edge_distances
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

        flights_affected = self.flights_affected
        rows = self.rows
        edge_distances = self.edge_distances
        converted_instance_matrix = self.converted_instance_matrix
        time_index = self.time_index
        bucket_index = self.bucket_index
        capacity_time_matrix = self.capacity_time_matrix
        capacity_demand_diff_matrix = self.capacity_demand_diff_matrix
        additional_time_increase = self.additional_time_increase
        fill_value = self.fill_value

        max_vertices_cutoff_value = self.max_vertices_cutoff_value
        max_delay_parameter = self.max_delay_parameter

        flight_sector_instances = []
        flight_times_instance = []
        timed_capacities = []

        considered_vertices = set()


        for flight_affected_index in range(flights_affected.shape[0]):

            flight_index = rows[flight_affected_index]

            flight_affected = flights_affected[flight_affected_index,:]

            start_time = np.flatnonzero(flight_affected != fill_value)
            start_time = start_time[0] if start_time.size else 0
            
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            delay, flight_sector_instances, flight_times_instance = self.handle_delay(flight_sector_instances, flight_times_instance, flight_index, flight_affected, origin)

            origin_to_destination_time = edge_distances[origin,destination]

            considered_vertices = considered_vertices.union(set([origin,destination]))

            timed_capacities += self.handle_origin_sector_instance_generation(capacity_demand_diff_matrix, additional_time_increase, max_delay_parameter, start_time, origin, delay)


            prev_vertices = None

            for from_origin_time in range(1,origin_to_destination_time + 1): # The +1 means that we add the destination airport
                
                time_to_destination_diff = origin_to_destination_time - from_origin_time

                origin_vertices = edge_distances[origin,:] == from_origin_time
                destination_vertices = edge_distances[destination,:] == time_to_destination_diff

                matching_vertices = origin_vertices & destination_vertices

                vertex_ids = np.where(matching_vertices)[0]
                vertex_ids = self.restrict_max_vertices(prev_vertices, vertex_ids, matching_vertices, flight_affected, from_origin_time, delay)

                prev_vertices = list(vertex_ids)

                vertex_instances = [f"sectorFlight({flight_index},{from_origin_time + delay},{vertex_id})." for vertex_id in vertex_ids]

                considered_vertices = considered_vertices.union(set(vertex_ids))

                flight_times_instance.append(f"flightTime({flight_index},{from_origin_time + delay}).")

                flight_sector_instances += vertex_instances

                timed_capacities += self.handle_sectors_instance_generation(capacity_demand_diff_matrix, additional_time_increase, max_delay_parameter, start_time, delay, from_origin_time, vertex_ids, flight_index)

        flight_sector_instances = "\n".join(flight_sector_instances)
        flight_times_instance = "\n".join(flight_times_instance)

        flight_plan_instance = self.flight_plan_strings(rows, flights_affected)
        flight_plan_instance = "\n".join(flight_plan_instance)

        edges_instance, airport_instance, timed_capacities = self.not_used_airports_removal_from_instance(fill_value, flights_affected, timed_capacities, capacity_demand_diff_matrix, considered_vertices, self.graph)
        timed_capacities_instance = '\n'.join(timed_capacities)

        time_instance = f"additionalTime({additional_time_increase})."
        timestep_granularity_instance = f"timestepGranularity({self.timestep_granularity})."
        instance = edges_instance + "\n" + timed_capacities_instance + "\n" + airport_instance + "\n" + time_instance + "\n" + flight_plan_instance + "\n" + flight_sector_instances + "\n" + flight_times_instance + "\n" + timestep_granularity_instance

        open("test_instance_4.lp","w").write(instance)

        encoding = self.encoding

        start_time = time.time()

        solver: Model = Solver(encoding, instance)
        model = solver.solve()

        end_time = time.time()
        #print(f">> Elapsed solving time: {end_time - start_time}")
        #print(model.get_flights())
        #quit()

        return model

    def handle_sectors_instance_generation(self, capacity_demand_diff_matrix, additional_time_increase, max_delay_parameter, start_time, delay, from_origin_time, vertex_ids, flight_index):

        timed_capacities = []

        for additional_time in range(-self.timestep_granularity,self.timestep_granularity * (additional_time_increase + max_delay_parameter + delay)):
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

    def handle_origin_sector_instance_generation(self, capacity_demand_diff_matrix, additional_time_increase, max_delay_parameter, start_time, origin, delay):

        timed_capacities = []

        for additional_time in range(-self.timestep_granularity,self.timestep_granularity * (additional_time_increase + max_delay_parameter + delay)):
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

    def handle_delay(self, flight_sector_instances, flight_times_instance, flight_index, flight_affected, origin):
        delay = 0

        for tmp_index in range(1,flight_affected.shape[0]):
            flight_sector_instances.append(f"sectorFlight({flight_index},{delay},{origin}).")
            flight_times_instance.append(f"flightTime({flight_index},{delay}).")

            if flight_affected[tmp_index] != origin:
                break
            
            delay += 1

        return delay, flight_sector_instances, flight_times_instance
    
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
        # np.savetxt("20250819_bucket_histogram.csv", bucket_histogram, delimiter=",",fmt="%i")

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

        return hist


