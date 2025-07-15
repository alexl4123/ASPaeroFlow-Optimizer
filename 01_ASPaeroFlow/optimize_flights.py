from __future__ import annotations

import argparse
import sys
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
                 seed = 11904657
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

        print("!!!!!!!!!!!!!!!")
        print(f"======== ADDITIONAL TIME INCREASE: {additional_time_increase}")


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

            source = flight_affected[0]
            destination = flight_affected[-1]

            delay = 0
            for tmp_index in range(1,flight_affected.shape[0]):

                flight_sector_instances.append(f"sectorFlight({flight_index},{delay},{source}).")
                flight_times_instance.append(f"flightTime({flight_index},{delay}).")

                if flight_affected[tmp_index] != source:
                    break
            
                delay += 1

            max_delay_parameter = 4

            source_to_target_time = edge_distances[source,destination]

            considered_vertices = considered_vertices.union(set([source,destination]))

            for additional_time in range(-self.timestep_granularity,self.timestep_granularity * (additional_time_increase + max_delay_parameter + delay)):
                current_time = start_time + additional_time
                if current_time >= self.timestep_granularity * (24 + additional_time_increase):
                    break
                elif current_time < 0:
                    continue

                if current_time >= capacity_demand_diff_matrix.shape[1]:
                    current_time = capacity_demand_diff_matrix.shape[1] - 1

                sector_time = f"sector({source},{current_time},{capacity_demand_diff_matrix[source,current_time]})."
                timed_capacities.append(sector_time)



            prev_vertices = None

            for from_source_time in range(1,source_to_target_time + 1): # The +1 means that we add the destination airport
                
                time_to_target_diff = source_to_target_time - from_source_time

                source_vertices = edge_distances[source,:] == from_source_time
                target_vertices = edge_distances[destination,:] == time_to_target_diff

                matching_vertices = source_vertices & target_vertices

                vertex_ids = np.where(matching_vertices)[0]

                #print(f"BEFORE: {len(vertex_ids)}")

                # ------------------------------------------------
                # START PATH STUFF:
                cutoff_value = 3

                if len(vertex_ids) > cutoff_value:
                    # GO INTO PATH MODE:
                    if prev_vertices is None:
                        # FALLBACK
                        rng = np.random.default_rng(seed = self.seed)
                        vertex_ids = rng.choice(vertex_ids, size=cutoff_value, replace=False)
                    else:
                        # prev_vertices is not None
                        # Create path

                        used_vertices = []

                        if len(prev_vertices) >= cutoff_value:
                            del prev_vertices[-1]

                        for prev_vertex in prev_vertices:

                            path_vertices_mask = edge_distances[prev_vertex,:] == 1 & matching_vertices
                            path_vertices_ids = np.where(path_vertices_mask)[0]

                            rng = np.random.default_rng(seed = self.seed)
                            used_vertex_id = rng.choice(path_vertices_ids, size=1, replace=False)
                            used_vertices.append(int(used_vertex_id))

                        flightPathVertex = int(flight_affected[from_source_time + delay])
                        if flightPathVertex not in used_vertices:
                            if len(used_vertices) >= cutoff_value:
                                del used_vertices[-1]
                            used_vertices.append(int(flightPathVertex))

                        vertex_ids = used_vertices

                else:
                    pass
                # -------------------------------------------------------
                
                #print(f"AFTER: {len(vertex_ids)}")

                prev_vertices = list(vertex_ids)

                vertex_instances = [f"sectorFlight({flight_index},{from_source_time + delay},{vertex_id})." for vertex_id in vertex_ids]

                considered_vertices = considered_vertices.union(set(vertex_ids))

                flight_times_instance.append(f"flightTime({flight_index},{from_source_time + delay}).")

                flight_sector_instances += vertex_instances


                for additional_time in range(-self.timestep_granularity,self.timestep_granularity * (additional_time_increase + max_delay_parameter + delay)):
                    current_time = start_time + additional_time + from_source_time
                    if current_time >= self.timestep_granularity * (24 + additional_time_increase):
                        break
                    elif current_time < 0:
                        continue

                    if current_time >= capacity_demand_diff_matrix.shape[1]:
                        current_time = capacity_demand_diff_matrix.shape[1] - 1

                    sector_times = [f"sector({vertex_id},{current_time},{capacity_demand_diff_matrix[vertex_id,current_time]})." for vertex_id in vertex_ids]

                    timed_capacities += sector_times

        flight_sector_instances = "\n".join(flight_sector_instances)
        flight_times_instance = "\n".join(flight_times_instance)

        #print(f">> NUMBER AFFECTED FLIGHTS: {len(rows)}")

        flight_plan_instance = self.flight_plan_strings(rows, flights_affected)
        flight_plan_instance = "\n".join(flight_plan_instance)

        #restricted_loads = self.system_loads_restricted(converted_instance_matrix, self.capacity.shape[0], time_idx = time_index, bucket_idx = bucket_index)
        #capacity_demand_diff_matrix_restricted = capacity_time_matrix - restricted_loads
        #timed_capacities = self.sector_string_matrix(capacity_demand_diff_matrix_restricted) 

        #timed_capacities = self.sector_string_matrix(capacity_demand_diff_matrix)

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
    def bucket_histogram(cls, instance_matrix: np.ndarray,
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

