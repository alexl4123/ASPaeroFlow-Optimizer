
import numpy as np
import networkx as nx

from ..optimize_flights import OptimizeFlights

def compute_total_number_sectors(navaid_sector_time_assignment):

    if navaid_sector_time_assignment.shape[0] == 0:
        number_sectors = 0
    else:
        S = np.sort(navaid_sector_time_assignment, axis=0)                       # sort within each column
        changes = (S[1:, :] != S[:-1, :])            # True where a new value starts
        uniq_per_col = 1 + changes.sum(axis=0)       # unique count per column
    number_sectors = int(uniq_per_col.sum())     # sum over all columns

    return number_sectors

 
def last_valid_pos(arr: np.ndarray) -> np.ndarray:
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


def system_loads_computation(converted_instance_matrix, fill_value, rows, capacity_demand_diff_matrix):

    #system_loads_cpy_2 = system_loads.copy()

    flights_affected = converted_instance_matrix[rows,:]
    # First entered capacity:
    #to_change = (flights_affected[:,1:] != fill_value) & (flights_affected[:,1:] != flights_affected[:,:-1])
    #to_change_first = np.reshape(flights_affected[:,0] != fill_value, (flights_affected.shape[0],1))
    #to_change = np.hstack((to_change_first, to_change)) 

    # Capacity slots:
    to_change = flights_affected != fill_value
    to_change_indices = np.nonzero(to_change)
    flight_affected_buckets = flights_affected[to_change_indices]


    #np.subtract.at(system_loads_cpy_2, (flight_affected_buckets, to_change_indices[1]), 1)
    np.add.at(capacity_demand_diff_matrix, (flight_affected_buckets, to_change_indices[1]), 1)
    # THIS CAN BE IMPROVED

    #capacity_demand_diff_matrix_cpy_2 = capacity_time_matrix - system_loads_cpy_2

    return capacity_demand_diff_matrix

def minimize_number_of_sectors_new(navaid_sector_time_assignment,converted_instance_matrix, converted_navpoint_matrix, capacity_time_matrix, system_loads, max_number_navpoints_per_sector, max_number_sectors, t_start, t_end, navpoint_networkx_graph, airport_vertices):
    # --- Basic setup ----------------------------------------------------
    # Work on inclusive time window [t_start, t_end]
    time_slice = slice(t_start, t_end + 1)
    window_length = t_end - t_start + 1

    # We treat sector IDs as integers in [0, max_sector_id]
    # (this is consistent with capacity_time_matrix / system_loads rows).
    if navaid_sector_time_assignment.size == 0:
        return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads 

    n_sectors = navaid_sector_time_assignment.shape[0]

    # Boolean mask of "active" sectors (can be merged / still exist)
    active = np.ones(n_sectors, dtype=bool)

    # --- Precompute navpoint counts per sector and time -----------------
    # navaid_sector_time_assignment: shape (n_navaids, n_times)
    navpoint_counts = np.zeros((n_sectors, window_length), dtype=np.int32)

    # For each time in the window, count how many navaids are assigned
    # to each sector. Using bincount is usually quite efficient.
    for local_t, global_t in enumerate(range(t_start, t_end + 1)):
        col = navaid_sector_time_assignment[:, global_t]

        # If you use -1 or similar as "no sector", ignore those.
        valid_mask = col >= 0
        if not np.any(valid_mask):
            continue

        counts = np.bincount(col[valid_mask], minlength=n_sectors)
        navpoint_counts[:, local_t] = counts

    # A sector "exists" in the interval iff it has at least one navpoint
    # assigned in that window.
    sector_has_navpoints = navpoint_counts.sum(axis=1) > 0

    # --- Demand / capacity aggregates over the window -------------------
    demand_window = system_loads[:, time_slice]
    capacity_window = capacity_time_matrix[:, time_slice]

    agg_demand = demand_window.sum(axis=1)          # shape (n_sectors,)
    agg_capacity = capacity_window.sum(axis=1)      # shape (n_sectors,)

    # Light sectors: exist, have capacity, and demand <= 50% of capacity
    has_capacity = agg_capacity > 0
    below_threshold = agg_demand <= 0.2 * agg_capacity

    if t_start >= navaid_sector_time_assignment.shape[1]:
        return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads 

    if t_end - t_start > 0:
        not_changed_sectors = np.nonzero(np.all(navaid_sector_time_assignment[:,t_start:t_end] == navaid_sector_time_assignment[:,t_start+1:t_end+1], axis=1))[0]
    else:
        not_changed_sectors = np.nonzero(navaid_sector_time_assignment[:,t_start] != -1)[0]

    light_sector_mask = sector_has_navpoints & has_capacity & below_threshold
    light_sectors = np.nonzero(light_sector_mask)[0]

    light_sectors = np.intersect1d(not_changed_sectors, light_sectors)

    # Sort by aggregated demand ascending (lowest demand first)
    light_sectors = list(light_sectors)
    light_sectors.sort(key=lambda s: agg_demand[s])


    # If there are no light sectors, nothing to do.
    if not light_sectors:
        #print("NO LIGHT SECTORS")
        #print(np.any(sector_has_navpoints))
        #print(np.any(has_capacity))
        #print(np.any(below_threshold))
        return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads 

    # --- Helper to count current active sectors in the window ----------
    def active_sector_count() -> int:
        # A sector is active if it's still marked active AND has some
        # positive demand in the considered window.
        # (You can adapt this condition if your notion of "active"
        #  differs, e.g., also check capacities / navpoints.)
        return int(
            np.logical_and(
                active,
                (system_loads[:, time_slice].sum(axis=1) > 0),
            ).sum()
        )

    tmp_sectors = []
    for sec in light_sectors:
        if sec not in airport_vertices:
            tmp_sectors.append(sec)
    light_sectors = tmp_sectors

    all_relevant_navaids = np.nonzero(np.isin(navaid_sector_time_assignment[:,t_start],light_sectors))[0]
    sub_graph = navpoint_networkx_graph.subgraph(list(all_relevant_navaids)).copy()

    for sector in light_sectors:
        navaids = np.nonzero(navaid_sector_time_assignment[:,t_start] == sector)[0]

        demand = system_loads[sector, t_start:t_end+1]
        capacity = capacity_time_matrix[sector, t_start:t_end+1]

        master_node = sector

        for navaid in navaids[1:]:
            nx.contracted_nodes(sub_graph, master_node, navaid, copy=False, self_loops=False)

        #sub_graph.collapse(navaids, into_node)
        nx.set_node_attributes(sub_graph, {master_node:{"demand":demand,"capacity":capacity, "navaids":list(navaids)}})

    unmarked = list(light_sectors).copy()

    new_sectors = {}

    while len(unmarked) > 0:

        seed = unmarked.pop(0)

        cur_demand = sub_graph.nodes[seed]["demand"]
        cur_capacity = sub_graph.nodes[seed]["capacity"]
        cur_nodes = sub_graph.nodes[seed]["navaids"]

        neighbors = list(sub_graph.neighbors(seed))

        neighbors = [neighbor for neighbor in neighbors if neighbor in unmarked]

        sector_bag = [seed]

        while len(neighbors) > 0:
            neighbor = neighbors.pop(0)
            if neighbor not in unmarked:
                continue

            neigh_demand = sub_graph.nodes[neighbor]["demand"]
            neigh_capacity = sub_graph.nodes[neighbor]["capacity"]
            neigh_nodes = sub_graph.nodes[neighbor]["navaids"]

            demand = cur_demand + neigh_demand
            capacity = cur_capacity + neigh_capacity
            navaids = cur_nodes + neigh_nodes


            if len(navaids) > max_number_navpoints_per_sector:
                continue

            if np.any(demand > capacity):
                continue
            
            sector_bag.append(neighbor)

            unmarked.remove(neighbor)
            neighbors_tmp = list(sub_graph.neighbors(neighbor))
            neighbors_tmp = [neighbor for neighbor in neighbors_tmp if neighbor in unmarked]
            neighbors = list(set(neighbors + neighbors_tmp))

            if seed in neighbors:
                neighbors.remove(seed)
            
            nx.contracted_nodes(sub_graph, seed, neighbor, copy=False, self_loops=False)
            nx.set_node_attributes(sub_graph, {seed:{"demand":demand,"capacity":capacity, "navaids":navaids}})

            cur_capacity = capacity
            cur_demand = demand
            cur_nodes = navaids

        new_sectors[seed] = sector_bag

    for sec in new_sectors.keys():
        merged_sectors = new_sectors[sec]
        capacity = sub_graph.nodes[sec]["capacity"]
        demand = sub_graph.nodes[sec]["demand"]
        navaids = sub_graph.nodes[sec]["navaids"]

        capacity_time_matrix[merged_sectors, t_start:t_end + 1] = 0
        system_loads[merged_sectors, t_start:t_end + 1] = 0

        capacity_time_matrix[sec, t_start:t_end + 1] = capacity
        system_loads[sec, t_start:t_end + 1] = demand

        navaid_sector_time_assignment[navaids, t_start:t_end + 1] = sec


    converted_instance_matrix = OptimizeFlights.instance_computation_after_sector_change(list(range(converted_instance_matrix.shape[0])),
                        converted_navpoint_matrix, converted_instance_matrix, navaid_sector_time_assignment)
    

    return converted_instance_matrix, converted_navpoint_matrix, navaid_sector_time_assignment, capacity_time_matrix, system_loads


