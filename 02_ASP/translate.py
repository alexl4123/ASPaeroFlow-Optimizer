
#!/usr/bin/env python3
"""read_csv_files.py

Usage
-----
    python read_csv_files.py edges.csv instance_10.csv capacity_10.csv # will read instance_10.csv and capacity_10.csv along with edges.csv
"""

import csv
import pathlib
import sys
from pathlib import Path
from typing import Iterable, List, Any

import math
import numpy as np
import networkx as nx

LINEAR = "linear"
TRIANGULAR = "triangular"
MAX = "max"

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
    def conv(x):
        return int(math.ceil(float(x)))

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=1, converters=conv)
    except ValueError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc



class TranslateCSVtoLogicProgram:

    def read_csv(self, path: pathlib.Path) -> Iterable[List[str]]:
        """Yield rows from *path* as lists of strings.

        Parameters
        ----------
        path : pathlib.Path
            The path to the CSV file to read.
        """
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader)
                for row in reader:
                    yield row
        except FileNotFoundError:
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)


    def convert_graph(self, unit_graphs) -> List[str]:
        """Convert Edges to ASP"""

        navpoint_edges = []

        for speed in unit_graphs.keys():

            for edge in unit_graphs[speed].edges(data=True):

                x0 = edge[0]
                x1 = edge[1]
                unit_distance = edge[2]["weight"]        

                navpoint_edges.append(f"navpoint_edge({x0},{x1},{speed},{unit_distance}).")

        return navpoint_edges


    def convert_flights(self, flights) -> List[str]:
        """Convert Instance to ASP"""

        max_time = 0 
        asp_atoms = []
        for row_index in range(flights.shape[0]):
            # ASP expects flightPlan(<ID>,<TIME>,<LOCATION>)
            asp_atoms.append(f"navpoint_flight_plan({flights[row_index,0]},{flights[row_index,1]},{flights[row_index,2]}).")

            if int(flights[row_index,2]) > max_time:
                max_time = int(flights[row_index,2])

        return asp_atoms, max_time

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
                f"navpoint_sector_time_assignment must be shape (N, n_times) = ({N}, {n_times})"
            )

        S = navaid_sector_time_assignment.astype(np.int64, copy=False)
        if S.min() < 0 or S.max() >= N:
            print(S)
            raise ValueError("Sector ids in navpoint_sector_time_assignment must be in [0, N-1].")

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

 



    def convert_sectors(self, sectors, timestep_granularity, max_time, max_time_flights, navaid_sector_time_assignment) -> List[str]:
        """Convert Capacity to ASP"""

        max_time_dim = int(max(max_time_flights, (max_time + 1) * timestep_granularity, navaid_sector_time_assignment.shape[1]))

        tmp_navaid_sector_time_assignment = np.zeros(navaid_sector_time_assignment.shape, dtype=int)

        for index in range(navaid_sector_time_assignment.shape[0]):
            tmp_navaid_sector_time_assignment[index,:] = index

        sectors_matrix = TranslateCSVtoLogicProgram.capacity_time_matrix(sectors, max_time_dim, timestep_granularity, tmp_navaid_sector_time_assignment)
        
        asp_atoms = []
        for navaid_index in range(sectors_matrix.shape[0]):
            for time_index in range(sectors_matrix.shape[1]):
                # ASP expects flightPlan(<ID>,<TIME>,<LOCATION>)
                asp_atoms.append(f"atomic_sector({navaid_index},{sectors_matrix[navaid_index,time_index]},{time_index}).")

        return asp_atoms

    def convert_airports(self, airports) -> List[str]:
        """Convert Airport Vertices to ASP"""
        
        airport_instance = []

        for row_index in range(airports.shape[0]):
            airport_instance.append(f"airport({airports[row_index]}).")

        return airport_instance

    def convert_airplanes(self, airplanes) -> List[str]:
        """Convert airplanes to ASP"""
        
        airplane_instance = []

        for row_index in range(airplanes.shape[0]):
            airplane_instance.append(f"aircraft({airplanes[row_index,0]},{airplanes[row_index,1]}).")

        return airplane_instance

    def convert_airplane_flight(self, airplane_flight) -> List[str]:
        """Convert airplane_flight to ASP"""
        
        airplane_flight_instance = []

        for row_index in range(airplane_flight.shape[0]):
            airplane_flight_instance.append(f"aircraft_flight({airplane_flight[row_index,0]},{airplane_flight[row_index,1]}).")

        return airplane_flight_instance

    def convert_navaid_sector(self, navaid_sector, networkx_graph) -> List[str]:
        """Convert navaid_sector to ASP"""
        
        navaid_sector_instance = []

        for row_index in range(navaid_sector.shape[0]):
            navaid_sector_instance.append(f"navpoint_sector({navaid_sector[row_index,0]},{navaid_sector[row_index,1]},0).")

        # Build per-sector navpoint lists
        from collections import defaultdict, deque
        import networkx as nx

        sector_to_navs = defaultdict(set)
        for row_index in range(navaid_sector.shape[0]):
            nav = str(navaid_sector[row_index, 0])
            sec = str(navaid_sector[row_index, 1])
            sector_to_navs[sec].add(nav)

        def _bfs_order(G: nx.Graph, start: str) -> List[str]:
            """Deterministic BFS order (neighbors sorted by str)."""
            seen = {start}
            q = deque([start])
            order = []
            while q:
                u = q.popleft()
                order.append(u)
                for v in sorted(G.neighbors(u), key=str):
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            return order

        def _split_connected(G: nx.Graph, nodes: List[str]):
            """
            Attempt to split 'nodes' into two connected parts, roughly half/half.
            Deterministic: BFS-prefix is connected; we search for a cut where the remainder is connected.
            Returns (A, B) as sets, or None if no valid split found.
            """
            if len(nodes) < 2:
                return None
            nodes_sorted = sorted(nodes, key=str)
            root = nodes_sorted[0]
            order = _bfs_order(G, root)
            if len(order) != len(nodes_sorted):
                return None  # not connected

            desired = len(nodes_sorted) // 2
            for delta in range(len(nodes_sorted)):
                for m in (desired - delta, desired + delta):
                    if m <= 0 or m >= len(nodes_sorted):
                        continue
                    A = set(order[:m])           # connected by construction
                    B = set(nodes_sorted) - A
                    if not B:
                        continue
                    # B must be connected
                    if len(B) == 1 or nx.is_connected(G.subgraph(B)):
                        return A, B
            return None

        def _partition_k(G: nx.Graph, nodes: List[str], k: int):
            """k in {2,4}. Returns list[set] of connected parts, or None if impossible."""
            nodes_sorted = sorted(nodes, key=str)
            if not nodes_sorted:
                return []
            sub = G.subgraph(nodes_sorted)
            if k == 2:
                split = _split_connected(sub, nodes_sorted)
                return list(split) if split is not None else None
            if k == 4:
                split = _split_connected(sub, nodes_sorted)
                if split is None:
                    return None
                A, B = split
                splitA = _split_connected(sub.subgraph(A), sorted(A, key=str))
                splitB = _split_connected(sub.subgraph(B), sorted(B, key=str))
                if splitA is None or splitB is None:
                    return None
                return [splitA[0], splitA[1], splitB[0], splitB[1]]
            raise ValueError("k must be 2 or 4")

        def _emit(sec: str, dec: int, nav_to_sec1: dict):
            for nav in sorted(nav_to_sec1.keys(), key=str):
                sec1 = nav_to_sec1[nav]
                navaid_sector_instance.append(
                    f"navpoint_sector_restricted_sector_allocation({sec},{dec},{nav},{sec1})."
                )

        # For each sector: offer DEC in {0,1,2,3}
        for sec, navs_set in sector_to_navs.items():
            navs = sorted(navs_set, key=str)
            navs = [int(n) for n in navs]

            # Induced graph on sector navpoints (add missing nodes as isolated)
            present = [n for n in navs if n in networkx_graph]
            missing = [n for n in navs if n not in networkx_graph]
            induced = networkx_graph.subgraph(present).copy()
            if missing:
                induced.add_nodes_from(missing)

            # DEC 0: unchanged allocation (SEC1 == SEC)
            _emit(sec, 0, {nav: sec for nav in navs})

            # DEC 3: atomic allocation (SEC1 == NAV1)
            _emit(sec, 3, {nav: nav for nav in navs})

            if induced.number_of_nodes() > 1:
                # DEC 1: two connected parts (fallback: DEC 0 mapping)
                parts2 = _partition_k(induced, navs, 2)
                if parts2 is None:
                    nav_to_rep2 = {nav: sec for nav in navs}
                else:
                    reps2 = {tuple(sorted(part, key=str))[0]: part for part in parts2}  # rep is min node
                    nav_to_rep2 = {}
                    for rep, part in reps2.items():
                        for nav in part:
                            nav_to_rep2[nav] = rep
                _emit(sec, 1, nav_to_rep2)

                # DEC 2: four connected parts (fallback: DEC 1 mapping; if that fell back, effectively DEC 0)
                parts4 = _partition_k(induced, navs, 4)
                if parts4 is None:
                    nav_to_rep4 = dict(nav_to_rep2)
                else:
                    reps4 = {tuple(sorted(part, key=str))[0]: part for part in parts4}
                    nav_to_rep4 = {}
                    for rep, part in reps4.items():
                        for nav in part:
                            nav_to_rep4[nav] = rep
                _emit(sec, 2, nav_to_rep4)
            else:
                _emit(sec, 1, {nav: sec for nav in navs})
                _emit(sec, 2, {nav: sec for nav in navs})

        return navaid_sector_instance
    
    def load_data(self, graph_path, sectors_path, flights_path, airports_path, airplanes_path, airplane_flight_path, navaid_sector_path, encoding_path) -> None:
        """Load all CSV files provided on the command line."""
        self.graph = _load_csv(graph_path)
        self.sectors = _load_csv(sectors_path)
        self.flights = _load_csv(flights_path)
        self.airports = _load_csv(airports_path)
        self.airplanes = _load_csv(airplanes_path)
        self.airplane_flight = _load_csv(airplane_flight_path)
        self.navaid_sector = _load_csv(navaid_sector_path)

        with open(encoding_path, "r") as file:
            self.encoding = file.read()


    def get_bounded_choice_routes(self, flights, airplane_flight, airplanes, navaid_sector, max_time, timestep_granularity,navaid_sector_time_assignment, unit_graphs, networkx_graph):

        #converted_navpoint_matrix, _ = TranslateCSVtoLogicProgram.instance_navpoint_matrix(self.flights, navaid_sector_time_assignment.shape[1], fill_value=-1)
        converted_instance_matrix, planned_arrival_times = TranslateCSVtoLogicProgram.instance_to_matrix(flights, airplane_flight, navaid_sector_time_assignment.shape[1], timestep_granularity, navaid_sector_time_assignment)

        flights_affected = converted_instance_matrix
        problematic_flights = converted_instance_matrix
        converted_instance_matrix = converted_instance_matrix

        flight_navpoint_instance = []
        flight_times_instance = []
        
        needed_capacities_for_navpoint = {}

        sector_instance = {}
        path_fact_instances = []

        planned_departure_time_instance = []
        actual_departure_time_instance = []
        planned_arrival_time_instance = []
        actual_arrival_time_instance = []

        fill_value = -1
        k = 4
        largest_considered_time = 0

        regulation_restricted_rerouting_instance = []


        for flight_affected_index in range(flights_affected.shape[0]):


            #flight_index = int(problematic_flights[flight_affected_index])
            flight_index = flight_affected_index

            airplane_id = (airplane_flight[airplane_flight[:,1] == flight_index])[0,0]
            airplane_speed_kts = airplanes[airplane_id,1]
            #current_flight = self.flights[self.flights[:,0] == flight_index]

            flight_affected = flights_affected[flight_affected_index,:]

            filed_flight_path = flights[flights[:,0] == flight_index,:]

            #actual_arrival_time = (flight_affected >= 0).argmax() - 1
            actual_arrival_time = np.flatnonzero(flight_affected >= 0)[-1] 

            # TURNAROUND TIME:
            # If a flight needs more than 1 timestep to prepare for departure they coincide; otherwise different
            actual_flight_departure_time = 0

            #actual_flight_departure_time = planned_flight_departure_time + actual_delay
            
            flight_affected = flight_affected[flight_affected != fill_value]

            origin = flight_affected[0]
            destination = flight_affected[-1]

            # WITH A FILED FLIGHT PATH WE GET PATH NUMBER = 0
            paths = self.k_diverse_near_shortest_paths(unit_graphs[airplane_speed_kts], origin, destination, {},
                                                k=k, eps=0.1, jaccard_max=0.6, penalty_scale=0.1, max_tries=50, weight_key="weight",
                                                filed_path=list(filed_flight_path[:,1]))


            path_id = 0

            for path in paths:
                navpoint_trajectory = self.get_flight_navpoint_trajectory(flights_affected, networkx_graph, flight_index, actual_flight_departure_time, airplane_speed_kts, path, timestep_granularity)


                for flight_id, flight_navpoint, flight_time in navpoint_trajectory:
                    regulation_restricted_rerouting_instance.append(f"regulation_restricted_rerouting_paths({flight_index},{flight_navpoint},{flight_time},{path_id}).")

                path_id += 1


        return regulation_restricted_rerouting_instance
            
    
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


           

    def main(self, graph_path: str, flights_path: str, sectors_path: str, airports_path: str, airplanes_path, airplane_flight_path,
             navaid_sector_path, encoding_path, timestep_granularity: int, max_time,
             sector_capactiy_factor,
             regulation_ground_delay_active,
             regulation_rerouting_active,
             regulation_dynamic_sectorization_active) -> None:
        """Read and print rows from the three required CSV files."""

        self.load_data(graph_path, sectors_path, flights_path, airports_path, airplanes_path,
                       airplane_flight_path, navaid_sector_path, encoding_path)

        sources = self.graph[:,0]
        targets = self.graph[:,1]
        dists = self.graph[:,2]
        self.networkx_navpoint_graph = nx.Graph()
        self.networkx_navpoint_graph.add_weighted_edges_from(zip(sources, targets, dists))

        self.unit_graphs = {}

        navaid_sector_time_assignment = self.create_initial_navpoint_sector_assignment(self.flights, self.airplane_flight, self.navaid_sector, max_time, timestep_granularity)

        different_speeds = list(set(list(self.airplanes[:,1])))
        for cur_airplane_speed in different_speeds:

            self.unit_graphs[cur_airplane_speed] = self.networkx_navpoint_graph.copy()
            graph = self.unit_graphs[cur_airplane_speed]

            tmp_edges = {}

            for edge in graph.edges(data=True):

                distance = edge[2]["weight"]
                # CONVERT AIRPLANE SPEED TO m/s
                airplane_speed_ms = cur_airplane_speed * 0.51444
                duration_in_seconds = distance/airplane_speed_ms
                factor_to_unit_standard = 3600.00 / float(timestep_granularity)
                duration_in_unit_standards = math.ceil(duration_in_seconds / factor_to_unit_standard)

                duration_in_unit_standards = max(duration_in_unit_standards, 1)

                tmp_edges[(edge[0],edge[1])] = {"weight":duration_in_unit_standards}

                #print(f"{distance}/{airplane_speed_ms} = {duration_in_seconds}")
                #print(f"{duration_in_seconds}/{factor_to_unit_standard} = {duration_in_unit_standards}")

            nx.set_edge_attributes(graph, tmp_edges)

        graph_instance = self.convert_graph(self.unit_graphs)
        flights_instance, max_time_flights = self.convert_flights(self.flights)
        regulation_restricted_rerouting_trajectories = []
        if regulation_rerouting_active == 1:
            regulation_restricted_rerouting_trajectories = self.get_bounded_choice_routes(self.flights, self.airplane_flight, self.airplanes, self.navaid_sector, max_time, timestep_granularity,navaid_sector_time_assignment, self.unit_graphs, self.networkx_navpoint_graph)

        sectors_instance = self.convert_sectors(self.sectors, timestep_granularity, max_time, max_time_flights, navaid_sector_time_assignment)
        airports_instance = self.convert_airports(self.airports)
        airplanes_instance = self.convert_airplanes(self.airplanes)
        airplane_flight_instance = self.convert_airplane_flight(self.airplane_flight)
        navaid_sector_instance = self.convert_navaid_sector(self.navaid_sector, self.networkx_navpoint_graph)

        sector_capactiy_factor_instance = [f"sector_capacity_factor({sector_capactiy_factor})."]
        timestep_granularity_instance = [f"timestep_granularity({timestep_granularity})."]
        max_time_instance = [f"max_time({max_time*timestep_granularity})."]
        #max_time_instance = [f"maxTime(30)."]

        regulation_instance = []

        if regulation_ground_delay_active == 0:
            regulation_instance.append("-regulation_delaying.")
        elif regulation_ground_delay_active == 1:
            regulation_instance.append("regulation_restricted_delaying.")
            regulation_instance.append("restricted_delaying_max_delay(4).")
        elif regulation_ground_delay_active == 2:
            regulation_instance.append("regulation_delaying.")

        if regulation_rerouting_active == 0:
            regulation_instance.append("-regulation_rerouting.")
        elif regulation_rerouting_active == 1:
            regulation_instance.append("regulation_restricted_rerouting.")
        elif regulation_rerouting_active == 2:
            regulation_instance.append("regulation_rerouting.")

        if regulation_dynamic_sectorization_active == 0:
            regulation_instance.append("-regulation_dynamic_sector_allocation.")
        elif regulation_dynamic_sectorization_active == 1:
            regulation_instance.append("regulation_restricted_dynamic_sector_allocation.")
        elif regulation_dynamic_sectorization_active == 2:
            regulation_instance.append("regulation_dynamic_sector_allocation.")


        instance = graph_instance + flights_instance + sectors_instance + airplanes_instance +\
            airports_instance + airplane_flight_instance + navaid_sector_instance +\
            timestep_granularity_instance + max_time_instance+\
            sector_capactiy_factor_instance + regulation_instance+\
            regulation_restricted_rerouting_trajectories
    
        return instance

    def create_initial_navpoint_sector_assignment(self,
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



    @classmethod
    def instance_to_matrix(cls,
                            flights: np.ndarray,
                            airplane_flight: np.ndarray,
                            max_time: int,
                            time_granularity: int,
                            navaid_sector_time_assignment: np.ndarray,
                            *,
                            fill_value: int = -1,
                            compress: bool = False):

        vals = flights[:, 1]
        cols = flights[:, 2].astype(int)


        # --- output matrix shape (airplane_id rows, time columns)
        n_rows = int(airplane_flight[:, 1].max()) + 1
        out = np.full((n_rows, int(max_time)), fill_value, dtype=np.int64)
       
        flight_ids = flights[:,0]
        flight_ids = np.unique(flight_ids)

        planned_arrival_times = {}

        for flight_id in flight_ids:

            #airplane_id = (airplane_flight[airplane_flight[:,1] == flight_id])[0,0]

            current_flight = flights[flights[:,0] == flight_id]

            for flight_hop_index in range(current_flight.shape[0]):

                navaid = current_flight[flight_hop_index,1]
                time = current_flight[flight_hop_index,2]
                #sector = (navaid_sector[navaid_sector[:,0] == navaid])[0,1]

                if flight_hop_index == 0:
                    sector = navaid_sector_time_assignment[navaid,time]
                    out[flight_id,time] = sector
                else:

                    prev_navaid = current_flight[flight_hop_index - 1,1]
                    prev_time = current_flight[flight_hop_index - 1,2]

                    for time_index in range(1, time-prev_time + 1):

                        if time_index <= math.floor((time - prev_time)/2):
                            prev_sector = navaid_sector_time_assignment[prev_navaid,prev_time + time_index]
                            out[flight_id,prev_time + time_index] = prev_sector

                        else:
                            sector = navaid_sector_time_assignment[navaid,prev_time + time_index]
                            out[flight_id,prev_time + time_index] = sector

                if flight_id not in planned_arrival_times:
                    planned_arrival_times[flight_id] = time
                elif planned_arrival_times[flight_id] < time:
                    planned_arrival_times[flight_id] = time

        #np.savetxt("20250826_converted_instance.csv", out, delimiter=",",fmt="%i")

        return out, planned_arrival_times
    
    @classmethod
    def instance_navpoint_matrix(cls, triplets, max_time, *, t0=0, fill_value=-1):
        """
        triplets: (N x 3) array-like of [flight_id, navpoint_id, time_first_reached]
        max_time: largest time index to include (inclusive)
        t0:       optional time offset (use t0=min_time if your times don't start at 0)
        """
        a = np.asarray(triplets, dtype=int)
        f = a[:, 0]
        n = a[:, 1]
        t = a[:, 2] - t0  # shift if needed so time starts at 0

        # Map possibly non-contiguous flight IDs to row indices
        flights, f_idx = np.unique(f, return_inverse=True)
        F = flights.size
        T = max_time - t0  # inclusive of max_time

        out = np.full((F, T), fill_value, dtype=int)

        # keep only times that land inside [0, T-1]
        m = (t >= 0) & (t < T)
        out[f_idx[m], t[m]] = n[m]  # last write wins if duplicates

        return out, flights  # 'flights' tells you which row corresponds to which flight_id

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
                #print(f"prev_vertex:{prev_vertex},vertex:{vertex}")
                distance = networkx_graph[prev_vertex][vertex]["weight"]

                # CONVERT SPEED TO m/s
                airplane_speed_ms = airplane_speed_kts * 0.51444

                # Compute duration from prev to vertex in unit time:
                duration_in_seconds = distance/airplane_speed_ms
                factor_to_unit_standard = 3600.00 / float(timestep_granularity)
                duration_in_unit_standards = math.ceil(duration_in_seconds / factor_to_unit_standard)

                if duration_in_unit_standards == 0:
                    duration_in_unit_standards = 1

                current_time = current_time + duration_in_unit_standards

                t_slot=current_time

                if t_slot >= flights_affected.shape[1]:
                    raise Exception("In optimize_flights max time exceeded current allowed time.")

            traj.append((flight_index, vertex, t_slot))

        return traj
    