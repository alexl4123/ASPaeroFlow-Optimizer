
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
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=1)
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

                navpoint_edges.append(f"navpointEdge({x0},{x1},{speed},{unit_distance}).")

        return navpoint_edges


    def convert_flights(self, flights) -> List[str]:
        """Convert Instance to ASP"""

        max_time = 0 
        asp_atoms = []
        for row_index in range(flights.shape[0]):
            # ASP expects flightPlan(<ID>,<TIME>,<LOCATION>)
            asp_atoms.append(f"navpointFlightPlan({flights[row_index,0]},{flights[row_index,2]},{flights[row_index,1]}).")

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
 



    def convert_sectors(self, sectors, timestep_granularity, max_time, max_time_flights, navaid_sector_time_assignment) -> List[str]:
        """Convert Capacity to ASP"""

        max_time_dim = int(max(max_time_flights, (max_time + 1) * timestep_granularity, navaid_sector_time_assignment.shape[1]))

        sectors_matrix = TranslateCSVtoLogicProgram.capacity_time_matrix(sectors, max_time_dim, timestep_granularity, navaid_sector_time_assignment)
        
        asp_atoms = []
        for row_index in range(sectors_matrix.shape[0]):
            for column_index in range(sectors_matrix.shape[1]):
                # ASP expects flightPlan(<ID>,<TIME>,<LOCATION>)
                asp_atoms.append(f"atomic_sector({row_index},{column_index},{sectors_matrix[row_index,column_index]}).")

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
            airplane_instance.append(f"airplane({airplanes[row_index,0]},{airplanes[row_index,1]}).")

        return airplane_instance

    def convert_airplane_flight(self, airplane_flight) -> List[str]:
        """Convert airplane_flight to ASP"""
        
        airplane_flight_instance = []

        for row_index in range(airplane_flight.shape[0]):
            airplane_flight_instance.append(f"airplane_flight({airplane_flight[row_index,0]},{airplane_flight[row_index,1]}).")

        return airplane_flight_instance

    def convert_navaid_sector(self, navaid_sector) -> List[str]:
        """Convert navaid_sector to ASP"""
        
        navaid_sector_instance = []

        for row_index in range(navaid_sector.shape[0]):
            navaid_sector_instance.append(f"navaid_sector({navaid_sector[row_index,0]},{navaid_sector[row_index,1]},0).")

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

    def main(self, graph_path: str, flights_path: str, sectors_path: str, airports_path: str, airplanes_path, airplane_flight_path,
             navaid_sector_path, encoding_path, timestep_granularity: int, max_time,
             sector_capactiy_factor) -> None:
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
        sectors_instance = self.convert_sectors(self.sectors, timestep_granularity, max_time, max_time_flights, navaid_sector_time_assignment)
        airports_instance = self.convert_airports(self.airports)
        airplanes_instance = self.convert_airplanes(self.airplanes)
        airplane_flight_instance = self.convert_airplane_flight(self.airplane_flight)
        navaid_sector_instance = self.convert_navaid_sector(self.navaid_sector)

        sector_capactiy_factor_instance = [f"sectorCapacityFactor({sector_capactiy_factor})."]
        timestep_granularity_instance = [f"timestepGranularity({timestep_granularity})."]
        max_time_instance = [f"maxTime({max_time*timestep_granularity})."]
        #max_time_instance = [f"maxTime(30)."]

        instance = graph_instance + flights_instance + sectors_instance + airplanes_instance +\
            airports_instance + airplane_flight_instance + navaid_sector_instance +\
            timestep_granularity_instance + max_time_instance+\
            sector_capactiy_factor_instance
    
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


