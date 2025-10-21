
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
    
    def capacity_time_matrix(self,
                            cap: np.ndarray,
                            n_times: int,
                            time_granularity: int) -> np.ndarray:
        """
        cap: shape (N, >=2), capacity in column 1
        returns: (N, n_times)
        """
        N = cap.shape[0]
        T = int(time_granularity)

        # Integer math: base fill + remainder
        capacity = np.asarray(cap[:, 1], dtype=np.int64)
        base = capacity // T                 # per-slot baseline
        rem  = capacity %  T                 # how many +1 to sprinkle

        # Start with the baseline replicated across T columns
        # (broadcast then copy to get a writeable array)
        template = np.broadcast_to(base[:, None], (N, T)).astype(np.int32, copy=True)

        # Fast remainder placement: for each row i, add +1 at
        # columns: step[i] * np.arange(rem[i]), where step = floor(T/rem)
        max_r = int(rem.max())
        if max_r > 0:
            # step is irrelevant where rem==0, but we still fill an array (won't be used)
            step = np.empty_like(rem, dtype=np.int64)
            # Avoid division-by-zero; values where rem==0 are ignored by mask below
            np.floor_divide(T, rem, out=step, where=rem > 0)

            J = np.arange(max_r, dtype=np.int64)                  # 0..max(rem)-1
            mask = J[None, :] < rem[:, None]                      # N x max_r (True only for first rem[i])
            rows2d = np.broadcast_to(np.arange(N)[:, None], (N, max_r))
            cols2d = step[:, None] * J[None, :]

            r_idx = rows2d[mask]
            c_idx = cols2d[mask]

            # Scatter-add the remainders
            np.add.at(template, (r_idx, c_idx), 1)

        # Repeat the base block to cover n_times
        if n_times % T != 0:
            raise ValueError("n_times must be a multiple of time_granularity")
        reps = n_times // T

        cap_mat = np.tile(template, (1, reps))   # (N, n_times)

        #np.savetxt("20250819_cap_mat.csv", cap_mat, delimiter=",",fmt="%i")

        return cap_mat



    def convert_sectors(self, sectors, timestep_granularity, max_time, max_time_flights) -> List[str]:
        """Convert Capacity to ASP"""

        max_time_dim = int(max(max_time_flights, (max_time + 1) * timestep_granularity))
        sectors_matrix = self.capacity_time_matrix(sectors, max_time_dim, timestep_granularity)
        
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
        sectors_instance = self.convert_sectors(self.sectors, timestep_granularity, max_time, max_time_flights)
        airports_instance = self.convert_airports(self.airports)
        airplanes_instance = self.convert_airplanes(self.airplanes)
        airplane_flight_instance = self.convert_airplane_flight(self.airplane_flight)
        navaid_sector_instance = self.convert_navaid_sector(self.navaid_sector)

        sector_capactiy_factor_instance = [f"sectorCapacityFactor({sector_capactiy_factor})."]
        timestep_granularity_instance = [f"timestepGranularity({timestep_granularity})."]
        #max_time_instance = [f"maxTime({max_time*timestep_granularity})."]
        max_time_instance = [f"maxTime(30)."]

        instance = graph_instance + flights_instance + sectors_instance + airplanes_instance +\
            airports_instance + airplane_flight_instance + navaid_sector_instance +\
            timestep_granularity_instance + max_time_instance+\
            sector_capactiy_factor_instance

        return instance

