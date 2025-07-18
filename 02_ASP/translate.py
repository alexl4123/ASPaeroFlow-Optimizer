
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
from typing import Iterable, List

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


    def convert_edges(self, rows: Iterable[List[str]]) -> List[str]:
        """Convert Edges to ASP"""
        
        asp_edges = []
        for row in rows:
            asp_edges.append(f"sectorEdge({','.join(row)}).")

        return asp_edges

    def convert_instance(self, rows: Iterable[List[str]]) -> List[str]:
        """Convert Instance to ASP"""

        max_time = 0 
        asp_atoms = []
        for row in rows:
            # ASP expects flightPlan(<ID>,<TIME>,<LOCATION>)
            asp_atoms.append(f"flightPlan({row[0]},{row[2]},{row[1]}).")

            if int(row[2]) > max_time:
                max_time = int(row[2])

        return asp_atoms, max_time

    def convert_capacity(self, rows: Iterable[List[str]]) -> List[str]:
        """Convert Capacity to ASP"""
        
        asp_atoms = []
        for row in rows:
            # ASP expects flightPlan(<ID>,<TIME>,<LOCATION>)
            asp_atoms.append(f"sector({','.join(row)}).")

        return asp_atoms
    

    def convert_airport_vertices(self, rows: Iterable[List[str]]) -> List[str]:
        """Convert Airport Vertices to ASP"""
        
        airport_instance = []

        for row in rows:
            airport_instance.append(f"airport({row[0]}).")

        return airport_instance

    def main(self, edges_file: str, instance_file: str, capacity_file: str, airport_file: str, timestep_granularity: int) -> None:
        """Read and print rows from the three required CSV files."""
        base_dir = pathlib.Path.cwd()

        files = {
            "edges": edges_file,
            "instance": instance_file,
            "capacity": capacity_file,
            "airport_vertices": airport_file,
        }
        graph_iterable = self.read_csv(files["edges"])
        instance_iterable = self.read_csv(files["instance"])
        capacity_iterable = self.read_csv(files["capacity"])
        airport_vertices_iterable = self.read_csv(files["airport_vertices"])

        output_list = self.convert_edges(graph_iterable)
        output_list_temp, max_time = self.convert_instance(instance_iterable)
        output_list += output_list_temp
        output_list += self.convert_capacity(capacity_iterable)
        output_list += self.convert_airport_vertices(airport_vertices_iterable)
        output_list.append(f"timestepGranularity({timestep_granularity}).")

        return output_list, max_time

if __name__ == "__main__":
    if len(sys.argv) != 5:
        script_name = pathlib.Path(sys.argv[0]).name
        print(
            f"Usage: python {script_name} <EDGES.csv> <INSTANCE_X.csv> <CAPACITY_X.csv>\n\n" "<X> should be a natural number identifying the instance/ capacity files."
        )
        sys.exit(1)

    edges_file = sys.argv[1]
    instance_file = sys.argv[2]
    capacity_file = sys.argv[3]
    airport_vertices = sys.argv[4]

    TranslateCSVtoLogicProgram().main(edges_file, instance_file, capacity_file, airport_vertices)

