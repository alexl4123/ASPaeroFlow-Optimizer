#!/usr/bin/env python3
"""
checker.py

Simple checker for ASPaeroFlow experiment output.

Usage:
    python checker.py experiment_output/0005233_SEED11904657
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import math

import numpy as np

# Expected CSV files in the experiment output directory
FILENAMES = {
    "capacity_time_matrix": "capacity_time_matrix.csv",
    "converted_instance_matrix": "converted_instance_matrix.csv",
    "navaid_sector_time_assignment": "navaid_sector_time_assignment.csv",
    "converted_navpoint_matrix": "converted_navpoint_matrix.csv",
}


def load_csv_as_array(path: Path) -> np.ndarray:
    """
    Load a CSV file into a NumPy array.

    Adjust 'skip_header' or 'dtype' here if your files have headers or
    specific types.
    """
    try:
        arr = np.genfromtxt(path, delimiter=",", dtype=int) 
    except OSError as e:
        raise SystemExit(f"Error reading {path}: {e}")
    if arr.size == 0:
        raise SystemExit(f"Error: file {path} seems to be empty or unreadable.")
    return arr


def run_checks(
    capacity_time_matrix: np.ndarray,
    converted_instance_matrix: np.ndarray,
    navaid_sector_time_assignment: np.ndarray,
    converted_navpoint_matrix: np.ndarray,
) -> None:
    """
    Add your checker logic here.

    All four CSVs are already loaded as NumPy arrays.

    Example: replace the prints below with assertions / consistency checks.
    """
    print("=== Dummy checks (replace with your own) ===")
    print("capacity_time_matrix shape:         ", capacity_time_matrix.shape)
    print("converted_instance_matrix shape:    ", converted_instance_matrix.shape)
    print("navaid_sector_time_assignment shape:", navaid_sector_time_assignment.shape)
    print("converted_navpoint_matrix shape:    ", converted_navpoint_matrix.shape)



    print("-- RUN SIMPLIFIED MATRIX SHAPE CHECKS --")
    if capacity_time_matrix.shape != navaid_sector_time_assignment.shape:
        print(f"[ERROR] - Capacity-time and navaid-time matrix shapes do not coincide.")

    if converted_instance_matrix.shape != converted_navpoint_matrix.shape:
        print(f"[ERROR] - Converted-Instance and converted-navpoint shapes do not coincide.")
    print("[CHECK] -> SIMPLIFIED MATRIX SHAPE DONE")

    print("-- RUN SIMPLIFIED FLIGHT CHECKS --")
    for index in range(converted_instance_matrix.shape[0]):

        flight = converted_instance_matrix[index,:]

        indices = (np.nonzero(flight != -1))[0]
        
        for time_index in range(indices[0], indices[-1]+1):

            if converted_instance_matrix[index,time_index] == -1:
                print(time_index)
                print(f"[ERROR]: flight:{index} at time:{time_index} has a -1 sector (but is en-route)")

    print("[CHECK] -> SIMPLIFIED FLIGHT CHECKS DONE")

    print("-- RUN SIMPLIFIED NAVAID-SECTOR CHECKS")

    for navaid in range(navaid_sector_time_assignment.shape[0]):
        
        prev_sector = None

        for time in range(navaid_sector_time_assignment.shape[1]):

            sector = navaid_sector_time_assignment[navaid,time]

            if prev_sector is None:
                prev_sector = sector

            elif prev_sector != sector:
                pass
                #print(f"-> {navaid}@{time} NAVAID-SECTOR CHANGE: {prev_sector}->{sector}")

            prev_sector = sector

    number_sectors = None

    for time in range(navaid_sector_time_assignment.shape[1]):

        u = np.unique(navaid_sector_time_assignment[:,time])
        
        if number_sectors is None:
            number_sectors = len(u)
            print(f"[SEC]@{time} - Number sectors changed to: {number_sectors}")
        elif number_sectors != len(u):
            number_sectors = len(u)
            print(f"[SEC]@{time} - Number sectors changed to: {number_sectors}")


    print("[CHECK] -> SIMPLIFIED NAVPOINT-SECTOR INSPECTION DONE")


    print("-- RUN SIMPLIFIED NAVAID-FLIGHT CHECKS")

    for flight_id in range(converted_navpoint_matrix.shape[0]): 

        time_indices = list(np.nonzero(converted_navpoint_matrix[flight_id,:] != -1)[0])

        for flight_hop_index in range(len(time_indices)):

            time = time_indices[flight_hop_index]                
            navaid = converted_navpoint_matrix[flight_id, time]

            if flight_hop_index == 0:
                sector = navaid_sector_time_assignment[navaid,time]

                if converted_instance_matrix[flight_id,time] != sector:
                    print(f"[ERRORA] - SECTOR MISMATCH in NAVAID/FLIGHT (initial): FlightID:{flight_id}, TIME:{time}, navaid:{navaid}: sector-data:{converted_instance_matrix[flight_id,time]} != sector-comp.:{sector}")

            else:

                prev_time = time_indices[flight_hop_index - 1]                
                prev_navaid = converted_navpoint_matrix[flight_id, prev_time]

                for time_index in range(1, time-prev_time + 1):

                    if time_index <= math.floor((time - prev_time)/2):
                        prev_sector = navaid_sector_time_assignment[prev_navaid,prev_time + time_index]


                        if converted_instance_matrix[flight_id,prev_time + time_index] != prev_sector:
                            print(f"[ERRORB] - SECTOR MISMATCH in NAVAID/FLIGHT (<=): FlightID:{flight_id}, TIME:{prev_time+time_index}, prev-time:{prev_time}, next-time:{time}, prev_navaid:{prev_navaid}, navaid:{navaid}: sector-data:{converted_instance_matrix[flight_id,prev_time+time_index]} != sector-prev:{prev_sector}")


                    else:
                        sector = navaid_sector_time_assignment[navaid,prev_time + time_index]
                        if converted_instance_matrix[flight_id,prev_time + time_index] != sector:
                            print(f"[ERRORC] - SECTOR MISMATCH in NAVAID/FLIGHT (>): FlightID:{flight_id}, TIME:{prev_time+time_index}, prev-time:{prev_time}, next-time:{time}, prev_navaid:{prev_navaid}, navaid:{navaid}: sector-data:{converted_instance_matrix[flight_id,prev_time+time_index]} != sector-comp.:{sector}")

                        #converted_instance_matrix[flight_id,prev_time + time_index] = sector
    
    
    print("[CHECK] -> CHECK SIMPLIFIED NAVAID-FLIGHT DONE")



    print("-- RUN SIMPLIFIED CAPACITY CHECKS")
    

    for time_index in range(navaid_sector_time_assignment.shape[1]):

        for navaid_index in range(navaid_sector_time_assignment.shape[0]):
            if navaid_sector_time_assignment[navaid_index, time_index] == navaid_index:
                # THIS IS A SECTOR IF NAVAIDINDEX == ASSIGNMENT

                flights = np.nonzero(converted_instance_matrix[:,time_index] == navaid_index)[0]

                if len(flights) > capacity_time_matrix[navaid_index,time_index]:
                    print(flights)
                    print(f"[ERROR] - Capacity overload at sector:{navaid_index}, time:{time_index}")

    print("[CHECK2]")

    for time_index in range(navaid_sector_time_assignment.shape[1]):

        sectors = np.unique(navaid_sector_time_assignment[:,time_index])

        for sector in sectors:

            navaids = np.nonzero(navaid_sector_time_assignment[:,time_index] == sector)[0]
                
            actual_flights = []

            for navaid in navaids:
                flights = np.nonzero(converted_navpoint_matrix[:,time_index] == navaid)[0]
                actual_flights += list(flights)

            if len(actual_flights) > capacity_time_matrix[sector,time_index]:
                print(f"[ERROR2] - Capacity overload at sector:{navaid_index}, time:{time_index}")


            flights = np.nonzero(converted_instance_matrix[:,time_index] == sector)[0]
            if len(flights) > capacity_time_matrix[sector,time_index]:
                print(flights)
                print(f"[ERROR3] - Capacity overload at sector:{sector}, time:{time_index}")

    print("[CHECK] RUN SIMPLIFIED CAPACITY CHECKS DONE")
    print("[DEBUG] CHECKS")

    for time_index in range(navaid_sector_time_assignment.shape[1]):

        sectors = np.unique(navaid_sector_time_assignment[:,time_index])
        sec0 = []
        for sector in sectors:
            sec0.append(int(sector))

        sec1 = []
        for navaid_index in range(navaid_sector_time_assignment.shape[0]):
            if navaid_sector_time_assignment[navaid_index, time_index] == navaid_index:
                sec1.append(int(navaid_index))
        
        if sec0 != sec1:
            print("SECTOR COMPUTATION DIFFERS!")
       
        
        """
        for navaid_index in range(navaid_sector_time_assignment.shape[0]):
            if navaid_sector_time_assignment[navaid_index, time_index] == navaid_index:

                flights = np.nonzero(converted_instance_matrix[:,time_index] == navaid_index)[0]

        """
    print("[DEBUG] DONE")



    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple checker for ASPaeroFlow experiment output."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to experiment output directory "
             "(e.g., experiment_output/0005233_SEED11904657)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir

    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory or does not exist.", file=sys.stderr)
        sys.exit(1)

    # Build paths for all expected files
    paths = {key: output_dir / fname for key, fname in FILENAMES.items()}

    # Check that all files exist
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        print("Error: missing expected CSV files:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    # Load CSVs into NumPy arrays
    capacity_time_matrix = load_csv_as_array(paths["capacity_time_matrix"])
    converted_instance_matrix = load_csv_as_array(paths["converted_instance_matrix"])

    navaid_sector_time_assignment = load_csv_as_array(
        paths["navaid_sector_time_assignment"]
    )
    converted_navpoint_matrix = load_csv_as_array(paths["converted_navpoint_matrix"])

    # Call your checker
    run_checks(
        capacity_time_matrix,
        converted_instance_matrix,
        navaid_sector_time_assignment,
        converted_navpoint_matrix,
    )


if __name__ == "__main__":
    main()
