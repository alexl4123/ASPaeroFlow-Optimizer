#!/usr/bin/env python3
"""
checker.py

Simple checker for ASPaeroFlow experiment output.

Usage:
    python checker.py experiment_output/0005233_SEED11904657
    python checker.py experiment_output/0005233_SEED11904657 --original_data_dir original_input_folder/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import math
import csv
from collections import defaultdict

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

def load_flights_csv(path: Path) -> dict[int, list[tuple[int, int]]]:
    """
    Load flights.csv into a dict:
        flight_id -> [(time, position), ...] sorted by time (ascending)

    flights.csv format:
        Flight_ID,Position,Time
        0,27,1
        ...
    """
    if not path.is_file():
        raise SystemExit(f"Error: {path} does not exist.")

    flights: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Flight_ID", "Position", "Time"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"Error: {path} must contain header with columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            try:
                fid = int(row["Flight_ID"])
                pos = int(row["Position"])
                t = int(row["Time"])
            except Exception as e:
                raise SystemExit(f"Error parsing {path} row {row}: {e}")
            flights[fid].append((t, pos))

    # sort + basic sanity
    for fid, seq in flights.items():
        seq.sort(key=lambda x: x[0])
        # check strictly increasing time (common expectation)
        for i in range(1, len(seq)):
            if seq[i][0] == seq[i - 1][0]:
                print(f"[WARN] flights.csv has duplicate time for Flight_ID={fid} at t={seq[i][0]}")
            if seq[i][0] < seq[i - 1][0]:
                # should not happen due to sort, but keep the guard
                raise SystemExit(f"Error: non-monotone time sequence for Flight_ID={fid}")

    return dict(flights)


def run_filed_vs_actual_flight_checks(
    flights_filed: dict[int, list[tuple[int, int]]],
    converted_navpoint_matrix: np.ndarray,
) -> None:
    """
    Compare filed flights (flights.csv) against the converted_navpoint_matrix (|F|x|T|).

    Checks:
      (1) Flight existence by row index (= Flight_ID assumption) and matching endpoints:
          - first/last filed Position must match first/last non -1 in converted_navpoint_matrix[Flight_ID,:]
      (2) Compute and report delay:
          delay = actual_arrival_time - filed_arrival_time
          where actual_arrival_time is the column index of last non -1, converted to the same time base as flights.csv.
    """
    print("-- RUN FILED VS ACTUAL FLIGHT CHECKS (flights.csv vs converted_navpoint_matrix) --")

    if not flights_filed:
        print("[WARN] flights.csv appears empty; skipping filed-vs-actual checks.")
        return

    # Determine time base from flights.csv (often 1-based, sometimes 0-based)
    all_times = [t for seq in flights_filed.values() for (t, _) in seq]
    filed_min_time = 0

    # We map matrix column index -> "time" by: time = index + filed_min_time
    # This aligns common cases: filed_min_time=1 (sample) => col0 is time=1; filed_min_time=0 => col0 is time=0.
    missing_rows = 0
    endpoint_mismatches = 0
    delays: list[int] = []

    for fid, seq in flights_filed.items():
        if fid < 0 or fid >= converted_navpoint_matrix.shape[0]:
            print(f"[ERROR] filed Flight_ID={fid} has no corresponding row in converted_navpoint_matrix.")
            missing_rows += 1
            continue

        filed_dep_time, filed_dep_pos = seq[0]
        filed_arr_time, filed_arr_pos = seq[-1]

        row = converted_navpoint_matrix[fid, :]
        idxs = np.nonzero(row != -1)[0]
        if idxs.size == 0:
            print(f"[ERROR] Flight_ID={fid}: converted_navpoint_matrix row has no flown navpoints (all -1).")
            endpoint_mismatches += 1
            continue

        actual_dep_pos = int(row[idxs[0]])
        actual_arr_pos = int(row[idxs[-1]])

        if actual_dep_pos != filed_dep_pos or actual_arr_pos != filed_arr_pos:
            endpoint_mismatches += 1
            print(
                f"[ERROR] Flight_ID={fid}: endpoint mismatch "
                f"(filed dep/arr pos={filed_dep_pos}->{filed_arr_pos}, "
                f"actual dep/arr pos={actual_dep_pos}->{actual_arr_pos})."
            )

        actual_arr_time = int(idxs[-1]) + filed_min_time
        delay = actual_arr_time - filed_arr_time
        delays.append(int(delay))

        # Output per-flight delay line if non-zero (or if endpoint mismatch already reported above)
        if delay != 0:
            print(
                f"[DELAY] Flight_ID={fid}: filed_arr_time={filed_arr_time}, "
                f"actual_arr_time={actual_arr_time} -> delay={delay}"
            )

    if delays:
        d = np.array(delays, dtype=int)
        print("[SUMMARY] Filed vs actual:")
        print(f"  filed flights checked:        {len(flights_filed)}")
        print(f"  missing matrix rows:          {missing_rows}")
        print(f"  endpoint mismatches:          {endpoint_mismatches}")
        print(f"  delay stats (time units):     min={int(d.min())}, median={int(np.median(d))}, mean={float(d.mean()):.3f}, max={int(d.max())}")
        print(f"  flights with delay > 0:       {int((d > 0).sum())}")
        print(f"  flights with delay < 0:       {int((d < 0).sum())}  (check time-base assumptions if unexpected)")
        print(f"  total delay (sum over flights): {int(d.sum())}")
    else:
        print("[WARN] No delays computed (no valid flights compared).")

    print("[CHECK] -> FILED VS ACTUAL FLIGHT CHECKS DONE")



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
    parser.add_argument(
        "--original_data_dir",
        type=Path,
        default=None,
        help=(
            "Optional: path to original input data folder containing flights.csv "
            "(e.g., folder with flights.csv, airports.csv, graph_edges.csv, ...). "
            "If provided, runs filed-vs-actual flight checks."
        ),
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

    # Optional: compare against original filed flights (flights.csv)
    if args.original_data_dir is not None:
        flights_path = args.original_data_dir / "flights.csv"
        try:
            flights_filed = load_flights_csv(flights_path)
        except SystemExit:
            raise
        run_filed_vs_actual_flight_checks(flights_filed, converted_navpoint_matrix)
 


if __name__ == "__main__":
    main()
