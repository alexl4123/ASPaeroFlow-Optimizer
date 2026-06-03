import os
import shutil
import csv
import math
from pathlib import Path

def generate_relative_capacity_instances(
    base_dir: str, 
    nominal_caps_file: str, 
    output_root_dir: str, 
    time_granularity: int = 4
):
    base_path = Path(base_dir)
    out_root_path = Path(output_root_dir)
    
    # Load nominal capacities
    # Maps flight_instance -> { column_header: nominal_capacity }
    nominal_caps = {}
    with open(nominal_caps_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            flight_inst = row[0]
            caps = {headers[i]: float(row[i]) for i in range(1, len(headers))}
            nominal_caps[flight_inst] = caps

    percentages = [round(x * 0.1, 1) for x in range(1, 11)]

    print(nominal_caps)

    # Infer capacity prefix (e.g., 'CAP20') from base_dir to match CSV headers
    cap_prefix = base_path.name

    for pct in percentages:
        # e.g., output_root_dir/PCAP010/
        pct_out_dir = out_root_path / f"PCAP{int(pct * 100):03d}"
        
        for graph_dir in base_path.iterdir():
            if not graph_dir.is_dir():
                continue
                
            # Reconstruct CSV header format, e.g., "CAP20/00-0-.../"
            #csv_header_key = f"{cap_prefix}/{graph_dir.name}/"
            csv_header_key = f"{graph_dir.name}/"
            
            for flight_dir in graph_dir.iterdir():
                if not flight_dir.is_dir():
                    continue
                    
                flight_inst = flight_dir.name
                
                # Retrieve nominal capacity
                try:
                    nom_cap = nominal_caps[flight_inst][csv_header_key]
                except KeyError:
                    print(f"Warning: Key {csv_header_key} or flight {flight_inst} not found. Skipping.")
                    continue
                
                # Calculate new capacity
                new_cap = math.ceil(nom_cap * pct * time_granularity)
                
                # Create destination directory structure
                dest_dir = pct_out_dir / graph_dir.name / flight_dir.name
                shutil.copytree(flight_dir, dest_dir, dirs_exist_ok=True)
                
                # Overwrite sectors.csv
                sectors_csv_path = dest_dir / "sectors.csv"
                _overwrite_sectors_csv(sectors_csv_path, new_cap)

def _overwrite_sectors_csv(filepath: Path, new_capacity: int):
    # Read existing sectors to maintain Sector_IDs
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        sector_ids = [row[0] for row in reader if row]
        
    # Write new capacities
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s_id in sector_ids:
            writer.writerow([s_id, new_capacity])

if __name__ == "__main__":
    # Example usage
    generate_relative_capacity_instances(
        base_dir="../05_instances/20260513_TG4",
        nominal_caps_file="nominal_capacities.csv",
        output_root_dir="output",
        time_granularity=4
    )
