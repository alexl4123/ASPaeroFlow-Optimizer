import os
import subprocess
import csv
from pathlib import Path

# Configuration
PYTHON_EXEC = os.path.expanduser("~/miniconda3/envs/potassco/bin/python")
BASE_DATA_DIR = Path("../05_instances/20260513_TG4")
OUTPUT_CSV = "nominal_capacities.csv"

OLD_INSTANCES = [
    "04-0-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-50",
    "04-1-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-600-CLUSTERSIZE-50",
    "05-0-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-200",
    "05-1-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-600-CLUSTERSIZE-200",
    "06-0-USA-MAINLAND-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-800",
    "06-1-USA-MAINLAND-2019-06-01--2019-06-30-CAP-ENROUTE-600-CLUSTERSIZE-800",

]

# Target subset of instance groups (truncate or expand as needed)
OLD2_INSTANCE_GROUPS = [
    "CAP10/00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1",
    "CAP20/00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1",
    "CAP10/01-0-USA-EAST-COAST-20x10-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-2",
    "CAP20/01-0-USA-EAST-COAST-20x10-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-2",
    "CAP10/02-0-MAJOR-EUROPE-40x20-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-8",
    "CAP20/02-0-MAJOR-EUROPE-40x20-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-8",
    "CAP10/03-0-EAST-ASIA-40x40-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-16",
    "CAP20/03-0-EAST-ASIA-40x40-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-16",
    "CAP10/04-0-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-50-GABRIEL-GRAPH",
    "CAP20/04-0-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-50-GABRIEL-GRAPH",
    "CAP10/05-0-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-200-GABRIEL-GRAPH",
    "CAP20/05-0-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-200-GABRIEL-GRAPH",
    "CAP10/06-0-USA-MAINLAND-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-800-GABRIEL-GRAPH",
    "CAP20/06-0-USA-MAINLAND-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-800-GABRIEL-GRAPH",
    "CAP10/08-0-CENTRAL-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-30",
    "CAP20/08-0-CENTRAL-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-30",
]

INSTANCE_GROUPS = [
    "00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1",
    "01-0-USA-EAST-COAST-20x10-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-2",
    "02-0-MAJOR-EUROPE-40x20-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-8",
    "03-0-EAST-ASIA-40x40-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-16",
    "04-0-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-50-GABRIEL-GRAPH",
    "05-0-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-200-GABRIEL-GRAPH",
    "06-0-USA-MAINLAND-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-800-GABRIEL-GRAPH",
    "08-0-CENTRAL-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-30",
]

def main():
    results = {}  # Structure: results[seed_dir][instance_group] = capacity
    all_seeds = set()

    for ig in INSTANCE_GROUPS:
        ig_path = BASE_DATA_DIR / ig
        if not ig_path.exists():
            print(f"Warning: Directory not found -> {ig_path}")
            continue

        for seed_dir in os.listdir(ig_path):
            seed_path = ig_path / seed_dir
            if not seed_path.is_dir():
                continue
            
            all_seeds.add(seed_dir)
            if seed_dir not in results:
                results[seed_dir] = {}

            cmd = [
                PYTHON_EXEC, "main.py",
                f"--data-dir={seed_path}/",
                "--controller-enabled=false",
                "--wandb-enabled=false",
                "--timestep-granularity=4",
                "--encoding-path=encoding.lp"
            ]

            try:
                # Capture standard output
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Extract the last line to filter out potential unexpected warnings
                capacity = process.stdout.strip().split('\n')[-1]
                results[seed_dir][ig] = capacity
            except subprocess.CalledProcessError as e:
                print(f"Error processing {seed_path}: {e.stderr}")
                results[seed_dir][ig] = "ERR"

    # Sort seeds for deterministic CSV output
    sorted_seeds = sorted(list(all_seeds))

    # Write to CSV
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = [""] + [f"{ig}/" for ig in INSTANCE_GROUPS]
        writer.writerow(header)
        
        # Write rows
        for seed in sorted_seeds:
            row = [seed] + [results[seed].get(ig, "") for ig in INSTANCE_GROUPS]
            writer.writerow(row)

if __name__ == "__main__":
    main()
