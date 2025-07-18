#!/usr/bin/env python3
"""
start_benchmarks.py – Benchmark runner for the Air Traffic Flow & Capacity Management (ATFCM) implementations.

Given a directory that contains sub‑folders with individual ATFCM instances, this script
runs four different solvers (01_ASPaeroFlow, 02_ASP, 03_Delay, 04_MIP) **sequentially**
for each instance, measures

* wall‑clock runtime
* peak RAM consumption (process + all descendants)
* solution value (first line of stdout)

and writes three CSV files:

    execution_time.csv   # seconds (‑1 timeout, ‑2 memout, ‑3 error)
    ram_usage.csv        # MiB     (‑1 timeout, ‑2 memout, ‑3 error)
    solution_value.csv   # integer (‑1 timeout, ‑2 memout, ‑3 error)

If the first failure for a solver is reached, the remaining (larger) instances are
skipped for that solver and marked with the same failure code.

The script is intended to live in the folder 06_benchmark_start_script and is typically
invoked via something like:

    nohup taskset --cpu-list 2-3 ./start_benchmarks.py ../05_instances/instances_005_005 \
        --time-limit 1800 --memory-limit 5 &> logs/instances_005_005.log &

Dependencies:  psutil  (install with  pip install psutil)
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import psutil

# ---------------------------------------------
# Exit‑code conventions for the CSVs
# ---------------------------------------------
TIMEOUT_CODE = -1  # time limit hit
MEMOUT_CODE = -2  # memory limit hit
ERROR_CODE = -3   # any other error / non‑zero return‑code / unparsable output

# ---------------------------------------------
# Helpers
# ---------------------------------------------

def get_recursive_memory_usage(pid: int) -> int:
    """Return RSS usage (bytes) of *pid* + all recursive children."""
    try:
        proc = psutil.Process(pid)
        mem = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return mem
    except psutil.NoSuchProcess:
        return 0


def kill_process_tree(pid: int) -> None:
    """Best‑effort termination of a process group (parent + children)."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def kill_descendants(pid: int):
    """Recursively terminate all descendants of a process."""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):  # Get all descendants
            try:
                child.kill()  # Kill each descendant
            except Exception as ex:
                pass
        parent.kill()  # Kill the parent itself, if still alive
    except psutil.NoSuchProcess:
        pass  # Process already terminated

# ---------------------------------------------
# System configuration (edit paths here if your layout differs)
# ---------------------------------------------

def build_system_config(base_dir: Path) -> List[Dict]:
    """Return the list with per‑solver metadata."""
    return [
        {
            "key": "01_ASPaeroFlow",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
        },
        {
            "key": "02_ASP",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
        },
        {
            "key": "03_Delay",
            "script": base_dir / "../03_Delay/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
        },
        {
            "key": "04_MIP",
            "script": base_dir / "../04_MIP/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
        },
    ]


# ---------------------------------------------
# Benchmarking logic
# ---------------------------------------------

def build_command(system: Dict, paths: Dict[str, Path], python_bin: str) -> List[str]:
    """Assemble the command‑line for one solver run."""
    cmd = [
        python_bin,
        str(system["script"].resolve()),
        f"--path-graph={paths['edges']}",
        f"--path-capacity={paths['capacity']}",
        f"--path-instance={paths['instance']}",
        f"--airport-vertices-path={paths['airport']}",
        "--seed=11904657",
        "--timestep-granularity=1",
        "--number-threads=1",
        "--max-explored-vertices=6",
        "--max-delay-per-iteration=-1",
        "--max-time=24",
        "--verbosity=0",
    ]
    if system["encoding"] is not None:
        cmd.append(f"--encoding-path={system['encoding']}")
    return cmd


def run_process(cmd: List[str], time_limit: int, mem_limit_bytes: int) -> Tuple[int | float, int | float, int | float]:
    """Execute *cmd* under limits and return (runtime, peak_mem_bytes, solution)."""

    peak: Dict[str, int] = {"value": 0}
    mem_exceeded = threading.Event()

    def monitor(pid: int) -> None:
        while psutil.pid_exists(pid) and not mem_exceeded.is_set():
            usage = get_recursive_memory_usage(pid)
            if usage > peak["value"]:
                peak["value"] = usage
            if usage > mem_limit_bytes:
                mem_exceeded.set()
                kill_descendants(pid)
                kill_process_tree(pid)
                break
            time.sleep(0.5)

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,  # start in new process‑group so we can SIGKILL everything
    )

    monitor_thread = threading.Thread(target=monitor, args=(proc.pid,), daemon=True)
    monitor_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=time_limit)
    except subprocess.TimeoutExpired:
        kill_descendants(proc.pid)
        kill_process_tree(proc.pid)
        return TIMEOUT_CODE, TIMEOUT_CODE, TIMEOUT_CODE

    print(stderr)

    # Memory limit hit?
    if mem_exceeded.is_set():
        return MEMOUT_CODE, MEMOUT_CODE, MEMOUT_CODE

    runtime = time.perf_counter() - start

    if proc.returncode != 0:
        return ERROR_CODE, ERROR_CODE, ERROR_CODE

    # Parse solution (first line of stdout)
    first_line = stdout.splitlines()[0].strip() if stdout else ""
    try:
        solution_val = int(first_line)
    except ValueError:
        solution_val = ERROR_CODE

    return runtime, peak["value"], solution_val


# ---------------------------------------------
# CSV helpers
# ---------------------------------------------

def write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------
# CLI
# ---------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ATFCM benchmark driver")
    parser.add_argument("instance_dir", type=Path, help="Folder containing instance sub‑directories")
    parser.add_argument("--time-limit", type=int, default=1800, help="Wall‑clock limit (s)")
    parser.add_argument("--memory-limit", type=int, default=5, help="Memory limit (GiB)")
    parser.add_argument("--python-bin", default="/home/thinklex/miniconda3/envs/potassco/bin/python", help="Python interpreter for the solvers")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where to place CSVs")

    args = parser.parse_args()
    mem_limit_bytes = args.memory_limit * 1024 ** 3

    base_dir = Path(__file__).resolve().parent
    systems = build_system_config(base_dir)

    # Collect & sort instances
    instances = sorted(p for p in args.instance_dir.iterdir() if p.is_dir())
    if not instances:
        print(f"[ERROR] No instance folders found in {args.instance_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Result containers
    exec_time: Dict[str, Dict[str, float | int]] = {inst.name: {} for inst in instances}
    ram_usage: Dict[str, Dict[str, float | int]] = {inst.name: {} for inst in instances}
    sol_value: Dict[str, Dict[str, float | int]] = {inst.name: {} for inst in instances}

    # Remember first failure per system to skip later instances
    first_failure: Dict[str, int | None] = {sys_["key"]: None for sys_ in systems}


    # Main loops (solver outermost ⇒ better CPU cache locality, easier skip logic)
    for inst_path in instances:
        inst_name = inst_path.name
        for system in systems:
            system_name = system["key"]

            # Propagate previous failure without running anything
            if system_name in first_failure and first_failure[system_name] is not None:
                exec_time[inst_name][system_name] = first_failure[system_name]
                ram_usage[inst_name][system_name] = first_failure[system_name]
                sol_value[inst_name][system_name] = first_failure[system_name]
                continue

            # Required files
            f_edges = inst_path / "edges.csv"
            f_capacity = inst_path / "capacity.csv"
            f_instance = inst_path / "instance.csv"
            f_airport = inst_path / "airports.csv"

            paths = {
                "edges": f_edges,
                "capacity": f_capacity,
                "instance": f_instance,
                "airport": f_airport,
            }

            cmd = build_command(system, paths, args.python_bin)
            print(f"[{system_name}] {inst_name}: running …", flush=True)
            rt, peak, sol = run_process(cmd, args.time_limit, mem_limit_bytes)

            # Store results (convert runtime to seconds with 3 decimals, memory to MiB int)
            exec_time[inst_name][system_name] = round(rt, 3) if rt not in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE) else rt
            ram_usage[inst_name][system_name] = int(peak // (1024 ** 2)) if peak not in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE) else peak
            sol_value[inst_name][system_name] = sol

            if sol in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE):
                first_failure[system_name] = sol

    # -----------------------------------------
    # Write CSVs
    # -----------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = ["Instance"] + [s["key"] for s in systems]

    def dicts_to_rows(container: Dict[str, Dict[str, float | int]]) -> List[List]:
        return [[inst] + [container[inst][s["key"]] for s in systems] for inst in (p.name for p in instances)]

    write_csv(args.output_dir / "execution_time.csv", header, dicts_to_rows(exec_time))
    write_csv(args.output_dir / "ram_usage.csv", header, dicts_to_rows(ram_usage))
    write_csv(args.output_dir / "solution_value.csv", header, dicts_to_rows(sol_value))

    print("Benchmarking finished. Results written to:")
    for fn in ("execution_time.csv", "ram_usage.csv", "solution_value.csv"):
        print("  -", args.output_dir / fn)


if __name__ == "__main__":
    main()
