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
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import psutil

# ---------------------------------------------
# Exit‑code conventions for the CSVs
# ---------------------------------------------
TIMEOUT_CODE = 'T'  # time limit hit
MEMOUT_CODE = 'M'  # memory limit hit
ERROR_CODE = 'E'   # any other error / non-zero output
UNPARSE_CODE = 'P'   #  unparsable output

# ---------------------------------------------
# Helpers
# ---------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def load_hotstart_state(state_path: Path) -> Dict:
    """
    Load hot-start progress from JSON.
    Structure:
      {
        "schema_version": 1,
        "created_utc": "...",
        "updated_utc": "...",
        "records": {
          "<instance_name>": {
            "<system_name>": {
              "instance_path": "...",
              "system_name": "...",
              "execution_time": <float|str>,
              "ram_usage": <int|str>,
              "solution_value": <int|str>,
              "experiment_failed": <0|str>,
              "timestamp_utc": "..."
            }
          }
        }
      }
    """
    if not state_path.exists():
        return {
            "schema_version": 1,
            "created_utc": _utc_now_iso(),
            "updated_utc": _utc_now_iso(),
            "records": {},
        }
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("hot-start state must be a JSON object")
        if "records" not in data or not isinstance(data["records"], dict):
            data["records"] = {}
        data.setdefault("schema_version", 1)
        data.setdefault("created_utc", _utc_now_iso())
        data["updated_utc"] = _utc_now_iso()
        return data
    except Exception:
        # If the file is corrupted (e.g., partial write during outage), move it aside and start fresh.
        try:
            backup = state_path.with_suffix(state_path.suffix + f".corrupt.{int(time.time())}")
            state_path.replace(backup)
        except Exception:
            pass
        return {
            "schema_version": 1,
            "created_utc": _utc_now_iso(),
            "updated_utc": _utc_now_iso(),
            "records": {},
        }


def save_hotstart_state_atomic(state_path: Path, state: Dict) -> None:
    """Atomic-ish write: write to temp file in same dir, then os.replace()."""
    state["updated_utc"] = _utc_now_iso()
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, state_path)


def hotstart_get(state: Dict, inst_name: str, system_name: str) -> Dict | None:
    return state.get("records", {}).get(inst_name, {}).get(system_name)


def hotstart_set(
    state: Dict,
    inst_name: str,
    inst_path: Path,
    system_name: str,
    execution_time,
    ram_usage,
    solution_value,
) -> None:
    records = state.setdefault("records", {})
    per_inst = records.setdefault(inst_name, {})
    failed = 0 if solution_value not in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE, UNPARSE_CODE) else solution_value
    per_inst[system_name] = {
        "instance_path": str(inst_path.resolve()),
        "system_name": system_name,
        "execution_time": execution_time,
        "ram_usage": ram_usage,
        "solution_value": solution_value,
        "experiment_failed": failed,
        "timestamp_utc": _utc_now_iso(),
    }



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

def build_system_config(base_dir: Path, output_path:Path, experiment_name:str, args) -> List[Dict]:
    """Return the list with per‑solver metadata."""
    system_config = []

    if args.experiment_asp_aero_flow != 0:
        system_config.append({
            "key": "01_ASPaeroFlow",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=3",
                "--max-delay-per-iteration=5",
                "--capacity-management-enabled=True",
                "--number-capacity-management-configs=7",
                "--sector-capacity-factor=6",
                "--convex-sectors=1",
                f"--results-root={output_path}/solver_outputs/01_ASPaeroFlow",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_01_ASPaeroFlow",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_asp_aero_flow_no_convex != 0:
        system_config.append({
            "key": "0A_ASPaeroFlow_NoConvex",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=3",
                "--max-delay-per-iteration=5",
                "--capacity-management-enabled=True",
                "--number-capacity-management-configs=7",
                "--sector-capacity-factor=6",
                "--convex-sectors=0",
                f"--results-root={output_path}/solver_outputs/01_ASPaeroFlow",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_0A_ASPaeroFlow_NoConvex",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_asp_aero_flow_nr_nd != 0:
        system_config.append({
            "key": "0B_Sector_NoReroute_NoDelay",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=1",
                "--max-delay-per-iteration=1",
                "--capacity-management-enabled=True",
                "--number-capacity-management-configs=7",
                "--sector-capacity-factor=6",
                "--convex-sectors=1",
                f"--results-root={output_path}/solver_outputs/01_ASPaeroFlow",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_0B_Sector_NoReroute_NoDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_asp_aero_flow_nr_d != 0:
        system_config.append({
            "key": "0C_Sector_NoReroute_NoDelay",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=1",
                "--max-delay-per-iteration=5",
                "--capacity-management-enabled=True",
                "--number-capacity-management-configs=7",
                "--sector-capacity-factor=6",
                "--convex-sectors=1",
                f"--results-root={output_path}/solver_outputs/01_ASPaeroFlow",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_0C_Sector_NoReroute_NoDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_asp_aero_flow_r_nd != 0:
        system_config.append({
            "key": "0D_Sector_Reroute_NoDelay",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=3",
                "--max-delay-per-iteration=1",
                "--capacity-management-enabled=True",
                "--number-capacity-management-configs=7",
                "--sector-capacity-factor=6",
                "--convex-sectors=1",
                f"--results-root={output_path}/solver_outputs/01_ASPaeroFlow",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_0D_Sector_Reroute_NoDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })




    if args.experiment_route_delay != 0:
        system_config.append({
            "key": "02_RerouteDelay",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=3",
                "--max-delay-per-iteration=5",
                "--capacity-management-enabled=False",
                "--number-capacity-management-configs=1",
                f"--results-root={output_path}/solver_outputs/02_RerouteDelay",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_02_RerouteDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_route != 0:
        system_config.append({
            "key": "2A_Reroute",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=3",
                "--max-delay-per-iteration=1",
                "--capacity-management-enabled=False",
                "--number-capacity-management-configs=1",
                f"--results-root={output_path}/solver_outputs/02_RerouteDelay",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_2A_Reroute",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_delay != 0:
        system_config.append({
            "key": "03_DELAY",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=1",
                "--max-delay-per-iteration=5",
                "--capacity-management-enabled=False",
                "--number-capacity-management-configs=1",
                f"--results-root={output_path}/solver_outputs/03_DELAY",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_03_Delay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=true",
                "--max-number-navpoints-per-sector=100",
                ]
        })

    if args.experiment_mip != 0:
        system_config.append({
            "key": "04_MIP",
            "script": base_dir / "../04_MIP/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/04_MIP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_05_MIP",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                ]
        })

    if args.experiment_asp_r_d_s != 0:

        system_config.append({
            "key": "05_ASP_r_d_s",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_05_ASP_r_d_s",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=True",
                "--regulation-rerouting-active=True",
                "--regulation-dynamic-sectorization=2",
                ]
        })

    if args.experiment_asp_r_d_ns != 0:
        system_config.append({
            "key": "06_ASP_r_d_ns",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_06_ASP_r_d_ns",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=True",
                "--regulation-rerouting-active=True",
                "--regulation-dynamic-sectorization=0",
                ]
        })

    if args.experiment_asp_r_nd_s != 0:
        system_config.append({
            "key": "07_ASP_r_nd_s",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_07_ASP_r_nd_s",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=False",
                "--regulation-rerouting-active=True",
                "--regulation-dynamic-sectorization=2",
                ]
        })

    if args.experiment_asp_nr_d_s != 0:
        system_config.append({
            "key": "08_ASP_nr_d_s",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_08_ASP_nr_d_s",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=True",
                "--regulation-rerouting-active=False",
                "--regulation-dynamic-sectorization=2",
                ]
        })

    if args.experiment_asp_r_nd_ns != 0:
        system_config.append({
            "key": "09_ASP_r_nd_ns",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_09_ASP_r_nd_ns",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=False",
                "--regulation-rerouting-active=True",
                "--regulation-dynamic-sectorization=0",
                ]
        })

    if args.experiment_asp_nr_nd_s != 0:
        system_config.append({
            "key": "10_ASP_nr_nd_s",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_10_ASP_nr_nd_s",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=False",
                "--regulation-rerouting-active=False",
                "--regulation-dynamic-sectorization=2",
                ]
        })

    if args.experiment_asp_nr_d_ns != 0:
        system_config.append({
            "key": "11_ASP_nr_d_ns",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_11_ASP_nr_d_ns",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=True",
                "--regulation-rerouting-active=False",
                "--regulation-dynamic-sectorization=0",
                ]
        })

    if args.experiment_asp_nr_nd_ns != 0:
        system_config.append({
            "key": "12_ASP_nr_nd_ns",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_12_ASP_nr_nd_ns",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=False",
                "--regulation-rerouting-active=False",
                "--regulation-dynamic-sectorization=0",
                "--allow-overloads=True",
                ]
        })

    if args.experiment_asp_r_d_sp != 0:
        system_config.append({
            "key": "13_ASP_r_d_sp",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_13_ASP_r_d_sp",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=True",
                "--regulation-rerouting-active=True",
                "--regulation-dynamic-sectorization=1",
                "--allow-overloads=True",
                ]
        })

    if args.experiment_asp_nr_d_sp != 0:
        system_config.append({
            "key": "14_ASP_nr_d_sp",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_14_ASP_nr_d_sp",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=True",
                "--regulation-rerouting-active=False",
                "--regulation-dynamic-sectorization=1",
                "--allow-overloads=True",
                ]
        })

    if args.experiment_asp_r_nd_sp != 0:
        system_config.append({
            "key": "15_ASP_r_nd_sp",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_15_ASP_r_nd_sp",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=False",
                "--regulation-rerouting-active=True",
                "--regulation-dynamic-sectorization=1",
                "--allow-overloads=True",
                ]
        })

    if args.experiment_asp_nr_nd_sp != 0:
        system_config.append({
            "key": "16_ASP_nr_nd_sp",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-root={output_path}/solver_outputs/02_ASP",
                "--wandb-enabled=True",
                "--wandb-experiment-name-suffix=_16_ASP_nr_nd_sp",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=False",
                "--regulation-rerouting-active=False",
                "--regulation-dynamic-sectorization=1",
                "--allow-overloads=True",
                ]
        })

    return system_config



# ---------------------------------------------
# Benchmarking logic
# ---------------------------------------------

def build_command(system: Dict, paths: Dict[str, Path], python_bin: str, timestep_granularity, seed:int = 11904657) -> List[str]:
    """Assemble the command‑line for one solver run."""
    cmd = [
        python_bin,
        str(system["script"].resolve()),
        f"--graph-path={paths['graph-edges']}",
        f"--sectors-path={paths['sectors']}",
        f"--flights-path={paths['flights']}",
        f"--airports-path={paths['airports']}",
        f"--airplanes-path={paths['airplanes']}",
        f"--airplane-flight-path={paths['airplane-flight']}",
        f"--navaid-sector-path={paths['navaid-sector']}",
        f"--seed={seed}",
        f"--timestep-granularity={timestep_granularity}",
        "--number-threads=5",
    ]

    if system["verbosity"] is not None:
        cmd.append(f"--verbosity={system['verbosity']}")
    else:
        cmd.append(f"--verbosity=0")


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
    except subprocess.TimeoutExpired as e:

        stdout = e.stdout or ""
        stderr = e.stderr or ""

        kill_descendants(proc.pid)
        kill_process_tree(proc.pid)

        
        try: 
            out2, err2 = proc.communicate()
            stdout += out2 or ""
            stderr += err2 or ""
        except Exception:
            pass
        
        output = []
        for line in stdout.splitlines():
            try:
                output.append(json.loads(line))
            except:
                pass

        runtime = time.perf_counter() - start

        if len(output) > 0:
            output[-1]["ERROR"] = TIMEOUT_CODE
        else:
            tmp_dict = {}
            tmp_dict["ERROR"] = TIMEOUT_CODE
            output.append(tmp_dict)

        return runtime, peak["value"], output

    
    print(stderr)

    runtime = time.perf_counter() - start
    output = []
    for line in stdout.splitlines():
        try:
            output.append(json.loads(line))
        except:
            pass

    # Memory limit hit?
    if mem_exceeded.is_set():

        if len(output) > 0:
            output[-1]["ERROR"] = MEMOUT_CODE
        else:
            tmp_dict = {}
            tmp_dict["ERROR"] = MEMOUT_CODE
            output.append(tmp_dict)

        return runtime, peak["value"], output


    if proc.returncode != 0:

        if len(output) > 0:
            output[-1]["ERROR"] = ERROR_CODE
        else:
            tmp_dict = {}
            tmp_dict["ERROR"] = ERROR_CODE
            output.append(tmp_dict)

        return runtime, peak["value"], output

    # Parse solution (first line of stdout)
    print("-------------")
    print(f"Overload:{output[-1]['OVERLOAD']}, Arrival Delay:{output[-1]['ARRIVAL-DELAY']}, Sector-Number: {output[-1]['SECTOR-NUMBER']}, Sector-Diff: {output[-1]['SECTOR-DIFF']}, Reroute: {output[-1]['REROUTE']}, Reconfig: {output[-1]['RECONFIG']}")
    print("==============")

    if len(output) > 0:
        output[-1]["ERROR"] = ""
    else:
        tmp_dict = {}
        tmp_dict["ERROR"] = ""
        output.append(tmp_dict)

    return runtime, peak["value"], output


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
    parser.add_argument("--memory-limit", type=int, default=20, help="Memory limit (GiB)")
    parser.add_argument("--python-bin", default="/home/guests/abeiser/miniconda3/envs/potassco/bin/python", help="Python interpreter for the solvers")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where to place CSVs")
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Where to place CSVs")
    parser.add_argument("--timestep-granularity", type=int, default=1, help="Timestep granularity")
    parser.add_argument("--experiment-name", type=str, default="", help="Specify an experiment name for various settings (such as wandb).")

    parser.add_argument("--scaling-experiments", type=int, default=0, help="true (val!=0), false (val=0)")

    parser.add_argument("--experiment-asp-aero-flow", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-no-convex", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-nr-nd", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-nr-d", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-r-nd", type=int, default=1, help="true (val!=0), false (val=0)")


    parser.add_argument("--experiment-route-delay", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-route", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-delay", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-mip", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-r-d-s", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-r-d-ns", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-r-nd-s", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-nr-d-s", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-r-nd-ns", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-nr-nd-s", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-nr-d-ns", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-nr-nd-ns", type=int, default=1, help="true (val!=0), false (val=0)")

    parser.add_argument("--experiment-asp-r-d-sp", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-nr-d-sp", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-r-nd-sp", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-nr-nd-sp", type=int, default=1, help="true (val!=0), false (val=0)")

    parser.add_argument(
        "--hot-start",
        action="store_true",
        help="Resume from existing hot-start JSON in output directory; skip completed (instance, solver) runs.",
    )
    

    args = parser.parse_args()
    mem_limit_bytes = args.memory_limit * (1024 ** 3)

    experiment_name = args.experiment_name

    output_root = args.output_root
    output_dir = args.output_dir
    output_path = Path(output_root, output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    hot_state = None
    hot_state_path = output_path / "hotstart_state.json"
    if args.hot_start:
        hot_state = load_hotstart_state(hot_state_path)
        save_hotstart_state_atomic(hot_state_path, hot_state)

    scaling_experiments = args.scaling_experiments
    if scaling_experiments == 0:
        scaling_experiments = False
    else:
        scaling_experiments = True

    base_dir = Path(__file__).resolve().parent
    systems = build_system_config(base_dir, output_path, experiment_name, args)

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
    first_failure: Dict[str, str | None] = {sys_["key"]: None for sys_ in systems}

    timestep_granularity = args.timestep_granularity


    # Main loops (solver outermost ⇒ better CPU cache locality, easier skip logic)
    for inst_path in instances:
        inst_name = inst_path.name
        for system in systems:
            system_name = system["key"]

            # Hot-start: reuse stored results if present
            if hot_state is not None:
                rec = hotstart_get(hot_state, inst_name, system_name)
                if rec is not None:
                    exec_time[inst_name][system_name] = rec.get("execution_time")
                    ram_usage[inst_name][system_name] = rec.get("ram_usage")
                    sol_value[inst_name][system_name] = rec.get("solution_value")
                    # Ensure scaling skip logic continues correctly after resuming
                    if (
                        scaling_experiments is True
                        and first_failure.get(system_name) is None
                        and sol_value[inst_name][system_name] in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE, UNPARSE_CODE)
                    ):
                        first_failure[system_name] = sol_value[inst_name][system_name]
                    continue


            # Propagate previous failure without running anything
            if system_name in first_failure and first_failure[system_name] is not None and scaling_experiments is True:
                exec_time[inst_name][system_name] = first_failure[system_name]
                ram_usage[inst_name][system_name] = first_failure[system_name]
                sol_value[inst_name][system_name] = first_failure[system_name]
                if hot_state is not None:
                    hotstart_set(
                        hot_state,
                        inst_name,
                        inst_path,
                        system_name,
                        exec_time[inst_name][system_name],
                        ram_usage[inst_name][system_name],
                        sol_value[inst_name][system_name],
                    )
                    save_hotstart_state_atomic(hot_state_path, hot_state)
                continue

            # Required files
            f_edges = inst_path / "graph_edges.csv"
            f_capacity = inst_path / "sectors.csv"
            f_instance = inst_path / "flights.csv"
            f_airplanes = inst_path / "airplanes.csv"
            f_airport = inst_path / "airports.csv"
            f_airplane_flight = inst_path / "airplane_flight_assignment.csv"
            f_navaid_sector = inst_path / "navaid_sector_assignment.csv"


            paths = {
                "graph-edges": f_edges,
                "sectors": f_capacity,
                "flights": f_instance,
                "airports": f_airport,
                "airplanes": f_airplanes,
                "airplane-flight": f_airplane_flight,
                "navaid-sector": f_navaid_sector,
            }


            cmd = build_command(system, paths, args.python_bin, timestep_granularity)
            cmd += system["cmd"]

            #print(" ".join(cmd))
            #continue

            print(f"[{system_name}] {inst_name}: running …", flush=True)
            rt, peak, sol = run_process(cmd, args.time_limit, mem_limit_bytes)

            # Store results (convert runtime to seconds with 3 decimals, memory to MiB int)
            exec_time[inst_name][system_name] = round(rt, 3) if rt not in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE, UNPARSE_CODE) else rt
            ram_usage[inst_name][system_name] = int(peak // (1024 ** 2)) if peak not in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE, UNPARSE_CODE) else peak
            sol_value[inst_name][system_name] = sol

            if rt in (TIMEOUT_CODE, MEMOUT_CODE, ERROR_CODE, UNPARSE_CODE):
                first_failure[system_name] = sol

            # Persist progress after every (instance, solver) is decided (run or failure-propagated)
            if hot_state is not None:
                hotstart_set(
                    hot_state,
                    inst_name,
                    inst_path,
                    system_name,
                    exec_time[inst_name][system_name],
                    ram_usage[inst_name][system_name],
                    sol_value[inst_name][system_name],
                )
                save_hotstart_state_atomic(hot_state_path, hot_state)

    # -----------------------------------------
    # Write CSVs
    # -----------------------------------------
    header = ["Instance"] + [s["key"] for s in systems]

    def dicts_to_rows(container: Dict[str, Dict[str, float | int]]) -> List[List]:
        return [[inst] + [container[inst][s["key"]] for s in systems] for inst in (p.name for p in instances)]

    def sol_value_to_rows(container, metric):
        
        own_heads = {}
        own_heads["Instance"] = 1

        output_list = []
        for inst in (p.name for p in instances):
            tmp_list = [inst]

            for system in systems:

                system_name = system["key"]

                final_sol_dict = container[inst][system_name][-1]

                if metric in final_sol_dict:
                    if system_name not in own_heads:
                        own_heads[system_name] = 1

                    tmp_list.append(final_sol_dict[metric])

                else:
                    if system_name in own_heads:
                        tmp_list.append(-1)

            output_list.append(tmp_list)

        return own_heads, output_list


    write_csv(output_path / "execution_time.csv", header, dicts_to_rows(exec_time))
    write_csv(output_path / "ram_usage.csv", header, dicts_to_rows(ram_usage))
    
    for metric in ["OVERLOAD", "ARRIVAL-DELAY", "SECTOR-NUMBER", "SECTOR-DIFF", "REROUTE", "RECONFIG", "COMPUTATION-FINISHED","GROUNDING-TIME","TOTAL-TIME-TO-THIS-POINT","ERROR"]:
        metric_values = sol_value_to_rows(sol_value, metric)
        write_csv(output_path / f"{metric.lower()}.csv", metric_values[0], metric_values[1])

    for inst in (p.name for p in instances):
        tmp_list = [inst]
        for system in systems:
            system_name = system["key"]

            tmp_path_root = output_path / "individual_outputs"
            tmp_path = tmp_path_root / f"{inst}_{system_name}.json"
            
            json_object_list = []
            for json_val_line in sol_value[inst][system_name]:
                json_object_list.append(json.dumps(json_val_line))

            json_object = "{\"object\":[" + ','.join(json_object_list) + "}]}"

            tmp_path_root.mkdir(parents=True, exist_ok=True)

            with tmp_path.open("w", encoding="utf-8") as fh:
                fh.write(json_object)

    print("Benchmarking finished. Results written to:")
    for fn in ("execution_time.csv", "ram_usage.csv", "solution_value.csv"):
        print("  -", output_path / fn)

if __name__ == "__main__":
    main()
