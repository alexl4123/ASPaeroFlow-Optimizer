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



def split_selection(values: List[str] | None) -> List[str] | None:
    """Flatten a repeatable, comma-separated CLI selection into a list of names.

    Returns None when nothing was selected, which every caller reads as "no filter".
    """
    if not values:
        return None
    names = [name.strip() for raw in values for name in raw.split(",")]
    return [name for name in names if name]


def select_systems(systems: List[Dict], wanted: List[str] | None) -> List[Dict]:
    """The named systems, in build_system_config()'s order. All of them if nothing is named.

    The order is the system list's, never the selection's, so naming systems in a different order
    cannot reorder the CSV columns. An unknown key raises instead of yielding an empty run: a typo
    in a job script would otherwise look like a solver that produced no results.
    """
    if wanted is None:
        return systems
    available = [system["key"] for system in systems]
    unknown = [key for key in wanted if key not in available]
    if unknown:
        raise ValueError(f"no such enabled system {unknown}; enabled here: {available}")
    keep = set(wanted)
    return [system for system in systems if system["key"] in keep]


def select_instances(instances: List[Path], wanted: List[str] | None) -> List[Path]:
    """The named instance folders, in directory order. All of them if nothing is named."""
    if wanted is None:
        return instances
    available = [inst.name for inst in instances]
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ValueError(f"no such instance folder: {unknown}")
    keep = set(wanted)
    return [inst for inst in instances if inst.name in keep]


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
                "--number-capacity-management-configs=2",
                "--sector-capacity-factor=6",
                "--convex-sectors=0",
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/01_ASPaeroFlow",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_01_ASPaeroFlow",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=1000000",
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
                "--number-capacity-management-configs=2",
                "--sector-capacity-factor=6",
                "--convex-sectors=0",
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/0A_ASPaeroFlow_NoConvex",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_0A_ASPaeroFlow_NoConvex",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
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
                "--number-capacity-management-configs=2",
                "--sector-capacity-factor=6",
                "--convex-sectors=0",
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/0B_Sector_NoReroute_NoDelay",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_0B_Sector_NoReroute_NoDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
                ]
        })

    if args.experiment_asp_aero_flow_nr_d != 0:
        system_config.append({
            "key": "0C_Sector_NoReroute_Delay",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=1",
                "--max-delay-per-iteration=5",
                "--capacity-management-enabled=True",
                "--number-capacity-management-configs=2",
                "--sector-capacity-factor=6",
                "--convex-sectors=0",
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/0C_Sector_NoReroute_Delay",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_0C_Sector_NoReroute_NoDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
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
                "--number-capacity-management-configs=2",
                "--sector-capacity-factor=6",
                "--convex-sectors=0",
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/0D_Sector_Reroute_NoDelay",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_0D_Sector_Reroute_NoDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
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
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/02_RerouteDelay",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_02_RerouteDelay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
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
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/0A_Reroute",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_2A_Reroute",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
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
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/03_DELAY",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_03_Delay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
                ]
        })


    if args.experiment_casa != 0:
        system_config.append({
            "key": "03A_CASA",
            "script": base_dir / "../01_ASPaeroFlow/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                "--max-explored-vertices=1",
                "--max-delay-per-iteration=10",
                "--capacity-management-enabled=False",
                "--number-capacity-management-configs=1",
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/03_DELAY",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_03_Delay",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--minimize-number-sectors=false",
                "--max-number-navpoints-per-sector=1000000",
                "--max-number-sectors=-2",
                "--max-considered-aircraft=1"
                ]
        })

    if args.experiment_mip != 0:
        system_config.append({
            "key": "04_MIP",
            "script": base_dir / "../04_MIP/main.py",
            "encoding": base_dir / "../01_ASPaeroFlow/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/04_MIP",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_05_MIP",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                ]
        })


    if args.experiment_all_asp_variants != 0:
        asp_index = 5
        asp_base_key = "ASP"
        asp_regulation_ground_delay = [0,1,2]
        asp_regulation_rerouting = [0,1,2]
        asp_regulation_DAC = [0,1,2]

        for ground_delay_regulation in asp_regulation_ground_delay:
            for rerouting_regulation in asp_regulation_rerouting:
                for dac_regulation in asp_regulation_DAC:

                    if ground_delay_regulation == 0:
                        ground_delay_flag = "nd"
                    elif ground_delay_regulation == 1:
                        ground_delay_flag = "dp"
                    elif ground_delay_regulation == 2:
                        ground_delay_flag = "d"

                    if rerouting_regulation == 0:
                        rerouting_flag = "nr"
                    elif rerouting_regulation == 1:
                        rerouting_flag = "rp"
                    elif rerouting_regulation == 2:
                        rerouting_flag = "r"

                    if dac_regulation == 0:
                        dac_flag = "ns"
                    elif dac_regulation == 1:
                        dac_flag = "sp"
                    elif dac_regulation == 2:
                        dac_flag = "s"

                    experiment_key = str(asp_index) + "_ASP_" + rerouting_flag + "_" + ground_delay_flag + "_" + dac_flag

                    system_config.append({
                        "key": experiment_key,
                        "script": base_dir / "../02_ASP/main.py",
                        "encoding": base_dir / "../02_ASP/encoding.lp",
                        "verbosity": None,
                        "cmd": [
                            f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/" + experiment_key,
                            f"--wandb-enabled={args.wandb_enabled}",
                            "--wandb-experiment-name-suffix=_" + experiment_key,
                            f"--wandb-experiment-name-prefix={experiment_name}_",
                            "--wandb-entity=thinklex",
                            "--regulation-ground-delay-active=" + str(ground_delay_regulation),
                            "--regulation-rerouting-active=" + str(rerouting_regulation),
                            "--regulation-dynamic-sectorization=" + str(dac_regulation),
                            ]
                    })
                    
                    asp_index += 1


    if args.experiment_asp_rp_dp_sp != 0:

        system_config.append({
            "key": "05_ASP_rp_dp_sp",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/05_ASP_rp_dp_sp",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_05_ASP_rp_dp_sp",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=1",
                "--regulation-rerouting-active=1",
                "--regulation-dynamic-sectorization=1",
                ]
        })

    if args.experiment_asp_rp_d_sp != 0:

        system_config.append({
            "key": "05_ASP_rp_d_sp",
            "script": base_dir / "../02_ASP/main.py",
            "encoding": base_dir / "../02_ASP/encoding.lp",
            "verbosity": None,
            "cmd": [
                f"--results-format={args.results_format}",
                f"--results-root={output_path}/solver_outputs/05_ASP_rp_d_sp",
                f"--wandb-enabled={args.wandb_enabled}",
                "--wandb-experiment-name-suffix=_05_ASP_rp_d_sp",
                f"--wandb-experiment-name-prefix={experiment_name}_",
                "--wandb-entity=thinklex",
                "--regulation-ground-delay-active=2",
                "--regulation-rerouting-active=1",
                "--regulation-dynamic-sectorization=1",
                ]
        })


    return system_config



# ---------------------------------------------
# Benchmarking logic
# ---------------------------------------------

#: --number-threads is passed to every system, and it means A DIFFERENT THING in each one:
#:
#:   02_ASP          clasp SEARCH threads. Dropped on the floor until this branch wired it to
#:                   clingo's --parallel-mode; every ASP run to date was therefore single-
#:                   threaded whatever this said. Now set to 1 EXPLICITLY, so the effective
#:                   configuration is exactly what produced the published LPNMR/ATMOS numbers
#:                   and the knob is real and available for a future solver-configuration sweep.
#:   01_ASPaeroFlow  `max_number_processors`, the outer Python loop's own processor accounting.
#:                   Not a clasp thread count. LEFT AT 5, unchanged.
#:   04_MIP          Gurobi's `model.Params.Threads` (04_MIP/mip_model.py:119). LEFT AT 5,
#:                   unchanged -- dropping Gurobi to one thread would change every MIP number in
#:                   the campaign, which is the opposite of the comparability this is protecting.
#:
#: So the count is per system rather than one shared literal. Only 02_ASP's entry is a clasp
#: setting; change the others only if you mean to change what those solvers do.
SOLVER_SEARCH_THREADS: Dict[str, int] = {"02_ASP": 1}
DEFAULT_SYSTEM_THREADS = 5


def thread_count_for(system: Dict) -> int:
    """--number-threads for one system. See SOLVER_SEARCH_THREADS for why it is not shared."""
    script = str(system.get("script", ""))
    for folder, count in SOLVER_SEARCH_THREADS.items():
        if folder in script:
            return count
    return DEFAULT_SYSTEM_THREADS


#: Solver folders whose main.py understands --solver-profile / --solver-arg. 03_Delay, 04_MIP
#: and CASA do not run clingo at all, and their argparse would reject the flag outright.
ASP_SOLVER_FOLDERS: Tuple[str, ...] = ("01_ASPaeroFlow", "02_ASP")


def solver_option_cli(system: Dict, args) -> List[str]:
    """The --solver-* flags this system should receive, if any.

    Returns [] for the default profile with no extra args, so a benchmark that does not ask for
    a solver strategy produces exactly the command line it always produced. That matters: the
    published results and any campaign in flight were produced by that command line.

    To run an ablation, invoke this script once per strategy INTO ITS OWN --output-dir:
        ./start_benchmark_caller.py <instances> --output-dir=<f>_bb
        ./start_benchmark_caller.py <instances> --output-dir=<f>_usc_domain --solver-profile=usc-domain
    """
    script = str(system.get("script", ""))
    if not any(folder in script for folder in ASP_SOLVER_FOLDERS):
        return []

    cli: List[str] = []
    if args.solver_profile != "default":
        cli.append(f"--solver-profile={args.solver_profile}")
    for raw in (args.solver_arg or []):
        cli.append(f"--solver-arg={raw}")
    # --solver-stats exists only on 02_ASP/main.py, which is where the JSON result line is built.
    if args.solver_stats == "True" and "02_ASP" in script:
        cli.append("--solver-stats=true")
    return cli


def build_command(system: Dict, paths: Dict[str, Path], python_bin: str, timestep_granularity, seed:int = 11904657, solver_cli: List[str] | None = None) -> List[str]:
    """Assemble the command‑line for one solver run."""
    cmd = [
        python_bin,
        "-u",
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
        f"--number-threads={thread_count_for(system)}",
    ]

    if system["verbosity"] is not None:
        cmd.append(f"--verbosity={system['verbosity']}")
    else:
        cmd.append(f"--verbosity=0")


    if system["encoding"] is not None:
        cmd.append(f"--encoding-path={system['encoding']}")

    # Empty unless --solver-profile / --solver-arg / --solver-stats were passed to this script.
    if solver_cli:
        cmd.extend(solver_cli)
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


#: The per-metric CSVs, in the order they are written. One column per system that reported the
#: metric at least once, plus "Instance".
RESULT_METRICS: List[str] = [
    "OVERLOAD", "ARRIVAL-DELAY", "SECTOR-NUMBER", "SECTOR-DIFF", "REROUTE", "RECONFIG",
    "COMPUTATION-FINISHED", "GROUNDING-TIME", "TOTAL-TIME-TO-THIS-POINT", "ERROR",
]


def write_result_csvs(
    output_path: Path,
    instance_names: List[str],
    system_names: List[str],
    exec_time: Dict[str, Dict],
    ram_usage: Dict[str, Dict],
    sol_value: Dict[str, Dict],
) -> None:
    """Write every result file for one problem directory: the CSVs and individual_outputs/.

    This is the tail of main(), lifted out unchanged so that merge_benchmark_shards.py can
    rebuild the same files from per-unit shards by CALLING it. Re-implementing it elsewhere
    would not be equivalent: sol_value_to_rows() discovers its own header while it walks the
    instances in order, so a metric that is missing on the first instance and present on a
    later one yields rows of differing length, and merged output would only match the
    monolithic output if that behaviour were reproduced exactly. Calling the same code is the
    only way to be sure it is.
    """
    header = ["Instance"] + list(system_names)

    def dicts_to_rows(container: Dict[str, Dict[str, float | int]]) -> List[List]:
        return [[inst] + [container[inst][name] for name in system_names] for inst in instance_names]

    def sol_value_to_rows(container, metric):

        own_heads = {}
        own_heads["Instance"] = 1

        output_list = []
        for inst in instance_names:
            tmp_list = [inst]

            for system_name in system_names:

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

    for metric in RESULT_METRICS:
        metric_values = sol_value_to_rows(sol_value, metric)
        write_csv(output_path / f"{metric.lower()}.csv", metric_values[0], metric_values[1])

    for inst in instance_names:
        for system_name in system_names:

            tmp_path_root = output_path / "individual_outputs"
            tmp_path = tmp_path_root / f"{inst}_{system_name}.json"

            tmp_path_root.mkdir(parents=True, exist_ok=True)

            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump({"object": sol_value[inst][system_name]}, fh)


# ---------------------------------------------
# CLI
# ---------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """The command line, as a parser rather than as a side effect of main().

    build_worklist.py needs to know which systems a given set of --experiment-* flags produces,
    and the only honest way to answer that is to parse the same flags and call the same
    build_system_config(). Returning the parser is what lets it do so without duplicating a
    single default.
    """
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

    # Weights & Biases was hard-coded on for every solver. That is fine for a handful of runs and
    # a problem for a full benchmark: 01_ASPaeroFlow/main.py RAISES FileNotFoundError when the
    # wandb.key file is missing, and wandb.login(relogin=True) needs network from the compute
    # node. Setting WANDB_MODE=disabled does not help, because the key file is checked before
    # wandb is consulted. Default stays True so existing scripts are unaffected.
    # A full benchmark writes one set of result matrices per (instance, solver). Uncompressed
    # those are ~38 MB for a DACH TG=60 run, i.e. ~2.9 TB across the full grid -- far past a
    # 100 GB quota. They compress ~254x (37.6 MB -> 0.15 MB) because they are highly repetitive,
    # so compressing is strictly better than switching them off: ~11 GB for the whole grid, and
    # the solutions stay verifiable. csv.gz is preferred over npz for published results because
    # pandas and every other tool read it directly.
    parser.add_argument("--results-format", type=str, default="csv",
                        choices=["csv", "csv.gz", "npz"],
                        help="Format for the per-run result matrices. Use csv.gz for large runs.")

    parser.add_argument("--wandb-enabled", type=str, default="True",
                        choices=["True", "False"],
                        help="Pass False for large unattended benchmark runs: no key file "
                             "needed, no network dependency, no thousands of logged runs. "
                             "The CSVs in --output-dir are the authoritative results either way.")

    parser.add_argument("--experiment-asp-aero-flow", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-no-convex", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-nr-nd", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-nr-d", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-aero-flow-r-nd", type=int, default=1, help="true (val!=0), false (val=0)")

    parser.add_argument("--experiment-casa", type=int, default=1, help="true (val!=0), false (val=0)")

    parser.add_argument("--experiment-route-delay", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-route", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-delay", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-mip", type=int, default=1, help="true (val!=0), false (val=0)")

    parser.add_argument("--experiment-all-asp-variants", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-rp-dp-sp", type=int, default=1, help="true (val!=0), false (val=0)")
    parser.add_argument("--experiment-asp-rp-d-sp", type=int, default=1, help="true (val!=0), false (val=0)")

    # LEGACY:
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

    # Clingo search strategy for the ASP systems (01_ASPaeroFlow, 02_ASP), selected by NAME so a
    # benchmark configuration does not carry raw solver flags. "default" passes nothing at all,
    # which is what every previous campaign did -- the published LPNMR/ATMOS numbers and any run
    # in flight are produced by that path. See common/clingo_options.py for the flag lists.
    #
    # "usc-domain" is the combination measured to solve instances in seconds that branch-and-bound
    # does not close in 30 minutes; it needs --heuristic=Domain for the encoding's #heuristic
    # directive to have any effect at all.
    #
    # NOTE when comparing across profiles at a TIMEOUT: core-guided (usc) search optimises the
    # objective levels roughly top-down and can report ONE model and then nothing for the rest of
    # the budget, while branch-and-bound improves every level together. A timed-out usc number and
    # a timed-out bb number are therefore not like-for-like. --solver-stats adds the solver's own
    # LOWER BOUND to each result line, which is what makes the two comparable.
    parser.add_argument("--solver-profile", type=str, default="default",
                        choices=["default", "usc", "domain", "usc-domain"],
                        help="Named clingo search configuration for the ASP systems. "
                             "'default' passes no flags (what every previous campaign did).")
    parser.add_argument("--solver-arg", type=str, action="append", default=None, metavar="FLAG",
                        help="Extra raw clingo flag for the ASP systems, repeatable, e.g. "
                             "--solver-arg=--parallel-mode=5. Appended after the profile's flags.")
    parser.add_argument("--solver-stats", type=str, default="False", choices=["True", "False"],
                        help="Add clingo diagnostics (cost vector, lower bound, models reported, "
                             "whether the search was exhausted) to 02_ASP's JSON result lines.")

    parser.add_argument(
        "--hot-start",
        action="store_true",
        help="Resume from existing hot-start JSON in output directory; skip completed (instance, solver) runs.",
    )

    # ---- selecting a SUBSET of the cross product ---------------------------------------
    # Without these, the smallest thing this script can be asked to do is one problem directory:
    # every instance times every enabled system, which is the tens-of-hours array task that
    # run_all_benchmarks.slurm submits. run_benchmark_units.slurm submits ONE (instance, system)
    # per invocation instead, and needs to name both.
    #
    # Absent, both are no-ops: the full instance list and the full system list are used, so every
    # existing command line behaves exactly as before.
    #
    # --only-system is not redundant with the --experiment-* flags. --experiment-all-asp-variants
    # switches TWENTY-SEVEN systems on or off in one flag (5_ASP_nr_nd_ns .. 31_ASP_r_d_s), so no
    # combination of those flags can isolate one of them. Filtering the built list by key can,
    # and it filters the list build_system_config() actually produced, so the key spelling and
    # the column order are the real ones rather than a second guess at them.
    parser.add_argument(
        "--only-instance", type=str, action="append", default=None, metavar="NAME",
        help="Run only this instance folder. Repeatable, and accepts a comma-separated list. "
             "Order follows the instance directory, not the order given here.",
    )
    parser.add_argument(
        "--only-system", type=str, action="append", default=None, metavar="KEY",
        help="Run only this solver system, by the key that heads its CSV column (e.g. 04_MIP, "
             "05_ASP_rp_dp_sp, 7_ASP_nr_dp_ns). Repeatable, and accepts a comma-separated list. "
             "Applied AFTER the --experiment-* flags, so the system must also be enabled by them.",
    )
    

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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

    try:
        systems = select_systems(systems, split_selection(args.only_system))
    except ValueError as exc:
        print(f"[ERROR] --only-system: {exc}", file=sys.stderr)
        sys.exit(1)
    if not systems:
        print("[ERROR] No solver systems enabled -- every --experiment-* flag is 0", file=sys.stderr)
        sys.exit(1)

    # Collect & sort instances
    instances = sorted(p for p in args.instance_dir.iterdir() if p.is_dir())
    if not instances:
        print(f"[ERROR] No instance folders found in {args.instance_dir}", file=sys.stderr)
        sys.exit(1)

    # Same contract as --only-system: directory order survives, unknown names are an error.
    try:
        instances = select_instances(instances, split_selection(args.only_instance))
    except ValueError as exc:
        print(f"[ERROR] --only-instance: {exc} in {args.instance_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Result containers
    exec_time: Dict[str, Dict[str, float | int]] = {inst.name: {} for inst in instances}
    ram_usage: Dict[str, Dict[str, float | int]] = {inst.name: {} for inst in instances}
    sol_value: Dict[str, Dict[str, float | int]] = {inst.name: {} for inst in instances}

    # Remember first failure per system to skip later instances
    first_failure: Dict[str, str | None] = {sys_["key"]: None for sys_ in systems}

    timestep_granularity = args.timestep_granularity

    # Say which solver configuration produced these numbers, so the log identifies the run.
    _solver_desc = args.solver_profile
    if args.solver_arg:
        _solver_desc += " + " + " ".join(args.solver_arg)
    print(f"[config] clingo solver profile: {_solver_desc}"
          f"{' (+ --solver-stats)' if args.solver_stats == 'True' else ''}", flush=True)

    # ---- progress accounting -------------------------------------------------------------
    # The only previous output was "[system] instance: running ...", with no outcome, no
    # counters and no timing -- unreadable over SSH on a run this size. We now emit one line
    # per finished (instance, solver) with position, wall time, outcome and a running ETA, and
    # append the same to progress.jsonl so the aggregator can summarise across array tasks.
    _total_runs = len(instances) * len(systems)
    _done_runs = 0
    _t_start = time.time()
    _progress_path = Path(output_path) / "progress.jsonl"
    _progress_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_hms(seconds: float) -> str:
        seconds = int(max(0, seconds))
        return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def _note_progress(inst_name, system_name, rt, peak, sol, reused=False):
        nonlocal _done_runs
        _done_runs += 1
        elapsed = time.time() - _t_start
        eta = (elapsed / _done_runs) * (_total_runs - _done_runs) if _done_runs else 0.0
        # The outcome lives in the LAST output dict's "ERROR" key, not in the runtime.
        # run_process always returns a float runtime -- the codes never appear there -- so
        # keying off rt silently reported every timeout and memout as "ok".
        err = None
        if isinstance(sol, list) and sol and isinstance(sol[-1], dict):
            err = sol[-1].get("ERROR")
        elif isinstance(sol, str):
            err = sol
        outcome = {TIMEOUT_CODE: "TIMEOUT", MEMOUT_CODE: "MEMOUT",
                   ERROR_CODE: "ERROR", UNPARSE_CODE: "UNPARSED"}.get(err, "ok")
        last = sol[-1] if isinstance(sol, list) and sol and isinstance(sol[-1], dict) else {}
        obj = " ".join(f"{k.lower()}={last[k]}" for k in ("OVERLOAD", "ARRIVAL-DELAY")
                       if k in last) or "-"
        rt_str = f"{rt:8.1f}s" if outcome == "ok" else " " * 9
        print(f"[{_done_runs:>5}/{_total_runs}] {(_done_runs / _total_runs) * 100:5.1f}%  "
              f"{system_name:<28} {inst_name:<22} {outcome:<8} {rt_str} "
              f"{obj}  elapsed {_fmt_hms(elapsed)}  eta {_fmt_hms(eta)}"
              f"{'  (hot-start)' if reused else ''}", flush=True)
        try:
            with _progress_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "instance": inst_name, "system": system_name, "outcome": outcome,
                    "runtime_s": rt, "ram_mb": peak, "objective": sol,
                    "error_code": err,
                    "done": _done_runs, "total": _total_runs, "reused": reused,
                }) + "\n")
        except OSError:
            pass   # progress logging must never take the benchmark down



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
                    # Count reused results too, or a resumed run (e.g. the MIP-only second
                    # pass, where almost everything comes from hot-start) reports a position
                    # and ETA computed from a handful of fresh runs.
                    _note_progress(inst_name, system_name,
                                   exec_time[inst_name][system_name],
                                   ram_usage[inst_name][system_name],
                                   sol_value[inst_name][system_name], reused=True)
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


            cmd = build_command(system, paths, args.python_bin, timestep_granularity,
                                solver_cli=solver_option_cli(system, args))
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

            _note_progress(inst_name, system_name,
                           exec_time[inst_name][system_name],
                           ram_usage[inst_name][system_name],
                           sol_value[inst_name][system_name], reused=False)

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
    write_result_csvs(output_path,
                      [inst.name for inst in instances],
                      [sys_["key"] for sys_ in systems],
                      exec_time, ram_usage, sol_value)

    print("Benchmarking finished. Results written to:")
    for fn in ("execution_time.csv", "ram_usage.csv", "solution_value.csv"):
        print("  -", output_path / fn)

if __name__ == "__main__":
    main()
