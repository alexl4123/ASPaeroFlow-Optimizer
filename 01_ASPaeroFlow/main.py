#!/usr/bin/env python3
"""A minimal, extensible template for loading CSV inputs via the command line.

The script accepts (optional) paths to three CSV files:
    --path-graph     Path to the graph description
    --path-capacity  Path to capacity information
    --path-instance  Path to an instance specification

Each file (when provided) is loaded into a NumPy array for subsequent processing.
Extend the :py:meth:`Main.run` method to add your application logic.
"""
from __future__ import annotations

import argparse
import os
import json

import zmq

import numpy as np

from pathlib import Path
from typing import Any, Final, List, Optional, Callable, Dict
from datetime import datetime, timezone

from src.aspaeroflow.main_optimizer_loop import Main
from src.aspaeroflow.optimize_flights import MAX, TRIANGULAR, LINEAR

# ---------------------------------------------------------------------------
# CLI utilities (with config + bundle directory support)
# ---------------------------------------------------------------------------


DEFAULT_FILENAMES = {
    "graph_path":              "graph_edges.csv",
    "sectors_path":            "sectors.csv",
    "flights_path":            "flights.csv",
    "airports_path":           "airports.csv",
    "airplanes_path":          "airplanes.csv",
    "airplane_flight_path":    "airplane_flight_assignment.csv",
    "navaid_sector_path":      "navaid_sector_assignment.csv",
    "encoding_path":           "encoding.lp",
    "wandb_api_key":               "wandb.key",
}

def _cfg_get(cfg: Dict, key: str, default=None):
    """Get key from cfg with hyphen/underscore tolerance."""
    if key in cfg: return cfg[key]
    alt = key.replace("-", "_")
    if alt in cfg: return cfg[alt]
    alt2 = key.replace("_", "-")
    if alt2 in cfg: return cfg[alt2]
    return default

def _preparse(argv: Optional[List[str]]):
    """Parse only --config and --data-dir early, so we can load config defaults."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    return p.parse_known_args(argv)

def _build_arg_parser(cfg: Dict) -> argparse.ArgumentParser:
    """Build parser with defaults coming from cfg (if present)."""
    def C(key, default=None): return _cfg_get(cfg, key, default)

    parser = argparse.ArgumentParser(
        prog="ATFM-NM-Tool",
        description="Highly efficient ATFM problem solver for the network manager - including XAI.",
    )

    # New: config + bundle directory
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON config file whose values serve as defaults (overridden by CLI)."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=C("data-dir", None),
        help="Directory containing the 7 standard optimizer CSVs. "
             "Any individual --*-path not provided will default to this directory + default filename."
    )

    # Individual paths (CLI overrides config)
    parser.add_argument("--graph-path",           type=Path, metavar="FILE", default=C("graph-path", None),
                        help="Location of the graph CSV file (graph_edges.csv).")
    parser.add_argument("--sectors-path",         type=Path, metavar="FILE", default=C("sectors-path", None),
                        help="Location of the sector (capacity) CSV file.")
    parser.add_argument("--flights-path",         type=Path, metavar="FILE", default=C("flights-path", None),
                        help="Location of the flights CSV file.")
    parser.add_argument("--airports-path",        type=Path, metavar="FILE", default=C("airports-path", None),
                        help="Location of the airport-vertices CSV file.")
    parser.add_argument("--airplanes-path",       type=Path, metavar="FILE", default=C("airplanes-path", None),
                        help="Location of the airplanes CSV file.")
    parser.add_argument("--airplane-flight-path", type=Path, metavar="FILE", default=C("airplane-flight-path", None),
                        help="Location of the airplane-flight-assignment CSV file.")
    parser.add_argument("--navaid-sector-path",   type=Path, metavar="FILE", default=C("navaid-sector-path", None),
                        help="Location of the navaids-sector assignment CSV file.")

    # Results saving
    parser.add_argument(
        "--save-results",
        type=str,
        default=str(C("save-results", "true")),
        help="true/false: save optimizer result matrices to disk (default: true).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(C("results-root", "experiment_output")),
        help="Root folder to store result matrices (default: experiment_output).",
    )
    parser.add_argument(
        "--results-format",
        type=str,
        choices=["csv","csv.gz","npz"],
        default=C("results-format", "csv"),
        help="File format for matrices: 'csv' (default, uncompressed), 'csv.gz', or 'npz'.",
    )

    # Encoding + knobs
    parser.add_argument("--encoding-path", type=Path, default=Path(C("encoding-path", DEFAULT_FILENAMES["encoding_path"])),
                        metavar="FILE", help="Location of the encoding for the optimization problem.")
    parser.add_argument("--seed", type=int, default=int(C("seed", 11904657)),
                        help="Set the random seed.")
    parser.add_argument("--number-threads", type=int, default=int(C("number-threads", 20)),
                        help="Number of parallel ASP solving threads.")
    parser.add_argument("--timestep-granularity", type=int, default=int(C("timestep-granularity", 1)),
                        help="Granularity: 1=1h, 4=15min, etc.")
    parser.add_argument("--max-explored-vertices", type=int, default=int(C("max-explored-vertices", 6)),
                        help="Max vertices explored in parallel.")
    parser.add_argument("--max-considered-aircraft", type=int, default=int(C("max-considered-aircraft", 2)),
                        help="Max considered aircraft in one optimization step.")
    parser.add_argument("--max-delay-per-iteration", type=int, default=int(C("max-delay-per-iteration", -1)),
                        help="Max hours of delay per iteration (−1 = auto).")
    parser.add_argument("--max-time", type=int, default=int(C("max-time", 24)),
                        help="Number of timesteps for one day.")
    parser.add_argument("--verbosity", type=int, default=int(C("verbosity", 0)),
                        help="Verbosity levels (0,1,2).")
    parser.add_argument("--sector-capacity-factor", type=int, default=int(C("sector-capacity-factor", 6)),
                        help="Defines capacity of composite sectors.")
    parser.add_argument("--convex-sectors", type=int, default=int(C("convex-sectors", 0)),
                        help="--convex-sectors=0 (false), --convex-sectors=1 (true)")
    parser.add_argument("--max-number-navpoints-per-sector", type=int, default=int(C("max-number-navpoints-per-sector", -1)),
                        help="Defines the maximum number of navpoints per composite sectors (-1=initial max number).")
    parser.add_argument("--max-number-sectors", type=int, default=int(C("max-number-sectors", -1)),
                        help="Defines the maximum number of sectors that can exist at one timepoint (-1=initial number).")
    parser.add_argument("--minimize-number-sectors-enabled", type=str, default=str(C("minimize-number-sectors-enabled", "false")),
                        help="true/false: If enabled, minimized every timestep-granularity the sectors.")

    parser.add_argument("--composite-sector-function", type=str, default=str(C("composite-sector-function", "max")),
                        help="Defines the function of the composite sector - available: max, triangular, linear")

    parser.add_argument("--controller-enabled", type=str, default=str(C("controller-enabled", "false")))
    parser.add_argument("--explainability-enabled", type=str, default=str(C("explainability-enabled", "false")))
    parser.add_argument("--controller-control-socket-port", type=int, default=5555)
    parser.add_argument("--controller-data-socket-port", type=int, default=5556)

    # DYNAMIC SECTORIZATION:
    parser.add_argument("--number-capacity-management-configs", type=int, default=int(C("number-capacity-management-configs", 7)), help="How many compisitions/partitions to consider (only works when cap-mgmt. is enabled.")
    parser.add_argument("--capacity-management-enabled",
        type=str,
        default=str(C("capacity-management-enabled", "true")),
        help="true/false: true when cap-mgmt. is enabled.",
    )

    # WANDB:
    parser.add_argument("--wandb-enabled", type=str, default=str(C("wandb-enabled", "false")),
                        help="true/false: If enabled, trace run on wandb.")
    parser.add_argument("--wandb-experiment-name-prefix", type=str, default=str(C("wandb-experiment-name-prefix","")),
                        help="Defines the wandb prefix name for tracing experiments (only used when wandb is enabled).")
    parser.add_argument("--wandb-experiment-name-suffix", type=str, default=str(C("wandb-experiment-name-suffix","")),
                        help="Defines the wandb suffix name for tracing experiments (only used when wandb is enabled).")
    parser.add_argument("--wandb-api-key-path", type=Path, default=Path(C("wandb-api-key-path", DEFAULT_FILENAMES["wandb_api_key"])),
                        metavar="FILE", help="Location of the wandb API key file (only searched when wandb is enabled).")
    parser.add_argument("--wandb-project", type=str, default=str(C("wandb-project", "ASPaeroFlow")),
                        help="Weights & Biases project name (default: ASPaeroFlow).")
    parser.add_argument("--wandb-entity", type=str, default=C("wandb-entity", None),
                        help="Weights & Biases entity (username or team/organization). Leave empty to use your default entity.")
    
    parser.add_argument("--optimizer", type=str, default=C("optimizer","ASP"), help="Either ASP or Enumerate")

    return parser

def _apply_data_dir_defaults(args: argparse.Namespace, folder=None) -> argparse.Namespace:
    """For any missing *-path, use data_dir / default_filename."""

    if folder is None:
        if not args.data_dir:
            return args
        base = args.data_dir
    else:
        base = folder

    def fill(cur: Optional[Path], fname_key: str) -> Path:
        return cur if cur else (base / DEFAULT_FILENAMES[fname_key])

    args.graph_path           = fill(args.graph_path,           "graph_path")
    args.sectors_path         = fill(args.sectors_path,         "sectors_path")
    args.flights_path         = fill(args.flights_path,         "flights_path")
    args.airports_path        = fill(args.airports_path,        "airports_path")
    args.airplanes_path       = fill(args.airplanes_path,       "airplanes_path")
    args.airplane_flight_path = fill(args.airplane_flight_path, "airplane_flight_path")
    args.navaid_sector_path   = fill(args.navaid_sector_path,   "navaid_sector_path")
    # encoding_path already has a default; leave as-is

    return args

def _validate_inputs(args: argparse.Namespace):
    missing = []
    for k in [
        "graph_path","sectors_path","flights_path","airports_path",
        "airplanes_path","airplane_flight_path","navaid_sector_path","encoding_path"
    ]:
        p: Path = getattr(args, k)

        if p is None or not Path(p).exists():
            missing.append((k, str(p)))
    if missing:
        lines = ["Input files not found:"]
        lines += [f"  - {k}: {v}" for k, v in missing]
        raise FileNotFoundError("\n".join(lines))

def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI with priority: CLI > config > built-in defaults."""

    # 1) preparse to get --config
    pre, _ = _preparse(argv)
    cfg = {}
    if pre.config and pre.config.exists():
        with open(pre.config, "r") as fh:
            cfg = json.load(fh) or {}

    # allow config to set a default data-dir as well
    if pre.data_dir is None:
        cfg_data_dir = _cfg_get(cfg, "data-dir", None)
        if cfg_data_dir:
            pre.data_dir = Path(cfg_data_dir)

    # 2) build the full parser with cfg-derived defaults
    parser = _build_arg_parser(cfg)
    args = parser.parse_args(argv)
    # normalize booleans
    def _str2bool(v):
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        return s in ("1","true","t","yes","y","on")

    args.save_results = _str2bool(args.save_results)
    args.wandb_enabled = _str2bool(args.wandb_enabled)
    args.minimize_number_sectors = _str2bool(args.minimize_number_sectors_enabled)

    args.controller_enabled = _str2bool(args.controller_enabled)
    args.explainability_enabled = _str2bool(args.explainability_enabled)


    # Keep the config path in args for traceability
    if args.config is None and pre.config:
        args.config = pre.config

    # 3) If data-dir provided (CLI or config), auto-fill missing file paths
    if args.data_dir is None and pre.data_dir:
        args.data_dir = pre.data_dir
    args = _apply_data_dir_defaults(args)

    if args.controller_enabled is False and args.explainability_enabled is False:
        # 4) Final validation
        _validate_inputs(args)


    return args

def _derive_output_name(args: argparse.Namespace) -> str:
    """
    Use the last segment of --data-dir as the experiment name, e.g. 0000763_SEED42.
    Assumes --data-dir is provided (as per user workflow).
    """
    if args.data_dir:
        return Path(args.data_dir).name
    # Fallback: try to use flights' parent directory name
    if args.flights_path:
        return Path(args.flights_path).parent.name or "RESULTS"
    return "RESULTS"

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_npz(path: Path, arr, key: str):
    a = np.asarray(arr)
    np.savez_compressed(path, **{key: a})

def _save_csv_gz(path: Path, arr):
    a = np.asarray(arr)
    # Write as integer CSV (or fall back to float if needed)
    # We avoid pandas for speed/dep-minimization; numpy.savetxt with gzip works well.
    import gzip
    fmt = "%d" if a.dtype.kind in ("i","u","b") else "%g"
    with gzip.open(path, "wt", encoding="utf-8") as gz:
        np.savetxt(gz, a, fmt=fmt, delimiter=",")

def _save_csv(path: Path, arr):
    a = np.asarray(arr)
    fmt = "%d" if a.dtype.kind in ("i","u","b") else "%g"
    np.savetxt(path, a, fmt=fmt, delimiter=",")

def _save_results(args: argparse.Namespace, app) -> None:
    """
    Persist the three result matrices if present on `app`:
      - navaid_sector_time_assignment  (|N| x |T|)
      - converted_instance_matrix      (|F| x |T|)
      - converted_navpoint_matrix      (|F| x |T|)
    """
    out_name = _derive_output_name(args)
    out_dir  = _ensure_dir(Path(args.results_root) / out_name)

    mats = {
        "navaid_sector_time_assignment": getattr(app, "navaid_sector_time_assignment", None),
        "converted_instance_matrix":     getattr(app, "converted_instance_matrix", None),
        "converted_navpoint_matrix":     getattr(app, "converted_navpoint_matrix", None),
        "capacity_time_matrix":     getattr(app, "capacity_time_matrix", None),
    }

    saved = {}
    for key, val in mats.items():
        if val is None:
            continue
        if args.results_format == "npz":
            _save_npz(out_dir / f"{key}.npz", val, key)
        elif args.results_format == "csv.gz":
            _save_csv_gz(out_dir / f"{key}.csv.gz", val)
        else:  # "csv"
            _save_csv(out_dir / f"{key}.csv", val)

        a = np.asarray(val)
        saved[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}

    # Write a small manifest
    manifest = {
        "saved": saved,
        "format": args.results_format,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "seed": args.seed,
            "timestep_granularity": args.timestep_granularity,
        }
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if args.verbosity > 0:
        print(f"[✓] Saved results → {out_dir}")



# ---------------------------------------------------------------------------
# Top-level script wrapper
# ---------------------------------------------------------------------------


def initialize_explainability(args):

    control_context = zmq.Context()
    
    control_socket_port = args.controller_control_socket_port 
    data_socket_port = args.controller_data_socket_port
    
    # 1. Control Channel (PAIR)
    control_ctrl_socket = control_context.socket(zmq.PAIR)
    control_ctrl_socket.bind(f"tcp://127.0.0.1:6000")
    
    # 2. Telemetry Channel (PUB)
    control_pub_socket = control_context.socket(zmq.PUB)
    control_pub_socket.connect(f"tcp://127.0.0.1:{data_socket_port}")



    # Configure the Poller for I/O multiplexing
    init_poller = zmq.Poller()
    init_poller.register(control_ctrl_socket, zmq.POLLIN)
    

    controller_defined_instance = False

    while True:
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
            message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)

            explain_dict = json.loads(message[9:])
            break

    control_ctrl_socket.send_string("ack")

    control_ctrl_socket.setsockopt(zmq.LINGER, 0)
    control_ctrl_socket.close()

    return control_pub_socket, explain_dict


def initialize_controller(args):

    control_context = zmq.Context()
    
    control_socket_port = args.controller_control_socket_port 
    data_socket_port = args.controller_data_socket_port
    
    # 1. Control Channel (PAIR)
    control_ctrl_socket = control_context.socket(zmq.PAIR)
    control_ctrl_socket.connect(f"tcp://127.0.0.1:{control_socket_port}")
    
    # 2. Telemetry Channel (PUB)
    control_pub_socket = control_context.socket(zmq.PUB)
    control_pub_socket.connect(f"tcp://127.0.0.1:{data_socket_port}")
    
    # Non-blocking I/O setup for the Control Channel
    control_poller = zmq.Poller()
    control_poller.register(control_ctrl_socket, zmq.POLLIN)

    control_ctrl_socket.send_string("INITIALIZED OPTIMIZER")


    # Configure the Poller for I/O multiplexing
    init_poller = zmq.Poller()
    init_poller.register(control_ctrl_socket, zmq.POLLIN)

    controller_defined_instance = False

    while True:
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
            message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "CONTROLLER DEFINED INSTANCE":
                controller_defined_instance = True
                break
            elif message == "OPTIMIZER DEFINED INSTANCE":
                controller_defined_instance = False
                break
            else:
                print(f"OPTIMIZER BUSY:\n{message}")

    init_poller = zmq.Poller()
    init_poller.register(control_ctrl_socket, zmq.POLLIN)
    control_ctrl_socket.send_string("ack")

    while True:
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
            message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "GET OPTIONS":
                break

    config_dict = {}

    config_dict["timestep_granularity"] = {}
    config_dict["timestep_granularity"]["type"] = "int"
    config_dict["timestep_granularity"]["value"] = str(args.timestep_granularity)
    config_dict["timestep_granularity"]["name"] = "Timestep Granularity"

    config_dict["max_explored_vertices"] = {}
    config_dict["max_explored_vertices"]["type"] = "int"
    config_dict["max_explored_vertices"]["value"] = str(args.max_explored_vertices)
    config_dict["max_explored_vertices"]["name"] = "Max Explored Vertices"

    config_dict["max_delay_per_iteration"] = {}
    config_dict["max_delay_per_iteration"]["type"] = "int"
    config_dict["max_delay_per_iteration"]["value"] = str(args.max_delay_per_iteration)
    config_dict["max_delay_per_iteration"]["name"] = "Max Delay Per Iteration"

    config_dict["number_capacity_management_configs"] = {}
    config_dict["number_capacity_management_configs"]["type"] = "int"
    config_dict["number_capacity_management_configs"]["value"] = str(args.number_capacity_management_configs)
    config_dict["number_capacity_management_configs"]["name"] = "Number Capacity Management Configs"

    config = json.dumps(config_dict)

    init_poller = zmq.Poller()
    init_poller.register(control_ctrl_socket, zmq.POLLIN)
    control_ctrl_socket.send_string(config)

    while True:
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
            message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "ack":
                break

    while True:
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
            message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "GET OBJECTIVE FUNCTIONS":
                break


    optimization_criteria_config = {}
    optimization_criteria_config[0] = {}
    optimization_criteria_config[0]["name"] = "Overload"
    optimization_criteria_config[0]["id"] = "OVERLOAD"
    optimization_criteria_config[1] = {}
    optimization_criteria_config[1]["name"] = "Arrival Delay"
    optimization_criteria_config[1]["id"] = "ARRIVAL-DELAY"
    optimization_criteria_config[2] = {}
    optimization_criteria_config[2]["name"] = "Number of Sectors"
    optimization_criteria_config[2]["id"] = "SECTOR-NUMBER"
    optimization_criteria_config[3] = {}
    optimization_criteria_config[3]["name"] = "Sector Changes"
    optimization_criteria_config[3]["id"] = "SECTOR-DIFF"
    optimization_criteria_config[4] = {}
    optimization_criteria_config[4]["name"] = "Number of Reroutes"
    optimization_criteria_config[4]["id"] = "REROUTE"
    optimization_criteria_config[5] = {}
    optimization_criteria_config[5]["name"] = "Number of Reconfigs"
    optimization_criteria_config[5]["id"] = "RECONFIG"

    optimization_criteria = json.dumps(optimization_criteria_config)

    init_poller = zmq.Poller()
    init_poller.register(control_ctrl_socket, zmq.POLLIN)
    control_ctrl_socket.send_string(optimization_criteria)

    while True:
        # Poll with a 1000ms timeout to prevent deadlocks
        socks = dict(init_poller.poll(1000))

        # Process Optimizer control socket events
        if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
            message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
            if message == "ack":
                break
    
    if controller_defined_instance is True:
        # Configure the Poller for I/O multiplexing

        while True:
            # Poll with a 1000ms timeout to prevent deadlocks
            socks = dict(init_poller.poll(1000))

            # Process Optimizer control socket events
            if control_ctrl_socket in socks and socks[control_ctrl_socket] == zmq.POLLIN:
                message = control_ctrl_socket.recv_string(flags=zmq.NOBLOCK)
                if message.startswith("<LOAD>"):
                    message = message[6:]
                    data_dir_path = Path(message)
                    args.data_dir = data_dir_path

                    args.graph_path           = None
                    args.sectors_path         = None
                    args.flights_path         = None
                    args.airports_path        = None
                    args.airplanes_path       = None
                    args.airplane_flight_path = None
                    args.navaid_sector_path   = None

                    args = _apply_data_dir_defaults(args, data_dir_path)
                    _validate_inputs(args)
                    break
                elif message.startswith("<OPTION>"):
                    message = message[8:]
                    message = json.loads(message)
                    for key in message.keys():
                        if hasattr(args,key):
                            setattr(args,key, int(message[key]))
                        else:
                            print(f"NOT FOUND ATTR:{key}:{message[key]}")

    return control_context, control_ctrl_socket, control_pub_socket, control_poller


def main(argv: Optional[List[str]] = None) -> None:
    """Script entry-point compatible with both `python -m` and `poetry run`."""
    args = parse_cli(argv)
    
    if args.controller_enabled is True:
        control_context, control_ctrl_socket, control_pub_socket, control_poller = initialize_controller(args)
        explainability_context = None
    elif args.explainability_enabled is True:
        control_pub_socket, explainability_context = initialize_explainability(args)
        control_context = None
        control_ctrl_socket = None
        control_poller = None

    else:
        control_context = None
        control_ctrl_socket = None
        control_pub_socket = None
        control_poller = None
        explainability_context = None

    if args.verbosity > 0:
        # Optional: small echo of resolved inputs
        print("[i] Using inputs:")
        print(f"    graph:           {args.graph_path}")
        print(f"    sectors:         {args.sectors_path}")
        print(f"    flights:         {args.flights_path}")
        print(f"    airports:        {args.airports_path}")
        print(f"    airplanes:       {args.airplanes_path}")
        print(f"    airplane-flight: {args.airplane_flight_path}")
        print(f"    navaid-sector:   {args.navaid_sector_path}")
        print(f"    encoding:        {args.encoding_path}")
        if args.data_dir:
            print(f"    data-dir:        {args.data_dir}")
        if args.config:
            print(f"    config:          {args.config}")
        if args.wandb_enabled:
            print(f"    wandb:           entity={args.wandb_entity or '(default)'} project={args.wandb_project} "
                  f"name={args.wandb_experiment_name_prefix}{_derive_output_name(args)}{args.wandb_experiment_name_suffix}")


    composite_sector_function = args.composite_sector_function.lower()
    if composite_sector_function not in [MAX,LINEAR,TRIANGULAR]:
        raise Exception(f"Specified composite sector function {composite_sector_function} not in {[MAX,LINEAR,TRIANGULAR]}")

    experiment_name = _derive_output_name(args)
    
    # W&B setup (optional)
    run = None
    wandb_log = None
    if args.wandb_enabled:
        try:
            import wandb  # installed by user
        except ImportError as e:
            raise ImportError("wandb is not installed, but --wandb-enabled was set to true.") from e
        api_key_path: Path = args.wandb_api_key_path
        if not api_key_path.exists():
            raise FileNotFoundError(f"W&B API key file not found: {api_key_path}")
        key = api_key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"W&B API key file is empty: {api_key_path}")
        os.environ["WANDB_API_KEY"] = key
        wandb.login(key=key, relogin=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity if args.wandb_entity else None),
            name=f"{args.wandb_experiment_name_prefix}{experiment_name}{args.wandb_experiment_name_suffix}",
            config={
                "timestep_granularity": args.timestep_granularity,
                "max_explored_vertices": args.max_explored_vertices,
                "number_threads": args.number_threads,
                "max_delay_per_iteration": args.max_delay_per_iteration,
                "number_capacity_management_configs": args.number_capacity_management_configs,
                "capacity_management_enabled": args.capacity_management_enabled,
                "composite_sector_function": composite_sector_function,
                "seed": args.seed,
                "max_time": args.max_time,
            },
        )
        wandb_log = run.log


    while True:

        app = Main(args.graph_path, args.sectors_path, args.flights_path,
                args.airports_path, args.airplanes_path,
                args.airplane_flight_path, args.navaid_sector_path,
                args.encoding_path,
                args.seed, args.number_threads, args.timestep_granularity,
                args.max_explored_vertices, args.max_delay_per_iteration,
                args.max_time, args.verbosity,
                args.sector_capacity_factor,
                args.number_capacity_management_configs,
                args.capacity_management_enabled,
                composite_sector_function,
                experiment_name,
                wandb_log,
                args.optimizer, args.max_number_navpoints_per_sector, args.max_number_sectors, args.minimize_number_sectors,
                args.convex_sectors,
                control_context, control_ctrl_socket, control_pub_socket, control_poller, 
                args.controller_enabled, args.data_dir,
                args.max_considered_aircraft,
                explainability_context
                )
        key, value = app.run()

        if args.controller_enabled is True:
            if key != "<LOAD>":
                # poller.poll() with None blocks indefinitely until I/O occurs
                while key not in ["<LOAD>","<QUIT>"]:
                    events = dict(app._control_poller.poll(timeout=None))
                    if app._control_ctrl_socket in events:
                        command = app._control_ctrl_socket.recv_string()
                        if command == "START":
                            print("[CONTROL->OPTIMIZER]: START NOT EXECUTED (idle).")
                            app._control_ctrl_socket.send_string("TELEMETRY: [STATUS] RESUMED")
                            continue
                        elif command.startswith("<LOAD>"):
                            print("[CONTROL->OPTIMIZER]: LOAD")
                            key = "<LOAD>"
                            value = command[6:]
                        elif command.startswith("<OPTION>"):
                            print("[CONTROL->OPTIMIZER]: Option not executed (idle).")
                        elif command.startswith("<QUIT>"):
                            print("[CONTROL->OPTIMIZER]: QUIT")
                            key = "<QUIT>"

            if key == "<LOAD>":
                print("RECONFIGURE TO:")
                print(value)
                data_dir_path = Path(value)
                args.data_dir = data_dir_path
                args.graph_path           = None
                args.sectors_path         = None
                args.flights_path         = None
                args.airports_path        = None
                args.airplanes_path       = None
                args.airplane_flight_path = None
                args.navaid_sector_path   = None
                args = _apply_data_dir_defaults(args, data_dir_path)
                _validate_inputs(args)
            elif key == "<QUIT>":
                break
            else:
                break
        else:
            break


    # Save results if requested
    if args.save_results:
        _save_results(args, app)

    if run is not None:
        run.finish()

    #print(app.get_total_atfm_delay())


if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()

