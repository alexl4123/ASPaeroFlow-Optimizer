# MAIN.py
# Author: Alexander Beiser

import argparse
import sys
import numpy as np
import os



from solver import Solver, Model

from pathlib import Path
from typing import Any, List, Optional, Final

from datetime import datetime, timezone

from translate import TranslateCSVtoLogicProgram


AFFIRMATIVE: Final[set[str]] = {"yes", "y"}
NEGATIVE: Final[set[str]] = {"no", "n", "exit"}

# ---------------------------------------------------------------------------
# CLI utilities (with config + bundle directory support)
# ---------------------------------------------------------------------------

import argparse, json
from pathlib import Path
from typing import Optional, List, Dict

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
    parser.add_argument("--max-delay-per-iteration", type=int, default=int(C("max-delay-per-iteration", -1)),
                        help="Max hours of delay per iteration (−1 = auto).")
    parser.add_argument("--max-time", type=int, default=int(C("max-time", 24)),
                        help="Number of timesteps for one day.")
    parser.add_argument("--verbosity", type=int, default=int(C("verbosity", 0)),
                        help="Verbosity levels (0,1,2).")
    parser.add_argument("--sector-capacity-factor", type=int, default=int(C("sector-capacity-factor", 6)),
                        help="Defines capacity of composite sectors.")

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
    
    # REGULATIONS ACTIVE:
    parser.add_argument("--regulation-ground-delay-active", type=str, default=str(C("regulation-ground-delay-active", "true")),
                        help="true/false: Enable ground delays of aircraft.")
    parser.add_argument("--regulation-rerouting-active", type=str, default=str(C("regulation-rerouting-active", "true")),
                        help="true/false: Enable rerouting of aircraft.")
    parser.add_argument("--regulation-dynamic-sectorization-active", type=str, default=str(C("regulation-dynamic-sectorization-active", "true")),
                        help="true/false: Enable dynamic sectorization.")

    return parser

def _apply_data_dir_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """For any missing *-path, use data_dir / default_filename."""
    if not args.data_dir:
        return args
    base = args.data_dir

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

    # Keep the config path in args for traceability
    if args.config is None and pre.config:
        args.config = pre.config

    # 3) If data-dir provided (CLI or config), auto-fill missing file paths
    if args.data_dir is None and pre.data_dir:
        args.data_dir = pre.data_dir
    args = _apply_data_dir_defaults(args)

    # 4) Final validation
    _validate_inputs(args)

    # normalize booleans
    def _str2bool(v):
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        return s in ("1","true","t","yes","y","on")
    args.save_results = _str2bool(args.save_results)
    args.wandb_enabled = _str2bool(args.wandb_enabled)
    
    args.regulation_ground_delay_active = _str2bool(args.regulation_ground_delay_active)
    args.regulation_rerouting_active = _str2bool(args.regulation_rerouting_active)
    args.regulation_dynamic_sectorization_active = _str2bool(args.regulation_dynamic_sectorization_active)

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


class ModelData:

    def __init__(self, navaid_sector_time_assignment, converted_instance_matrix, converted_navpoint_matrix):

        self.navaid_sector_time_assignment = navaid_sector_time_assignment
        self.converted_instance_matrix = converted_instance_matrix
        self.converted_navpoint_matrix = converted_navpoint_matrix

# ---------------------------------------------------------------------------
# Top‑level script wrapper
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Script entry‑point compatible with both `python -m` and `poetry run`."""
    args = parse_cli(argv)

    graph_csv = args.graph_path
    sectors_csv = args.sectors_path
    flights_csv = args.flights_path
    airports_csv = args.airports_path
    airplanes_csv = args.airplanes_path
    airplane_flight_csv = args.airplane_flight_path
    navaid_sector_csv = args.navaid_sector_path

    encoding_path = args.encoding_path
    verbosity = args.verbosity
    sector_capacity_factor = args.sector_capacity_factor

    regulation_ground_delay_active = args.regulation_ground_delay_active
    regulation_rerouting_active = args.regulation_rerouting_active
    regulation_dynamic_sectorization_active = args.regulation_dynamic_sectorization_active

    seed = args.seed

    timestep_granularity = args.timestep_granularity

    max_time = args.max_time

    experiment_name = _derive_output_name(args)
    # WANDB (W&B) setup (optional)
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
                "seed": args.seed,
                "max_time": args.max_time,
            },
        )
        wandb_log = run.log
    else:
        wandb_log = None


    model = None


    while model is None:

        asp_instance = TranslateCSVtoLogicProgram().main(graph_csv, flights_csv, sectors_csv,
            airports_csv, airplanes_csv, airplane_flight_csv, navaid_sector_csv, encoding_path, timestep_granularity, max_time,
            sector_capacity_factor,
            regulation_ground_delay_active, regulation_rerouting_active, regulation_dynamic_sectorization_active)
        

        
        instance_asp_atoms = "\n".join(asp_instance)

        open("20251223_instance.lp","w").write(instance_asp_atoms)
        quit()

        encoding = open(encoding_path, "r").read()

        solver: Model = Solver(encoding, instance_asp_atoms, seed=seed)
        model = solver.solve()
            
        if verbosity > 0:
            print(f"""
        Result of Answer:
        - Overload: {model.get_total_overload()}
        - ATFM Delay: {model.get_total_atfm_delay()}
        - Computation time: {model.computation_time}s
        - Rerouted Airplanes: {model.get_rerouted_airplanes()}
            """)

        if model.get_total_overload() > 0:
            max_time += 1
            model = None

    if verbosity > 0:
        print(f"""
    Result of Answer:
    - ATFM Delay: {model.get_total_atfm_delay()}
    - Computation time: {model.computation_time}s
    - Rerouted Airplanes: {model.get_rerouted_airplanes()}
        """)
    flights = model.get_flights()

    distinct_flights = set()
    for flight in flights:
        distinct_flights.add(flight.arguments[0])


    distinct_navpoints = set()
    for navpoint_sector in model.get_navaid_sector_time_assignment():
        distinct_navpoints.add(navpoint_sector.arguments[0])
        if max_time < int(str(navpoint_sector.arguments[2])):
            max_time = int(str(navpoint_sector.arguments[2]))

    converted_instance_matrix = np.ones((len(distinct_flights),max_time+1)) * -1

    for flight in flights:
        flight_id = int(str(flight.arguments[0]))
        flight_time = int(str(flight.arguments[1]))
        flight_sector = int(str(flight.arguments[2]))

        converted_instance_matrix[flight_id, flight_time] = flight_sector

    converted_navpoint_matrix = np.ones((len(distinct_flights),max_time+1)) * -1

    for navpoint_flight in model.get_navpoint_flights():
        flight_id = int(str(navpoint_flight.arguments[0]))
        flight_time = int(str(navpoint_flight.arguments[1]))
        flight_navaid = int(str(navpoint_flight.arguments[2]))

        converted_navpoint_matrix[flight_id, flight_time] = flight_navaid


    navaid_sector_time_assignment = np.ones((len(distinct_navpoints),max_time+1)) * -1
    for navaid_sector in model.get_navaid_sector_time_assignment():
        navaid = int(str(navaid_sector.arguments[0]))
        sector = int(str(navaid_sector.arguments[1]))
        time = int(str(navaid_sector.arguments[2]))

        navaid_sector_time_assignment[navaid, time] = sector


    model_data = ModelData(navaid_sector_time_assignment,converted_instance_matrix,converted_navpoint_matrix)

    if verbosity > 0:
        print(model.get_rerouted_airplanes())

    print(model.get_total_atfm_delay())
    #np.savetxt(sys.stdout, converted_instance_matrix, delimiter=",", fmt="%i") 

    # Save results if requested
    if args.save_results:
        _save_results(args, model_data)

    if run is not None:
        run.finish()

    return model.get_total_atfm_delay()

if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


