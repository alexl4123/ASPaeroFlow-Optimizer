# MAIN.py
# Author: Alexander Beiser

# First, before the imports below (numpy, clingo) take their time: --solve-deadline counts from
# process start, and this is the fallback for when /proc cannot say when that was. Imported under
# an alias because main() has a local variable called `time`.
import time as _time
_MAIN_LOADED_AT = _time.monotonic()

import argparse
import sys
import numpy as np
import os



from solver import Solver, Model

from pathlib import Path
from typing import Any, List, Optional, Final

from datetime import datetime, timezone

from translate import TranslateCSVtoLogicProgram
from arrival_delay_bootstrap import (
    add_cli_argument as add_arrival_delay_metric_argument,
    asp_metric_fact,
)
from clingo_options_bootstrap import (
    add_cli_arguments as add_solver_option_arguments,
    describe as describe_solver_options,
    options_from_args as solver_options_from_args,
)


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

    # Export the grounded instance so that other groups can run their own solvers against
    # exactly the program this pipeline builds, without reimplementing the CSV -> ASP
    # translation. --export-instance-lp gives the facts alone; --export-program-lp gives a
    # self-contained file (encoding + arrival-delay metric fact + facts) that clingo can run
    # directly. Both write and then continue solving as normal; neither changes the search.
    parser.add_argument("--export-instance-lp", type=Path, default=None, metavar="FILE",
                        help="Write the translated instance facts to FILE (ASP, one atom per "
                             "line) and continue.")
    parser.add_argument("--export-program-lp", type=Path, default=None, metavar="FILE",
                        help="Write a self-contained program (encoding + metric fact + instance "
                             "facts) to FILE and continue. Runnable with: clingo FILE")
    parser.add_argument("--export-only", action="store_true",
                        help="With one of the --export-*-lp options, write the file(s) and exit "
                             "without solving.")
    parser.add_argument("--seed", type=int, default=int(C("seed", 11904657)),
                        help="Set the random seed.")
    # This used to be parsed and then dropped: it reached nothing but the W&B config dict, so
    # every ASP run ever made was single-threaded whatever it said. It now reaches clasp as
    # --parallel-mode. The default moved 20 -> 1 in the same change, deliberately: wiring the
    # option up while leaving the default at 20 would have turned every run 20-threaded and made
    # the numbers incomparable with the published LPNMR/ATMOS results, which is the opposite of
    # the intent. 1 is also clingo's own default, so this reproduces every run to date exactly.
    parser.add_argument("--number-threads", type=int, default=int(C("number-threads", 1)),
                        help="Number of parallel clasp SEARCH threads (clingo --parallel-mode). "
                             "Default 1, matching every published result. clasp runs a portfolio "
                             "in parallel mode, so a higher count changes the search itself, not "
                             "only its speed.")
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
    parser.add_argument("--regulation-ground-delay-active", type=int, default=str(C("regulation-ground-delay-active", 2)),
                        help="0=no ground delay, 1=restricted ground delay, 2=full dynamic delaying")
    parser.add_argument("--regulation-rerouting-active", type=int, default=str(C("regulation-rerouting-active", 2)),
                        help="0=no rerouting, 1=restricted rerouting, 2=full dynamic rerouting")
    parser.add_argument("--regulation-dynamic-sectorization", type=int, default=str(C("regulation-dynamic-sectorization", 2)),
                        help="0=no dynamic sectorization, 1=restricted dynamic sectorization, 2=full dynamic sectorization.")
    add_arrival_delay_metric_argument(parser, default=C("arrival-delay-metric", None))

    parser.add_argument("--allow-overloads", type=str, default=str(C("allow-overloads", "false")),
                        help="true/false: Allow solutions with overload constraint violations.")

    # Clingo search configuration, selected by name. The default profile is the empty flag list,
    # i.e. exactly the clingo.Control() this script has always built, so an invocation that does
    # not mention these options is unchanged. See common/clingo_options.py.
    add_solver_option_arguments(parser,
                                default_profile=C("solver-profile", None),
                                default_args=C("solver-arg", None))
    parser.add_argument("--solver-stats", type=str, default=str(C("solver-stats", "false")),
                        help="true/false: add clingo diagnostics (cost vector, LOWER BOUND, "
                             "models reported, whether the search was exhausted) to each JSON "
                             "line. Off by default so the reported line stays exactly what "
                             "downstream parsers expect. Worth turning on with "
                             "--solver-profile usc, where the incumbent alone does not say how "
                             "much of the gap has been closed.")
    # Deliberately CLI-only (no config default): it changes how a run ends, so it has to be asked
    # for on the command line that produced the result.
    parser.add_argument("--solve-deadline", type=float, default=None, metavar="SECONDS",
                        help="Stop the clingo search from inside this process once SECONDS have "
                             "passed since the PROCESS STARTED (interpreter start-up, translation "
                             "and grounding count), then print the best model with the solver's "
                             "statistics -- including the lower bound under --solver-stats -- and "
                             "a SOLVER-STOPPED-AT-DEADLINE field. Set it below an external time "
                             "limit (e.g. 1770 for 1800) so the run stops before it is killed. "
                             "Translation and grounding cannot be interrupted. Off by default: "
                             "without it the solve is the blocking call it has always been.")

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
    args.solver_stats = _str2bool(args.solver_stats)
    
    #args.regulation_ground_delay_active = _str2bool(args.regulation_ground_delay_active)
    #args.regulation_rerouting_active = _str2bool(args.regulation_rerouting_active)
    args.allow_overloads = _str2bool(args.allow_overloads)

    if args.solve_deadline is not None and not 0 < args.solve_deadline < float("inf"):
        parser.error(f"--solve-deadline must be a positive number of seconds, "
                     f"got {args.solve_deadline}")

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


# ---------------------------------------------------------------------------
# --solve-deadline
# ---------------------------------------------------------------------------

def _process_start_monotonic() -> float:
    """time.monotonic() at the moment this process was started.

    The benchmark caller's clock starts when it spawns the process, so the deadline has to count
    interpreter start-up and imports as well, which happen before any line of this file runs.
    Linux records the start in /proc/self/stat (field 22, clock ticks since boot); anywhere that
    cannot be read, the time this module began loading is used instead, which is later by the
    interpreter's start-up only.
    """
    try:
        with open("/proc/self/stat", "r") as fh:
            stat = fh.read()
        # Field 2 (the command name) is in parentheses and may contain spaces, so count the fields
        # after its closing parenthesis: field 3 is index 0 there, field 22 is index 19.
        start_ticks = int(stat[stat.rindex(")") + 2:].split()[19])
        age = _time.clock_gettime(_time.CLOCK_BOOTTIME) - start_ticks / os.sysconf("SC_CLK_TCK")
        now = _time.monotonic()
        # Sanity: the process cannot have started after this module loaded, and an interpreter
        # start-up of minutes means the two clocks disagree (e.g. a time namespace), not a slow
        # start -- fall back rather than put the deadline in the past.
        if 0.0 <= age and 0.0 <= _MAIN_LOADED_AT - (now - age) <= 300.0:
            return now - age
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return _MAIN_LOADED_AT


#: How much longer the next horizon's uninterruptible setup is allowed to be than the last one's
#: when the re-solve loop decides whether it still fits before the deadline. Measured: ctl.ground()
#: grew by 8-26% per horizon step (24 -> 34) on a 30-flight central-Europe instance, variant
#: rp_d_sp, so 1.5 leaves room for that and for timing noise.
RESOLVE_SETUP_GROWTH = 1.5


def _since(moment, process_start):
    return round(moment - process_start, 3) if moment is not None else None


def _deadline_fields(model, solver, args, process_start: float, last_max_time: int,
                     deadline_cut: bool) -> Dict[str, object]:
    """The keys --solve-deadline adds to the final result line.

    SOLVER-STOPPED-AT-DEADLINE  the deadline ended the run: the last search was cancelled, or a
                                longer horizon the re-solve loop wanted was not searched.
                                COMPUTATION-FINISHED stays what it always was, the last search's
                                "exhausted" flag.
    SOLVER-DEADLINE-S           the deadline, in seconds after process start.
    SOLVER-GROUNDING-ENDED-S    seconds after process start at which the last search was
    SOLVER-SEARCH-STARTED-S     requested (translation and grounding done), at which it actually
    SOLVER-SEARCH-ENDED-S       started (clasp's preparation of the ground program done), and at
                                which it returned. None of the time before SEARCH-STARTED can be
                                interrupted, so SEARCH-STARTED after the deadline means the
                                deadline passed during it; SEARCH-ENDED minus SOLVER-DEADLINE-S is
                                how long stopping took.
    SOLVER-MAX-TIME             the horizon (--max-time, extended by the re-solve loop) of the last
                                search -- the one the statistics on this line come from.
    SOLVER-HORIZON-FINAL        false when the statistics may belong to a horizon the run would
                                NOT have ended at.

    Why the last one is needed. Under full ground delay without overloads allowed, main() re-solves
    with max_time + 1 while the model found still has overloads. The optimum of a longer horizon
    can be lexicographically LOWER than a shorter one's (it can remove overloads the shorter one
    could not), so a lower bound, or a proof of optimality, obtained at a horizon the loop would
    have extended past is not a bound on, or the optimum of, the program a complete run solves.
    The objective values of the reported model are still a feasible incumbent. The flag is false
    exactly when that cannot be ruled out:
      - the loop applies, and the reported model is not from the last search (that search, at a
        longer horizon, was stopped before finding one), or
      - the loop applies, and a longer horizon was wanted but not searched, or
      - the loop applies, and the last search was stopped with an incumbent that still has
        overloads, so whether it would have ended with overloads is unknown.
    """
    loop_applies = args.regulation_ground_delay_active == 2 and args.allow_overloads is False
    from_last_solve = model is not None and model is solver.final_model
    horizon_final = (not loop_applies) or (
        from_last_solve and not deadline_cut
        and (model.get_total_overload() == 0 or not solver.stopped_at_deadline))
    return {
        "SOLVER-STOPPED-AT-DEADLINE": bool(solver.stopped_at_deadline or deadline_cut),
        "SOLVER-DEADLINE-S": args.solve_deadline,
        "SOLVER-GROUNDING-ENDED-S": _since(solver.solve_started_at, process_start),
        "SOLVER-SEARCH-STARTED-S": _since(solver.search_started_at, process_start),
        "SOLVER-SEARCH-ENDED-S": _since(solver.solve_ended_at, process_start),
        "SOLVER-MAX-TIME": last_max_time,
        "SOLVER-HORIZON-FINAL": bool(horizon_final),
    }


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
    regulation_dynamic_sectorization_active = args.regulation_dynamic_sectorization

    allow_overloads = args.allow_overloads

    arrival_delay_metric = args.arrival_delay_metric

    # --parallel-mode=<n>, then the profile's flags, then any raw --solver-arg. With the
    # defaults this is ["--parallel-mode=1"], which clingo configures identically to no flags
    # at all -- verified across every configuration key it exposes.
    solver_options = solver_options_from_args(args, threads=args.number_threads)

    seed = args.seed

    timestep_granularity = args.timestep_granularity

    max_time = args.max_time

    experiment_name = _derive_output_name(args)
    if verbosity > 0:
        print(f"    clingo solver:   "
              f"{describe_solver_options(args.solver_profile, args.solver_arg, args.number_threads)}")
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
    original_max_time = max_time

    # --solve-deadline. Without the flag deadline_at stays None and none of the bookkeeping below
    # takes part in anything.
    process_start = _process_start_monotonic() if args.solve_deadline is not None else None
    deadline_at = (process_start + args.solve_deadline) if process_start is not None else None
    last_solver = None           # the Solver of the most recent search
    last_max_time = None         # the horizon that search ran at
    best_model = None            # the most recent model the re-solve loop set aside for a
    best_max_time = None         #   longer horizon, and the horizon it was found at
    setup_seconds = 0.0          # translation + grounding + clasp preparation, last iteration
    deadline_cut = False         # a longer horizon was wanted, and the deadline did not allow it

    while model is None:

        if deadline_at is not None and best_model is not None \
                and deadline_at - _time.monotonic() <= RESOLVE_SETUP_GROWTH * setup_seconds:
            # The re-solve loop wants a longer horizon, but translating, grounding and preparing
            # it -- none of which can be interrupted -- took setup_seconds last time, a longer
            # horizon takes longer, and there is not enough left. Starting it would at best leave
            # no time to search, and at worst run into the external kill and lose the result in
            # hand. Report the model we have instead.
            deadline_cut = True
            break
        iteration_started_at = _time.monotonic()

        transalte_to_logic_program = TranslateCSVtoLogicProgram()
        asp_instance = transalte_to_logic_program.main(graph_csv, flights_csv, sectors_csv,
            airports_csv, airplanes_csv, airplane_flight_csv, navaid_sector_csv, encoding_path, timestep_granularity, max_time,
            sector_capacity_factor,
            regulation_ground_delay_active, regulation_rerouting_active, regulation_dynamic_sectorization_active)
        
        instance_asp_atoms = "\n".join(asp_instance)

        encoding = open(encoding_path, "r").read()
        # Select the arrival-delay metric the encoding should score.
        encoding += asp_metric_fact(arrival_delay_metric)

        if args.export_instance_lp is not None:
            args.export_instance_lp.parent.mkdir(parents=True, exist_ok=True)
            args.export_instance_lp.write_text(instance_asp_atoms + "\n", encoding="utf-8")
            print(f"[export] instance facts -> {args.export_instance_lp} "
                  f"({len(asp_instance):,} atoms)", flush=True)

        if args.export_program_lp is not None:
            args.export_program_lp.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f"% ASPaeroFlow instance, exported by 02_ASP/main.py\n"
                f"% encoding:              {encoding_path}\n"
                f"% arrival-delay metric:  {arrival_delay_metric}\n"
                f"% timestep granularity:  {timestep_granularity}\n"
                f"% max time:              {max_time}\n"
                f"% sector capacity factor:{sector_capacity_factor}\n"
                f"% regulations: ground-delay={regulation_ground_delay_active} "
                f"rerouting={regulation_rerouting_active} "
                f"dynamic-sectorization={regulation_dynamic_sectorization_active}\n"
                f"% Self-contained: run with `clingo {args.export_program_lp.name}`.\n\n")
            args.export_program_lp.write_text(header + encoding + "\n" + instance_asp_atoms + "\n",
                                              encoding="utf-8")
            print(f"[export] self-contained program -> {args.export_program_lp}", flush=True)

        if args.export_only and (args.export_instance_lp is not None
                                 or args.export_program_lp is not None):
            print("[export] --export-only given; not solving.", flush=True)
            return 0

        solver: Model = Solver(encoding, instance_asp_atoms, seed=seed, wandb_log = wandb_log,
                               solver_options=solver_options,
                               report_solver_stats=args.solver_stats,
                               deadline=deadline_at)
        model = solver.solve()

        if deadline_at is not None:
            last_solver, last_max_time = solver, max_time
            if solver.search_started_at is not None:
                # Translation, grounding and clasp's preparation: everything a longer horizon
                # would have to redo before its search could even be stopped.
                setup_seconds = solver.search_started_at - iteration_started_at
            if model is None and solver.stopped_at_deadline:
                # Stopped before a first model at this horizon. Whatever there is to report --
                # the model of a shorter horizon, or only the bound -- is reported after the loop.
                break

        if model is None:
            # clingo returned without ever calling on_model: unsatisfiable, or cancelled before
            # a first model. More likely under --solver-profile usc, whose first model can take
            # seconds. Without this the next line fails with an opaque AttributeError on None.
            raise RuntimeError(
                "the solver returned no model "
                f"(solver options: "
                f"{describe_solver_options(args.solver_profile, args.solver_arg, args.number_threads)}). "
                "The program is unsatisfiable, or the search was stopped before a first model.")
            
        if verbosity > 0:
            print(f"""
        Result of Answer:
        - Overload: {model.get_total_overload()}
        - ATFM Delay: {model.get_total_atfm_delay()} (metric: {arrival_delay_metric})
        - Computation time: {model.computation_time}s
        - Rerouted Airplanes: {model.get_rerouted_airplanes()}
            """)

        #quit()
        if model.get_total_overload() > 0 and allow_overloads is False:
            if regulation_ground_delay_active == 2:
                if deadline_at is not None:
                    # Kept, so that a deadline arriving before the longer horizon produces a model
                    # still has something to report.
                    best_model, best_max_time = model, max_time
                max_time += 1
                model = None

        number_flights = transalte_to_logic_program.airplane_flight.shape[0]
        up_bound_by_thm = (timestep_granularity * original_max_time) + int((1/2) * (timestep_granularity * original_max_time)  * number_flights * (number_flights + 1))
        if max_time > up_bound_by_thm:
            break

    if deadline_at is not None:
        # The result line is printed HERE, before any post-processing, and flushed. Without a
        # deadline a run that overruns is killed and never gets this far; with one, every stopped
        # run does, and the matrix building below can still fail (it is known to be able to raise
        # IndexError). Printed first, the line -- lower bound included -- is already in the
        # caller's hands whatever happens next: a later exception makes the caller mark THIS line
        # with ERROR=E, a later kill with T or M, and neither removes it.
        if model is None and best_model is not None:
            model, max_time = best_model, best_max_time
        fields = _deadline_fields(model, last_solver, args, process_start, last_max_time,
                                  deadline_cut)
        if model is None:
            # Nothing to report but the solver's own statistics. Still printed, because under
            # --solver-stats they can hold a lower bound; then the run fails as it always has
            # when no model was found.
            line = {"GROUNDING-TIME": getattr(last_solver, "grounding_time", None),
                    "COMPUTATION-FINISHED": bool(last_solver.search_exhausted)}
            line.update(last_solver.solve_summary or {})
            line.update(fields)
            print(json.dumps(line), flush=True)
            raise RuntimeError(
                "the solver returned no model before the solve deadline of "
                f"{args.solve_deadline} s after process start (solver options: "
                f"{describe_solver_options(args.solver_profile, args.solver_arg, args.number_threads)}).")
        if model is not last_solver.final_model:
            # The model is from a shorter horizon, and the last search -- at a longer one -- was
            # stopped before finding any. Every other key on the line describes that last search,
            # as it does in every other case, so COMPUTATION-FINISHED and the solver summary are
            # taken from it; SOLVER-HORIZON-FINAL is false.
            model.computation_finished = last_solver.search_exhausted
            if last_solver.solve_summary:
                model.set_solver_summary(last_solver.solve_summary)
        model.set_solver_summary(fields)
        print(model.get_model_optimization_string(), flush=True)

    if verbosity > 0:
        print(f"""
    Result of Answer:
    - ATFM Delay: {model.get_total_atfm_delay()} (metric: {arrival_delay_metric})
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

    # The result matrices must reach every timestep a flight uses. They were sized from the
    # navaid_sector atoms alone, which the encoding does not show, so a model with a flight after
    # timestep --max-time + 1 (any T_gran above 1) raised an IndexError after solving. Widen only
    # when needed, so every matrix that fitted before keeps its shape.
    for atom in list(flights) + list(model.get_navpoint_flights()):
        max_time = max(max_time, int(str(atom.arguments[2])) - 1)

    max_time += 1

    converted_instance_matrix = np.ones((len(distinct_flights),max_time+1)) * -1

    for flight in flights:
        flight_id = int(str(flight.arguments[0]))
        flight_sector = int(str(flight.arguments[1]))
        flight_time = int(str(flight.arguments[2]))

        converted_instance_matrix[flight_id, flight_time] = flight_sector

    converted_navpoint_matrix = np.ones((len(distinct_flights),max_time+1)) * -1

    for navpoint_flight in model.get_navpoint_flights():
        flight_id = int(str(navpoint_flight.arguments[0]))
        flight_navaid = int(str(navpoint_flight.arguments[1]))
        flight_time = int(str(navpoint_flight.arguments[2]))

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

    # COMPUTATION-FINISHED now comes from clingo's SolveResult.exhausted, set in Solver.solve().
    # It used to be forced True here, which made the field meaningless: every completed run said
    # "finished" whether or not the optimum had been proven, and every line a timeout could see
    # said False. Do not reinstate the assignment.
    if deadline_at is None:          # with a deadline it was printed straight after the search
        print(model.get_model_optimization_string())
    #np.savetxt(sys.stdout, converted_instance_matrix, delimiter=",", fmt="%i") 

    # Save results if requested
    if args.save_results:
        _save_results(args, model_data)

    if run is not None:
        run.finish()

    return model.get_total_atfm_delay()

if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


