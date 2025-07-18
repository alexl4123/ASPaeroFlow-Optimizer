# MAIN.py
# Author: Alexander Beiser

import argparse
import sys
import numpy as np



from solver import Solver, Model

from pathlib import Path
from typing import Any, List, Optional, Final

from translate import TranslateCSVtoLogicProgram


AFFIRMATIVE: Final[set[str]] = {"yes", "y"}
NEGATIVE: Final[set[str]] = {"no", "n", "exit"}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return a fully‑configured :pyclass:`argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        prog="ATFM-NM-Tool",
        description="Solving ATFM with pure ASP - standard encoding.",
    )

    parser.add_argument(
        "--path-graph",
        type=Path,
        default=Path("instances","edges.csv"),  
        metavar="FILE",
        help="Location of the graph CSV file.",
    )
    parser.add_argument(
        "--path-capacity",
        type=Path,
        default=Path("instances","capacity.csv"), 
        metavar="FILE",
        help="Location of the capacity CSV file.",
    )
    parser.add_argument(
        "--path-instance",
        type=Path,
        default=Path("instances","instance_100.csv"), 
        metavar="FILE",
        help="Location of the instance CSV file.",
    )
    parser.add_argument(
        "--airport-vertices-path",
        type=Path,
        default=Path("instances","airport_vertices.csv"), 
        metavar="FILE",
        help="Location of the airport-vertices CSV file.",
    )
    parser.add_argument(
        "--encoding-path",
        type=Path,
        default=Path("encoding.lp"), 
        metavar="FILE",
        help="Location of the encoding for the optimization problem.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11904657,
        help="Set the random see."
    )
    parser.add_argument(
        "--number-threads",
        type=int,
        default=20,
        help="NOT USED."
    )
    parser.add_argument(
        "--timestep-granularity",
        type=int,
        default=1,
        help="Specifies the computation window in timesteps."
    )
    parser.add_argument(
        "--max-explored-vertices",
        type=int,
        default=6,
        help="NOT USED"
    )
    parser.add_argument(
        "--max-delay-per-iteration",
        type=int,
        default=-1,
        help="NOT USED"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=24,
        help="Specifies how many timesteps are one day. "
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity levels (0,1)"
    )

    return parser


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Return parsed command‑line arguments for *argv* (or *sys.argv*)."""
    parser = _build_arg_parser()
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Top‑level script wrapper
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Script entry‑point compatible with both `python -m` and `poetry run`."""
    args = parse_cli(argv)

    edges_csv = args.path_graph
    capacity_csv = args.path_capacity
    instance_csv = args.path_instance
    airport_vertices_csv = args.airport_vertices_path
    encoding_path = args.encoding_path
    verbosity = args.verbosity

    seed = args.seed

    timestep_granularity = args.timestep_granularity

    adjusted_timesteps = args.max_time
    max_time = adjusted_timesteps

    asp_atoms, inferred_max_time = TranslateCSVtoLogicProgram().main(edges_csv,instance_csv, capacity_csv, airport_vertices_csv, timestep_granularity)

    if inferred_max_time > max_time:
        max_time = inferred_max_time

    instance_asp_atoms = "\n".join(asp_atoms)


    encoding = open(encoding_path, "r").read()

    model = None

    while model is None:
        instance = instance_asp_atoms + "\n" + f"maxTimeOriginal({max_time})."

        open("instance_test.lp","w").write(instance)
        solver: Model = Solver(encoding, instance, seed=seed)
        model = solver.solve()
        max_time += adjusted_timesteps
        
    if verbosity > 0:
        print(f"""
    Result of Answer:
    - ATFM Delay: {model.get_total_atfm_delay()}
    - Computation time: {model.computation_time}s
    - Rerouted Airplanes: {model.get_rerouted_airplanes()}
        """)

    flights = model.flights

    distinct_flights = set()
    for flight in flights:
        distinct_flights.add(flight.arguments[0])


    converted_instance_matrix = np.ones((len(distinct_flights),max_time+1)) * -1

    for flight in flights:
        flight_id = int(str(flight.arguments[0]))
        flight_time = int(str(flight.arguments[1]))
        flight_sector = int(str(flight.arguments[2]))

        converted_instance_matrix[flight_id, flight_time] = flight_sector

    if verbosity > 0:
        print(model.get_rerouted_airplanes())

    print(model.get_total_atfm_delay())
    np.savetxt(sys.stdout, converted_instance_matrix, delimiter=",", fmt="%i") 

    return model.get_total_atfm_delay()

if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


