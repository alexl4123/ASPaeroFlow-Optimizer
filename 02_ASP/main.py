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
        description="Highly efficient ATFM problem solver for the network manager - including XAI.",
    )

    parser.add_argument(
        "--graph-path",
        type=Path,
        metavar="FILE",
        help="Location of the graph CSV file.",
    )
    parser.add_argument(
        "--sectors-path",
        type=Path,
        metavar="FILE",
        help="Location of the sector (capacity) CSV file.",
    )
    parser.add_argument(
        "--flights-path",
        type=Path,
        metavar="FILE",
        help="Location of the flights CSV file.",
    )
    parser.add_argument(
        "--airports-path",
        type=Path,
        metavar="FILE",
        help="Location of the airport-vertices CSV file.",
    )
    parser.add_argument(
        "--airplanes-path",
        type=Path,
        metavar="FILE",
        help="Location of the airplanes CSV file.",
    )
    parser.add_argument(
        "--airplane-flight-path",
        type=Path,
        metavar="FILE",
        help="Location of the airplane-flight-assignment CSV file.",
    )
    parser.add_argument(
        "--navaid-sector-path",
        type=Path,
        metavar="FILE",
        help="Location of the navaids-sector assignment CSV file.",
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
        help="Number of parallel ASP solving threads."
    )
    parser.add_argument(
        "--timestep-granularity",
        type=int,
        default=1,
        help="Specifies how long one stimulated timestep is (time-step=1h/granularity). So granularity=1 means 1h, granularity=4 means 15 minutes."
    )
    parser.add_argument(
        "--max-explored-vertices",
        type=int,
        default=6,
        help="Maximum vertices explored in parallel - effectively restricts number of explored paths (larger=possibly better solution, but more compute time needed)."
    )
    parser.add_argument(
        "--max-delay-per-iteration",
        type=int,
        default=-1,
        help="Maximum hours of delay per solve iteration explored (larger=more compute time, but faster descent; -1 is automatically fetch  according to max. time steps)."
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
        help="Verbosity levels (0,1,2)"
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

    graph_csv = args.graph_path
    sectors_csv = args.sectors_path
    flights_csv = args.flights_path
    airports_csv = args.airports_path
    airplanes_csv = args.airplanes_path
    airplane_flight_csv = args.airplane_flight_path
    navaid_sector_csv = args.navaid_sector_path

    encoding_path = args.encoding_path
    verbosity = args.verbosity

    seed = args.seed

    timestep_granularity = args.timestep_granularity

    max_time = args.max_time

    asp_instance = TranslateCSVtoLogicProgram().main(graph_csv, flights_csv, sectors_csv,
        airports_csv, airplanes_csv, airplane_flight_csv, navaid_sector_csv, encoding_path, timestep_granularity, max_time)
    
    instance_asp_atoms = "\n".join(asp_instance)

    #open("20250827_instance.lp","w").write(instance_asp_atoms)

    encoding = open(encoding_path, "r").read()

    model = None

    while model is None:

        solver: Model = Solver(encoding, instance_asp_atoms, seed=seed)
        model = solver.solve()
        
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
    #np.savetxt(sys.stdout, converted_instance_matrix, delimiter=",", fmt="%i") 

    return model.get_total_atfm_delay()

if __name__ == "__main__":  # pragma: no cover — direct execution guard
    main()


