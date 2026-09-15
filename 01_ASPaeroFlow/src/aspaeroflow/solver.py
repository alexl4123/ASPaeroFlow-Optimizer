# solver.py
# Author: Alexander Beiser

import os
import sys
import time
import contextlib
import operator

import clingo

from typing import Final

ARRIVAL_DELAY: Final[str] = "arrivalDelay"
SECTOR_FLIGHT: Final[str] = "sector_flight"
NAVPOINT_FLIGHT: Final[str] = "navpoint_flight"
REROUTED: Final[str] = "reroute"
SECTOR_CONFIG: Final[str] = "chosen_config"
SIGNATURES: Final[set[str]] = {ARRIVAL_DELAY, SECTOR_FLIGHT, NAVPOINT_FLIGHT, REROUTED, SECTOR_CONFIG}

#: The seed this file used to hard-code. It is kept as the fallback so that a caller which does
#: not pass a seed produces exactly the runs it always did. It is also the default of
#: 01_ASPaeroFlow/main.py's --seed and of the benchmark caller's build_command(), so the value
#: reaching clingo does not move for any existing invocation -- but --seed now actually reaches
#: these sub-solves, which it never did before.
LEGACY_SEED: Final[int] = 11904657

class Solver:
    def __init__(self, encoding, instance, seed = None, solver_options = None):

        self.encoding = encoding
        self.instance = instance

        # None => the seed this file hard-coded. The heuristic's ASP sub-calls used to run at
        # 11904657 no matter what --seed the experiment asked for.
        self.seed = LEGACY_SEED if seed is None else int(seed)
        # Extra clingo flags chosen by name; see common/clingo_options.py. Empty by default.
        self.solver_options = list(solver_options) if solver_options else []

        self.final_model = None


    def solve(self):

        
        self.final_model = None

        start_time = time.time()

        ctl = clingo.Control([f"--seed={self.seed}"] + self.solver_options)

        ##########################################################
        # SILENCE CLINGO (all stdout/warnings directly to devnull):
        fd = sys.stdout.fileno()
        fd2 = sys.stderr.fileno()
        saved_fd = os.dup(fd)  
        saved_fd2 = os.dup(fd2)  

        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)     #  <-- redirect at the FD level
            os.dup2(devnull.fileno(), fd2)     #  <-- redirect at the FD level
            try:
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                    ctl.add("base",[],self.encoding + self.instance)
                    ctl.ground([("base",[])])
                    ctl.solve(on_model=self.on_model)
            finally:
                os.dup2(saved_fd2,fd2)
                os.close(saved_fd2)
                os.dup2(saved_fd, fd)         #  <-- restore
                os.close(saved_fd)
                pass
        ##########################################################

        end_time = time.time()

        runtime = end_time - start_time
        self.final_model.set_computation_time(runtime)

        return self.final_model

    def on_model(self, model):

        parsed = [symbol for symbol in model.symbols(atoms=True) if symbol.name in SIGNATURES]

        arrival_delays = [symbol for symbol in parsed if symbol.name == ARRIVAL_DELAY]

        sector_flights = [symbol for symbol in parsed if symbol.name in SECTOR_FLIGHT]
        navpoint_flights = [symbol for symbol in parsed if symbol.name in NAVPOINT_FLIGHT]

        reroutes = [symbol for symbol in parsed if symbol.name in REROUTED]
        sector_configs = [symbol for symbol in parsed if symbol.name in SECTOR_CONFIG]

        if len(sector_configs) > 1:
            raise Exception("Found multiple sector-config atoms in ASP output - must never happen!")
        
        sector_config = sector_configs[0]
        self.final_model = Model(sector_flights, navpoint_flights, reroutes, arrival_delays, sector_config)


class PickleAbleSymbol:

    def __init__(self, symbol):

        self.name = symbol.name
        self.arguments = []

        for argument in symbol.arguments:
            self.arguments.append(str(argument))

    def __str__(self):
        if not self.arguments:
            return self.name
        
        args_str = ",".join(self.arguments)
        return f"{self.name}({args_str})"


class Model:

    def __init__(self, sector_flights, navpoint_flights, reroutes, atfm_delays, sector_config):

        self.sector_flights = [PickleAbleSymbol(sector_flight) for sector_flight in sector_flights]
        self.navpoint_flights = [PickleAbleSymbol(navpoint_flight) for navpoint_flight in navpoint_flights]

        self.reroutes = [PickleAbleSymbol(reroute) for reroute in reroutes]
        self.atfm_delays = [PickleAbleSymbol(atfm_delay) for atfm_delay in atfm_delays]
        self.sector_config = PickleAbleSymbol(sector_config)

        self.computation_time = -1

    def get_sector_flights(self):
        return self.sector_flights
    
    def get_navpoint_flights(self):
        return self.navpoint_flights

    def get_total_atfm_delay(self):

        total = sum(operator.attrgetter("arguments")(arrival_delay)[1].number for arrival_delay in self.atfm_delays)

        return total

    def get_rerouted_airplanes(self):

        airplanes = [str(symbol.arguments[0]) for symbol in self.reroutes]

        return airplanes
    
    def set_computation_time(self, runtime):
        self.computation_time = round(runtime,2)

    def get_reroutes(self):
        return self.reroutes
    
    def get_sector_config(self):
        return self.sector_config

