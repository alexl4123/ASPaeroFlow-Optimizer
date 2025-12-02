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
FLIGHT: Final[str] = "flight"
NAVPOINT_FLIGHT: Final[str] = "navpointFlight"
NAVAID_SECTOR: Final[str] = "navaid_sector"
REROUTED: Final[str] = "reroute"
OVERLOAD: Final[str] = "overload"
SIGNATURES: Final[set[str]] = {ARRIVAL_DELAY, FLIGHT, REROUTED, NAVPOINT_FLIGHT, NAVAID_SECTOR, OVERLOAD}

class Solver:
    def __init__(self, encoding, instance, seed = 1):
        self.encoding = encoding
        self.instance = instance
        self.seed = seed

        self.final_model = None


    def solve(self):
        
        self.final_model = None

        start_time = time.time()

        ctl = clingo.Control()
        ctl.configuration.solver.seed = self.seed

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
                    ctl.add("base",[], self.encoding + self.instance )
                    ctl.ground([("base",[])])
                    ctl.solve(on_model=self.on_model)
            finally:
                os.dup2(saved_fd2,fd2)
                os.close(saved_fd2)
                os.dup2(saved_fd, fd)
                os.close(saved_fd)
                pass
        ##########################################################

        end_time = time.time()

        runtime = end_time - start_time

        if self.final_model is None:
            return None
        
        self.final_model.set_computation_time(runtime)

        return self.final_model

    def on_model(self, model):

        parsed = [symbol for symbol in model.symbols(atoms=True) if symbol.name in SIGNATURES]

        overload = [symbol for symbol in parsed if symbol.name == OVERLOAD]

        arrival_delays = [symbol for symbol in parsed if symbol.name == ARRIVAL_DELAY]
        flights = [symbol for symbol in parsed if symbol.name in FLIGHT]
        navpoint_flights = [symbol for symbol in parsed if symbol.name in NAVPOINT_FLIGHT]
        navaid_sector_time = [symbol for symbol in parsed if symbol.name in NAVAID_SECTOR]
        reroutes = [symbol for symbol in parsed if symbol.name in REROUTED]

        self.final_model = Model(overload, flights, reroutes, arrival_delays, navpoint_flights, navaid_sector_time)


class Model:

    def __init__(self, overloads, flights, reroutes, atfm_delays, navpoint_flights, navaid_sector_time):
        self.flights = flights
        self.reroutes = reroutes
        self.atfm_delays = atfm_delays
        self.computation_time = -1

        self.navpoint_flights = navpoint_flights
        self.navaid_sector_time = navaid_sector_time

        self.overloads = overloads

    def get_total_atfm_delay(self):

        total = sum(operator.attrgetter("arguments")(arrival_delay)[1].number for arrival_delay in self.atfm_delays)

        return total

    def get_rerouted_airplanes(self):

        airplanes = [str(symbol.arguments[0]) for symbol in self.reroutes]

        return airplanes
    
    def get_total_overload(self):

        overload_sum = sum([int(symbol.arguments[2].number) for symbol in self.overloads])

        return overload_sum
    
    def set_computation_time(self, runtime):
        self.computation_time = round(runtime,2)

    def get_flights(self):
        return self.flights
    
    def get_navpoint_flights(self):
        return self.navpoint_flights
    
    def get_navaid_sector_time_assignment(self):
        return self.navaid_sector_time
