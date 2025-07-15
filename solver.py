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
REROUTED: Final[str] = "reroute"
SIGNATURES: Final[set[str]] = {ARRIVAL_DELAY, FLIGHT, REROUTED}

class Solver:
    def __init__(self, encoding, instance):

        self.encoding = encoding
        self.instance = instance

        self.final_model = None


    def solve(self):

        
        self.final_model = None

        start_time = time.time()

        ctl = clingo.Control()

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
        print("FOUND MODEL")

        parsed = [symbol for symbol in model.symbols(atoms=True) if symbol.name in SIGNATURES]

        arrival_delays = [symbol for symbol in parsed if symbol.name == ARRIVAL_DELAY]
        flights = [symbol for symbol in parsed if symbol.name in FLIGHT]
        reroutes = [symbol for symbol in parsed if symbol.name in REROUTED]

        self.final_model = Model(flights, reroutes, arrival_delays)


class PickleAbleSymbol:

    def __init__(self, symbol):

        self.name = symbol.name
        self.arguments = []

        for argument in symbol.arguments:
            self.arguments.append(str(argument))


class Model:

    def __init__(self, flights, reroutes, atfm_delays):
        self.flights = [PickleAbleSymbol(flight) for flight in flights]
        self.reroutes = [PickleAbleSymbol(reroute) for reroute in reroutes]
        self.atfm_delays = [PickleAbleSymbol(atfm_delay) for atfm_delay in atfm_delays]
        self.computation_time = -1

    def get_flights(self):
        return self.flights

    def get_total_atfm_delay(self):

        total = sum(operator.attrgetter("arguments")(arrival_delay)[1].number for arrival_delay in self.atfm_delays)

        return total

    def get_rerouted_airplanes(self):

        airplanes = [str(symbol.arguments[0]) for symbol in self.reroutes]

        return airplanes
    
    def set_computation_time(self, runtime):
        self.computation_time = round(runtime,2)

