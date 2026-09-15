# solver.py
# Author: Alexander Beiser

import os
import sys
import time
import contextlib
import operator
import json

import clingo

from typing import Final

OVERLOAD: Final[str] = "overload"
ARRIVAL_DELAY: Final[str] = "arrival_delay"
SECTOR_NUMBER: Final[str] = "sector_number"
SECTOR_DIFF: Final[str] = "sector_diff"
REROUTE: Final[str] = "reroute"
RECONFIG: Final[str] = "reconfig"

FLIGHT: Final[str] = "flight"
NAVPOINT_FLIGHT: Final[str] = "navpoint_flight"
NAVAID_SECTOR: Final[str] = "navaid_sector"
SIGNATURES: Final[set[str]] = {ARRIVAL_DELAY, FLIGHT, REROUTE, NAVPOINT_FLIGHT, NAVAID_SECTOR, OVERLOAD, SECTOR_NUMBER, SECTOR_DIFF, RECONFIG}

def _solver_summary(ctl, models_reported):
    """Clingo's own view of the finished search.

    Reported only under --solver-stats, and worth having under --opt-strategy=usc: core-guided
    search can report ONE model and then spend the rest of the budget raising the LOWER bound
    with nothing to show on stdout (measured: 1 on_model call in 25 s on 20260915_test2.lp with
    usc-domain). SOLVER-LOWER-BOUND is where that work is visible, and it is what says whether an
    incumbent is near-optimal or merely the first thing found.

    SOLVER-OPTIMALITY-PROVEN is clingo's per-model flag and is ALWAYS false under the default
    --opt-mode=opt, even for a run that exhausted the search space; SOLVER-EXHAUSTED and
    SOLVER-OPTIMAL-MODELS are the fields that actually answer "was this proved optimal?".
    """
    summary = {"SOLVER-MODELS-REPORTED": models_reported}
    try:
        stats = ctl.statistics.get("summary", {})
        models = stats.get("models", {})
        summary["SOLVER-COSTS"] = stats.get("costs")
        summary["SOLVER-LOWER-BOUND"] = stats.get("lower")
        summary["SOLVER-MODELS-ENUMERATED"] = models.get("enumerated")
        summary["SOLVER-OPTIMAL-MODELS"] = models.get("optimal")
        summary["SOLVER-EXHAUSTED"] = bool(models.get("optimal"))
    except Exception:                       # statistics are diagnostics; never fail a solve
        pass
    return summary


class Solver:
    def __init__(self, encoding, instance, seed = 1, wandb_log = None,
                 solver_options = None, report_solver_stats = False):
        self.encoding = encoding
        self.instance = instance
        self.seed = seed
        self.wandb_log = wandb_log

        # Extra clingo flags, chosen by name via --solver-profile (see common/clingo_options.py).
        # An empty list is what this class always used: clingo.Control([]) is clingo.Control().
        self.solver_options = list(solver_options) if solver_options else []
        # Emit the solver diagnostics below as extra JSON keys. OFF by default, so the reported
        # line is byte-for-byte what it has always been and a running benchmark keeps parsing it.
        self.report_solver_stats = bool(report_solver_stats)

        self.final_model = None

        # Best cost seen across ALL on_model calls, not merely the last one. clasp reports a
        # monotonically improving sequence under branch-and-bound AND under --opt-strategy=usc,
        # measured on 02_ASP/20260915_test{,2}.lp at 1 and 4 threads, so this never fires today.
        # It is here so that a strategy which ever reported a non-improving model could not
        # silently replace a good incumbent with a worse one at a timeout.
        self.best_cost = None
        self.models_reported = 0


    def solve(self):
        
        self.final_model = None
        self.best_cost = None
        self.models_reported = 0

        start_time = time.time()
        self.total_time_start = start_time

        ctl = clingo.Control(self.solver_options)
        ctl.configuration.solver.seed = self.seed

        ##########################################################
        # SILENCE CLINGO (all stdout/warnings directly to devnull):
        fd = sys.stdout.fileno()
        fd2 = sys.stderr.fileno()
        saved_fd = os.dup(fd)  
        saved_fd2 = os.dup(fd2)  

        self.tmp_fd = saved_fd

        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)     #  <-- redirect at the FD level
            os.dup2(devnull.fileno(), fd2)     #  <-- redirect at the FD level
            try:
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                    ctl.add("base",[], self.encoding + self.instance )

                    grd_time_start = time.time()
                    ctl.ground([("base",[])])
                    grd_time_end = time.time()

                    self.grounding_time = grd_time_end - grd_time_start

                    ctl.solve(on_model=self.on_model)
            finally:
                os.dup2(saved_fd2,fd2)
                os.close(saved_fd2)
                saved_fd = self.tmp_fd
                os.dup2(saved_fd, fd)
                os.close(saved_fd)
                pass
        ##########################################################

        end_time = time.time()

        runtime = end_time - start_time

        if self.final_model is None:
            return None
        
        self.final_model.set_computation_time(runtime)
        if self.report_solver_stats:
            self.final_model.set_solver_summary(_solver_summary(ctl, self.models_reported))

        return self.final_model

    def on_model(self, model):

        # Keep the BEST model, not just the most recent one. See the note on self.best_cost.
        cost = tuple(model.cost)
        self.models_reported += 1
        if self.best_cost is not None and cost > self.best_cost:
            return
        self.best_cost = cost

        parsed = [symbol for symbol in model.symbols(atoms=True) if symbol.name in SIGNATURES]


        flights = [symbol for symbol in parsed if symbol.name in FLIGHT]
        navpoint_flights = [symbol for symbol in parsed if symbol.name in NAVPOINT_FLIGHT]
        navaid_sector_time = [symbol for symbol in parsed if symbol.name in NAVAID_SECTOR]

        overload = [symbol for symbol in parsed if symbol.name == OVERLOAD]
        arrival_delay = [symbol for symbol in parsed if symbol.name == ARRIVAL_DELAY]
        sector_number = [symbol for symbol in parsed if symbol.name == SECTOR_NUMBER]
        sector_diff = [symbol for symbol in parsed if symbol.name == SECTOR_DIFF]
        reroute = [symbol for symbol in parsed if symbol.name in REROUTE]
        reconfig = [symbol for symbol in parsed if symbol.name in RECONFIG]
        
        current_time = time.time() - self.total_time_start
        self.final_model = Model(overload, arrival_delay, sector_number, sector_diff, reroute, reconfig, flights, navpoint_flights, navaid_sector_time, self.grounding_time, current_time, model.optimality_proven)
        if self.report_solver_stats:
            self.final_model.set_solver_summary({
                "SOLVER-COST": list(model.cost),
                "SOLVER-MODEL-INDEX": model.number,
                "SOLVER-MODELS-REPORTED": self.models_reported,
                "SOLVER-OPTIMALITY-PROVEN": model.optimality_proven,
            })
        
        output_string = self.final_model.get_model_optimization_string()
        tmp = os.dup(self.tmp_fd)
        with os.fdopen(self.tmp_fd, 'w') as fdopen:
            fdopen.write(output_string + "\n")
            fdopen.close()
        self.tmp_fd = tmp
        
        if self.wandb_log:
            self.wandb_log({
                "OVERLOAD":int(self.final_model.get_total_overload()),
                "ARRIVAL-DELAY":int(self.final_model.get_total_atfm_delay()),
                "SECTOR-NUMBER":int(self.final_model.get_total_sector_number()),
                "SECTOR_DIFF":int(self.final_model.get_total_sector_diff()),
                "REROUTE":int(self.final_model.get_total_reroute()),
                "RECONFIG":int(self.final_model.get_total_reconfig()),
                "GROUNDING-TIME": int(self.grounding_time),
                "TOTAL-TIME-TO-THIS-POINT": int(current_time)
                })

class Model:

    def __init__(self, overloads, arrival_delay, sector_number, sector_diff, reroute, reconfig, flights, navpoint_flights, navaid_sector_time, grounding_time, current_time, computation_finished):

        self.overloads = overloads
        self.arrival_delays = arrival_delay
        self.sector_numbers = sector_number
        self.sector_diffs = sector_diff
        self.reroutes = reroute
        self.reconfig = reconfig

        self.computation_time = -1

        self.flights = flights

        self.navpoint_flights = navpoint_flights
        self.navaid_sector_time = navaid_sector_time

        self.grounding_time = grounding_time
        self.current_time = current_time
        self.computation_finished = computation_finished

        # Optional clingo diagnostics, emitted only under --solver-stats. None => the JSON line
        # below is exactly the six objective keys plus timings that it has always been.
        self.solver_summary = None

    def set_solver_summary(self, summary):
        """Merge clingo diagnostics into this model (see Solver.report_solver_stats)."""
        if not summary:
            return
        if self.solver_summary is None:
            self.solver_summary = {}
        self.solver_summary.update(summary)

    def get_model_optimization_string(self):

        output_dict = {}
        output_dict["OVERLOAD"] = self.get_total_overload()
        output_dict["ARRIVAL-DELAY"] = self.get_total_atfm_delay()
        output_dict["SECTOR-NUMBER"] = self.get_total_sector_number()
        output_dict["SECTOR-DIFF"] = self.get_total_sector_diff()
        output_dict["REROUTE"] = self.get_total_reroute()
        output_dict["RECONFIG"] = self.get_total_reconfig()
        output_dict["GROUNDING-TIME"] = self.grounding_time
        output_dict["TOTAL-TIME-TO-THIS-POINT"] = self.current_time
        output_dict["COMPUTATION-FINISHED"] = self.computation_finished
        if self.solver_summary:
            output_dict.update(self.solver_summary)

        output_string = json.dumps(output_dict)

        return output_string


    def get_total_overload(self):

        overload_sum = sum([int(symbol.arguments[2].number) for symbol in self.overloads])
        return overload_sum
    
    def get_total_reconfig(self):

        total = sum(1 for reroute in self.reconfig)
        return total

    def get_total_atfm_delay(self):

        total = sum(operator.attrgetter("arguments")(arrival_delay)[1].number for arrival_delay in self.arrival_delays)
        return total


    def get_total_sector_number(self):

        total = sum(operator.attrgetter("arguments")(sector_number)[1].number for sector_number in self.sector_numbers)
        return total

    def get_total_sector_diff(self):

        total = sum(operator.attrgetter("arguments")(sector_diff)[1].number for sector_diff in self.sector_diffs)
        return total

    def get_total_reroute(self):

        total = sum(1 for reroute in self.reroutes)
        return total

    def get_rerouted_airplanes(self):

        airplanes = [str(symbol.arguments[0]) for symbol in self.reroutes]

        return airplanes
    
    
    def set_computation_time(self, runtime):
        self.computation_time = round(runtime,2)

    def get_flights(self):
        return self.flights
    
    def get_navpoint_flights(self):
        return self.navpoint_flights
    
    def get_navaid_sector_time_assignment(self):
        return self.navaid_sector_time
