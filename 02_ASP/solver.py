# solver.py
# Author: Alexander Beiser

import os
import sys
import time
import contextlib
import operator
import json

import clingo

from collections import Counter
from typing import Final

from navpoint_sector_allocation_bootstrap import series_to_window

OVERLOAD: Final[str] = "overload"
ARRIVAL_DELAY: Final[str] = "arrival_delay"
SECTOR_NUMBER: Final[str] = "sector_number"
SECTOR_DIFF: Final[str] = "sector_diff"
REROUTE: Final[str] = "reroute"
RECONFIG: Final[str] = "reconfig"

FLIGHT: Final[str] = "flight"
NAVPOINT_FLIGHT: Final[str] = "navpoint_flight"

#: The atom encoding.lp derives for the navpoint-to-sector allocation. It is deliberately NOT
#: collected: there is one per (navpoint, timestep), and every improving model would pay for all
#: of them. The name is written out here because the constant it replaces read "navaid_sector",
#: which matches no atom this encoding ever derives -- so the allocation collected was always
#: empty, and nothing said so. 02_ASP/main.py writes the allocation from the instance instead;
#: see the note there.
NAVPOINT_SECTOR: Final[str] = "navpoint_sector"

#: The atom names collected from a model. Test membership with `==` against the constants above,
#: or with `in` against THIS SET -- never with `in` against one of the strings, because
#: `"flight" in "navpoint_flight"` is a true SUBSTRING test. That is how flight/3 atoms used to
#: be collected as navpoint flights and overwrite the saved navpoint matrix with the sector one.
SIGNATURES: Final[frozenset] = frozenset({
    ARRIVAL_DELAY, FLIGHT, REROUTE, NAVPOINT_FLIGHT, OVERLOAD,
    SECTOR_NUMBER, SECTOR_DIFF, RECONFIG,
})

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
                 solver_options = None, report_solver_stats = False,
                 evaluation_window = None, arrival_delay_metric = None):
        self.encoding = encoding
        self.instance = instance
        self.seed = seed
        self.wandb_log = wandb_log

        # The timesteps SECTOR-NUMBER and RECONFIG are scored over: the window the INSTANCE
        # defines, which is what the Python solvers score on too. The ASP time domain is
        # 0..max_time*T_gran, one column short of (max_time+1)*T_gran at T_gran 1 and shorter
        # still above it, and it GROWS when this run retries with a larger max_time -- so without
        # this the same static sectorisation scored 1067 here against 1100 from 03_DELAY.
        self.evaluation_window = evaluation_window

        # Reported as ARRIVAL-DELAY-METRIC, the way 01 and 04 report it, so an ASP result line
        # says which reading of t_actarr - t_exparr produced its ARRIVAL-DELAY.
        self.arrival_delay_metric = arrival_delay_metric

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

        # clingo's SolveResult.exhausted for the last solve(): the search space was closed, so
        # the optimum is PROVEN. This is the only reliable signal for that. Model.optimality_
        # proven is not: it is false on every model under the default --opt-mode=opt, even for a
        # run that exhausted the space, because clingo only flags a model optimal under
        # --opt-mode=optN. Measured both ways before this was written.
        self.search_exhausted = False


    def solve(self):
        
        self.final_model = None
        self.best_cost = None
        self.models_reported = 0
        self.search_exhausted = False

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

                    solve_result = ctl.solve(on_model=self.on_model)
                    self.search_exhausted = bool(solve_result.exhausted)
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
        # Whether this run actually finished, rather than "we got here". The caller used to set
        # this to True unconditionally just before printing, so the field said "finished" for a
        # run that had merely stopped. LIMITATION: the benchmark's time limit arrives as an
        # external SIGKILL, so a run that overruns never reaches this line -- and never prints a
        # final result line either. This makes the field meaningful for runs that COMPLETE within
        # the limit; it cannot rescue the killed ones, whose last visible line is an intermediate
        # model still carrying False.
        self.final_model.computation_finished = self.search_exhausted
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


        # Exact names throughout. `symbol.name in FLIGHT` asked whether the name is a SUBSTRING
        # of "flight", and `in NAVPOINT_FLIGHT` whether it is a substring of "navpoint_flight" --
        # which "flight" is, so every flight/3 atom was also collected as a navpoint flight and
        # the saved converted_navpoint_matrix came out as a copy of the sector matrix.
        flights = [symbol for symbol in parsed if symbol.name == FLIGHT]
        navpoint_flights = [symbol for symbol in parsed if symbol.name == NAVPOINT_FLIGHT]

        overload = [symbol for symbol in parsed if symbol.name == OVERLOAD]
        arrival_delay = [symbol for symbol in parsed if symbol.name == ARRIVAL_DELAY]
        sector_number = [symbol for symbol in parsed if symbol.name == SECTOR_NUMBER]
        sector_diff = [symbol for symbol in parsed if symbol.name == SECTOR_DIFF]
        reroute = [symbol for symbol in parsed if symbol.name == REROUTE]
        reconfig = [symbol for symbol in parsed if symbol.name == RECONFIG]

        current_time = time.time() - self.total_time_start
        self.final_model = Model(overload, arrival_delay, sector_number, sector_diff, reroute, reconfig, flights, navpoint_flights, self.grounding_time, current_time, model.optimality_proven,
                                 evaluation_window=self.evaluation_window,
                                 arrival_delay_metric=self.arrival_delay_metric)
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

    def __init__(self, overloads, arrival_delay, sector_number, sector_diff, reroute, reconfig, flights, navpoint_flights, grounding_time, current_time, computation_finished,
                 evaluation_window=None, arrival_delay_metric=None):

        self.overloads = overloads
        self.arrival_delays = arrival_delay
        self.sector_numbers = sector_number
        self.sector_diffs = sector_diff
        self.reroutes = reroute
        self.reconfig = reconfig

        self.computation_time = -1

        self.flights = flights

        self.navpoint_flights = navpoint_flights

        # See the Solver constructor: the window SECTOR-NUMBER and RECONFIG are scored over, and
        # how the delay they are reported beside was measured.
        self.evaluation_window = evaluation_window
        self.arrival_delay_metric = arrival_delay_metric

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
        # Which reading of t_actarr - t_exparr produced ARRIVAL-DELAY above, as 01 and 04 report
        # it on their final lines. Omitted when nothing was passed, so a caller that builds a
        # Model directly still gets exactly the keys it always got.
        if self.arrival_delay_metric is not None:
            output_dict["ARRIVAL-DELAY-METRIC"] = str(self.arrival_delay_metric)
        if self.solver_summary:
            output_dict.update(self.solver_summary)

        output_string = json.dumps(output_dict)

        return output_string


    def get_total_overload(self):

        overload_sum = sum([int(symbol.arguments[2].number) for symbol in self.overloads])
        return overload_sum
    
    def _asp_time_axis(self):
        """How many timesteps this model covers: the encoding's `time/1` domain.

        `sector_number(T,NUM)` is derived for every `time(T)`, so the highest T among those atoms
        IS the axis. A count series like reconfig cannot say this for itself -- it is silent
        wherever the count is zero -- which is why both metrics ask here.
        """
        if not self.sector_numbers:
            return None
        return max(symbol.arguments[0].number for symbol in self.sector_numbers) + 1

    def get_total_reconfig(self):

        # One per (navpoint, timestep) that differs from the reference allocation, summed over
        # the instance's window rather than over this run's own time axis. See the note in the
        # Solver constructor; padding repeats the last column's cells, which is what a persisting
        # reconfiguration means.
        per_timestep = Counter(symbol.arguments[1].number for symbol in self.reconfig)
        total = sum(series_to_window(per_timestep, self.evaluation_window,
                                     own_width=self._asp_time_axis()))
        return int(total)

    def get_total_atfm_delay(self):

        total = sum(operator.attrgetter("arguments")(arrival_delay)[1].number for arrival_delay in self.arrival_delays)
        return total


    def get_total_sector_number(self):

        # The open sectors per timestep, summed over the instance's window rather than over this
        # run's own time axis; see the note in the Solver constructor.
        per_timestep = {symbol.arguments[0].number: symbol.arguments[1].number
                        for symbol in self.sector_numbers}
        total = sum(series_to_window(per_timestep, self.evaluation_window,
                                     own_width=self._asp_time_axis()))
        return int(total)

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
