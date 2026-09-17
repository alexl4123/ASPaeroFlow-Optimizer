# solver.py
# Author: Alexander Beiser

import os
import sys
import time
import math
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


def _finite_or_none(vector):
    """A clingo cost or bound vector, or None when it holds anything but finite numbers.

    A solve stopped before its first model reports its costs as [inf, ...]. json.dumps would write
    that as the non-standard token Infinity, and int() on it raises, so it becomes None.
    """
    if vector is None:
        return None
    try:
        values = list(vector)
    except TypeError:
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return None
    return values


class _CostPriorities(clingo.Observer):
    """Records the priority of every minimize statement the grounder emits.

    clingo's cost and lower-bound vectors have one entry per priority level PRESENT in the ground
    program, highest first, and which levels are present depends on the regulation variant and the
    instance: 4 to 6 entries for this encoding (a _ns variant has no @5 level, for example). A bound
    of [16, 300, 0, 0] cannot be put against the six named objective levels without knowing which
    four priorities it stands for, and this is where that comes from.

    Zero-weight levels are kept because clasp keeps them: measured on 13 regulation variants under
    branch-and-bound and usc, the number of priorities seen here always equals the length of
    clingo's cost vector, and the number of levels with a non-zero weight does not.

    Only minimize() is implemented, and clingo registers only the callbacks an observer overrides,
    so nothing else about grounding changes.
    """

    def __init__(self):
        self.priorities = set()

    def minimize(self, priority, literals):
        self.priorities.add(priority)

    def ordered(self):
        return sorted(self.priorities, reverse=True)


class Solver:
    def __init__(self, encoding, instance, seed = 1, wandb_log = None,
                 solver_options = None, report_solver_stats = False, deadline = None):
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

        # clingo's SolveResult.exhausted for the last solve(): the search space was closed, so
        # the optimum is PROVEN. This is the only reliable signal for that. Model.optimality_
        # proven is not: it is false on every model under the default --opt-mode=opt, even for a
        # run that exhausted the space, because clingo only flags a model optimal under
        # --opt-mode=optN. Measured both ways before this was written.
        self.search_exhausted = False

        # Opt-in solve deadline, as an absolute time.monotonic() timestamp (02_ASP/main.py
        # --solve-deadline). None, the default, keeps the blocking ctl.solve() below, so a run
        # that does not ask for a deadline searches exactly as it always has.
        self.deadline = deadline
        # Filled in only when a deadline is set. See _solve_until_deadline().
        self.stopped_at_deadline = False
        self.solve_started_at = None
        self.solve_ended_at = None
        self.solve_summary = None
        self.cost_priorities = None


    def solve(self):

        self.final_model = None
        self.best_cost = None
        self.models_reported = 0
        self.search_exhausted = False
        self.stopped_at_deadline = False
        self.solve_started_at = None
        self.solve_ended_at = None
        self.solve_summary = None
        self.cost_priorities = None

        start_time = time.time()
        self.total_time_start = start_time

        ctl = clingo.Control(self.solver_options)
        ctl.configuration.solver.seed = self.seed

        priorities = None
        if self.deadline is not None and self.report_solver_stats:
            priorities = _CostPriorities()
            ctl.register_observer(priorities)

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

                    if self.deadline is None:
                        solve_result = ctl.solve(on_model=self.on_model)
                    else:
                        solve_result = self._solve_until_deadline(ctl)
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

        if self.deadline is not None and self.report_solver_stats:
            # Read for every solve under a deadline, WITH OR WITHOUT a model: a core-guided search
            # stopped before its first model has no incumbent to report but can still have a
            # lower bound, and main.py prints it.
            self.cost_priorities = priorities.ordered()
            self.solve_summary = _solver_summary(ctl, self.models_reported)
            for key in ("SOLVER-COSTS", "SOLVER-LOWER-BOUND"):
                if key in self.solve_summary:
                    self.solve_summary[key] = _finite_or_none(self.solve_summary[key])
            self.solve_summary["SOLVER-COST-PRIORITIES"] = self.cost_priorities

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
            if self.deadline is None:
                self.final_model.set_solver_summary(_solver_summary(ctl, self.models_reported))
            else:
                self.final_model.set_solver_summary(self.solve_summary)

        return self.final_model

    def _solve_until_deadline(self, ctl):
        """Search until the search ends or self.deadline passes, whichever comes first.

        The benchmark caller enforces its time limit with SIGKILL, and a killed process never reads
        ctl.statistics, so its lower bound is lost. Here the search runs on clingo's own thread
        (async_=True) while this thread waits for it, and at the deadline it is cancelled from
        inside the process. handle.get() then returns normally and the statistics, lower bound
        included, can be read exactly as after a search that ended by itself.

        Threads. on_model runs on clingo's search thread in this mode and writes its result line
        through self.tmp_fd. Nothing else touches that descriptor until the finally block in
        solve(), and that block only runs after this `with` block has ended: handle.get() waits
        for the search thread, and leaving the block closes the handle. An exception raised inside
        on_model stops the search and is raised here as a RuntimeError carrying its message
        (measured on clingo 5.6.2), so it still ends the run instead of hanging it.

        What this cannot stop: translation, ctl.add() and grounding run before the search and are
        not interruptible. A deadline that passes during them is noticed only when the search
        starts, which then gets no time at all, and if they overrun the deadline by more than the
        caller's margin the external kill arrives first, exactly as without a deadline.
        """
        self.solve_started_at = time.monotonic()
        with ctl.solve(on_model=self.on_model, async_=True) as handle:
            if not handle.wait(max(0.0, self.deadline - self.solve_started_at)):
                handle.cancel()
            result = handle.get()
        self.solve_ended_at = time.monotonic()
        # From the result, not from wait() timing out: a search that ends by itself between the
        # wait and the cancel is exhausted, not stopped.
        self.stopped_at_deadline = bool(result.interrupted)
        return result

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
