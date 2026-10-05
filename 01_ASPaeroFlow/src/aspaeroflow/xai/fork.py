"""Fork-and-continue: how much does one step's choice matter for the final plan?

At step k the recorded answer is replaced by the best answer of the same sub-problem that differs
from it (for a tied step an equally good one, otherwise the runner-up), and the run continues to
the end. Runs are deterministic, so steps 1..k-1 are the same as in the base run. Comparing the
final objectives over all k shows how path-dependent the heuristic is: the noise band any
sensitivity number must be read against, and the extent to which a step-local explanation speaks
for the whole run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..solver import Solver
from .session import _load_cli, OPTIMIZER_DIR

OBJECTIVES = ("OVERLOAD", "ARRIVAL-DELAY", "SECTOR-NUMBER", "REROUTE", "RECONFIG")


def _final(app) -> Dict[str, Any]:
    trace = app._xai_trace
    last = trace.last["objectives"] if trace is not None and trace.last else {}
    return {k: last.get(k) for k in OBJECTIVES} | {"iterations": last.get("ITERATION")}


def run(data_dir: Path, trace_dir: Path, options: Optional[Dict[str, Any]] = None,
        fork_at: Optional[int] = None) -> Dict[str, Any]:
    """One run; with fork_at = k the k-th step takes the best answer other than its own."""
    cli = _load_cli()
    argv = [f"--data-dir={data_dir}", f"--encoding-path={OPTIMIZER_DIR / 'encoding.lp'}",
            "--save-results=false", f"--xai-trace-dir={trace_dir}"]
    argv += [f"--{k.replace('_', '-')}={v}" for k, v in (options or {}).items()]
    args = cli.parse_cli(argv)
    app = cli.make_app(args, xai_trace_dir=trace_dir)
    info: Dict[str, Any] = {"fork_at": fork_at, "forked": False}

    def fork(app_, dto):
        solutions = dto["solutions"]
        if fork_at is None or int(dto["iteration"]) + 1 != fork_at:
            return solutions
        out = []
        for model, restore, instance in solutions:
            chosen = model.get_chosen_paths()
            config = int(str(model.get_sector_config().arguments[0]))
            body = ", ".join([f"chosen_config({config})"] + [f"chosen_path({f},{p})" for f, p in sorted(chosen.items())])
            try:
                alternative = Solver(app_.encoding, instance + f"\n:- {body}.\n", seed=app_._seed,
                                     solver_options=app_._solver_options).solve()
            except AttributeError:     # no other answer: Solver.solve() has no model to time
                alternative = None
            if alternative is None:
                out.append((model, restore, instance))
                continue
            info.update(forked=True, recorded_cost=model.cost, alternative_cost=alternative.cost,
                        tie=list(model.cost) == list(alternative.cost))
            out.append((alternative, restore, instance))
        return out

    app._xai_fork = fork
    app.run()
    return info | {"final": _final(app)}


def experiment(data_dir: Path, work_dir: Path, options: Optional[Dict[str, Any]] = None,
               steps: Optional[List[int]] = None) -> Dict[str, Any]:
    work_dir = Path(work_dir)
    base = run(data_dir, work_dir / "base", options)
    n = base["final"]["iterations"] or 0
    forks = [run(data_dir, work_dir / f"fork_{k:03d}", options, fork_at=k) for k in (steps or range(1, n + 1))]
    return {"base": base, "forks": forks}
