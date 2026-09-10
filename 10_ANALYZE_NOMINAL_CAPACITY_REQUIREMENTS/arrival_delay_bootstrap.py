"""Local access to the repository-wide arrival-delay metric.

Each solver folder is launched as its own top-level script, so only that folder sits on
sys.path and `common/` at the repository root is not importable by default. This shim puts the
root on sys.path and re-exports the metric API, so callers can simply write

    from <this module> import ARRIVAL_DELAY_METRICS, apply, delay_matrix

and there is still exactly one implementation, in common/arrival_delay.py.
"""
from pathlib import Path
import sys as _sys


def _bootstrap():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "common" / "arrival_delay.py").is_file():
            if str(parent) not in _sys.path:
                _sys.path.insert(0, str(parent))
            return
    raise RuntimeError(
        "repository root (the directory holding common/arrival_delay.py) not found above "
        f"{here}"
    )


_bootstrap()

from common.arrival_delay import (  # noqa: E402
    ABSOLUTE,
    ARRIVAL_DELAY_METRICS,
    CLI_HELP,
    DEFAULT_ARRIVAL_DELAY_METRIC,
    FLOORED,
    SIGNED,
    add_cli_argument,
    apply,
    asp_metric_fact,
    delay_matrix,
    normalise,
)

