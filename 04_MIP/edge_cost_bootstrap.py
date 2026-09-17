"""Local access to the repository-wide edge cost in timesteps.

Each solver folder is launched as its own top-level script, so only that folder sits on
sys.path and `common/` at the repository root is not importable by default. This shim puts the
root on sys.path and re-exports the edge-cost API, so callers can simply write

    from <this module> import edge_duration_timesteps, load_graph_edges

and there is still exactly one implementation, in common/edge_cost.py.
"""
from pathlib import Path
import sys as _sys


def _bootstrap():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "common" / "edge_cost.py").is_file():
            if str(parent) not in _sys.path:
                _sys.path.insert(0, str(parent))
            return
    raise RuntimeError(
        "repository root (the directory holding common/edge_cost.py) not found above "
        f"{here}"
    )


_bootstrap()

from common.edge_cost import (  # noqa: E402
    KNOTS_TO_METRES_PER_SECOND,
    edge_duration_timesteps,
    load_graph_edges,
)
