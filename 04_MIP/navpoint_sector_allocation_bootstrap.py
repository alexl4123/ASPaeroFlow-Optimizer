"""Local access to the repository-wide navpoint-to-sector allocation.

Each solver folder is launched as its own top-level script, so only that folder sits on
sys.path and `common/` at the repository root is not importable by default. This shim puts the
root on sys.path and re-exports the allocation API, so callers can simply write

    from <this module> import build_assignment, load_schedule_for

and there is still exactly one implementation, in common/navpoint_sector_allocation.py.
"""
from pathlib import Path
import sys as _sys


def _bootstrap():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "common" / "navpoint_sector_allocation.py").is_file():
            if str(parent) not in _sys.path:
                _sys.path.insert(0, str(parent))
            return
    raise RuntimeError(
        "repository root (the directory holding common/navpoint_sector_allocation.py) not found "
        f"above {here}"
    )


_bootstrap()

from common.navpoint_sector_allocation import (  # noqa: E402
    SCHEDULE_COLUMNS,
    SCHEDULE_FILENAME,
    ScheduleError,
    build_assignment,
    change_points,
    describe,
    epoch_starts,
    is_time_varying,
    load_schedule,
    load_schedule_for,
    schedule_path_beside,
)
