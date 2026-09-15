"""Local access to the repository-wide clingo solver options.

Each solver folder is launched as its own top-level script, so only that folder sits on
sys.path and `common/` at the repository root is not importable by default. This shim puts the
root on sys.path and re-exports the solver-option API, so callers can simply write

    from <this module> import add_cli_arguments, options_from_args

and there is still exactly one implementation, in common/clingo_options.py.
"""
from pathlib import Path
import sys as _sys


def _bootstrap():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "common" / "clingo_options.py").is_file():
            if str(parent) not in _sys.path:
                _sys.path.insert(0, str(parent))
            return
    raise RuntimeError(
        "repository root (the directory holding common/clingo_options.py) not found above "
        f"{here}"
    )


_bootstrap()

from common.clingo_options import (  # noqa: E402
    DEFAULT_SOLVER_PROFILE,
    SOLVER_ARG_HELP,
    SOLVER_PROFILE_HELP,
    SOLVER_PROFILES,
    add_cli_arguments,
    build_options,
    describe,
    normalise_profile,
    options_from_args,
    profile_options,
)
