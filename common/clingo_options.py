"""Clingo solver options, shared by every solver folder in this repository.

Every ASP solver folder here built its `clingo.Control` with a fixed argument list, so the search
configuration was whatever the two constructors happened to hard-code. A benchmark could not vary
it, and an ablation over solver strategies was not expressible.

The options are now selected by NAME, so a benchmark configuration can say `--solver-profile usc`
rather than carrying a list of raw clingo flags:

    default      (no flags)                                  branch-and-bound, clingo's default.
                                                             EXACTLY what this code did before
                                                             profiles existed, and the default.
    usc          --opt-strategy=usc,oll --opt-usc-shrink=min core-guided (unsat-core) optimisation
    domain       --heuristic=Domain                          honour the #heuristic directives in
                                                             the encoding
    usc-domain   both of the above                            the combination that the BSc project
                                                             measured as transforming runtime

`domain` matters because a `#heuristic` directive in an encoding is inert on its own: the grounder
emits it and clasp discards it unless the domain heuristic is switched on. Measured on
02_ASP/encoding.lp + 02_ASP/20260915_test.lp at a fixed conflict budget, adding or removing the
directives changes no search statistic at all without --heuristic=Domain.

`default` is the default everywhere, and it produces the empty flag list, so a run that passes no
new option constructs its Control exactly as before. The published LPNMR/ATMOS numbers and any
campaign in flight are therefore unaffected.

--solver-arg is the escape hatch for a flag that has no profile yet. It is repeatable and its
values are appended after the profile's flags, so it can also override them.

This module is imported from folders that each run as their own top-level script, so it lives at
the repository root and is reached through each folder's `clingo_options_bootstrap.py` shim, the
same arrangement as common/arrival_delay.py.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The option list each profile stands for. Order is the order clingo receives them.
SOLVER_PROFILES: Dict[str, Tuple[str, ...]] = {
    "default": (),
    "usc": ("--opt-strategy=usc,oll", "--opt-usc-shrink=min"),
    "domain": ("--heuristic=Domain",),
    "usc-domain": ("--heuristic=Domain", "--opt-strategy=usc,oll", "--opt-usc-shrink=min"),
}

#: No flags. Keeps every existing run byte-for-byte what it was.
DEFAULT_SOLVER_PROFILE = "default"

SOLVER_PROFILE_HELP = (
    "Named clingo search configuration. "
    "'default' = no flags (branch-and-bound; what this code has always done, and the default), "
    "'usc' = --opt-strategy=usc,oll --opt-usc-shrink=min (core-guided), "
    "'domain' = --heuristic=Domain (activates the #heuristic directives in the encoding, which "
    "are inert without it), "
    "'usc-domain' = both."
)

SOLVER_ARG_HELP = (
    "Extra raw clingo flag, e.g. --solver-arg=--opt-heuristic=sign. Repeatable. Appended after "
    "the profile's flags, so it can override them. Escape hatch for a flag with no profile."
)


def normalise_profile(profile: Optional[str]) -> str:
    """Validate and canonicalise a profile name."""
    if profile is None:
        return DEFAULT_SOLVER_PROFILE
    name = str(profile).strip().lower()
    if name not in SOLVER_PROFILES:
        raise ValueError(
            f"unknown solver profile {profile!r}; "
            f"expected one of {', '.join(SOLVER_PROFILES)}"
        )
    return name


def profile_options(profile: Optional[str] = None) -> List[str]:
    """The clingo flags a profile name stands for."""
    return list(SOLVER_PROFILES[normalise_profile(profile)])


def build_options(profile: Optional[str] = None,
                  extra: Optional[Iterable[str]] = None) -> List[str]:
    """The full clingo argument list for a profile plus any raw --solver-arg values.

    Returns [] for the default profile with no extras, which is what every call site used
    before profiles existed.
    """
    options = profile_options(profile)
    if extra:
        options.extend(str(a) for a in extra if str(a).strip())
    return options


def describe(profile: Optional[str] = None,
             extra: Optional[Iterable[str]] = None) -> str:
    """One-line, log-friendly rendering of a selection."""
    options = build_options(profile, extra)
    name = normalise_profile(profile)
    return f"{name}: {' '.join(options) if options else '(no flags)'}"


def add_cli_arguments(parser, default_profile: Optional[str] = None,
                      default_args: Optional[Sequence[str]] = None):
    """Register --solver-profile and --solver-arg on an argparse parser."""
    parser.add_argument(
        "--solver-profile",
        type=str,
        choices=sorted(SOLVER_PROFILES),
        default=normalise_profile(default_profile),
        help=SOLVER_PROFILE_HELP,
    )
    parser.add_argument(
        "--solver-arg",
        type=str,
        action="append",
        default=list(default_args) if default_args else None,
        metavar="FLAG",
        help=SOLVER_ARG_HELP,
    )
    return parser


def options_from_args(args) -> List[str]:
    """The clingo argument list implied by a parsed argparse namespace.

    Tolerates a namespace that carries neither option, so a caller that has not yet registered
    them keeps the old behaviour.
    """
    return build_options(getattr(args, "solver_profile", None),
                         getattr(args, "solver_arg", None))
