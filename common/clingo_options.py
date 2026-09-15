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
values are appended LAST -- after the thread count and after the profile's flags -- and override
either. Overriding has to be done by REPLACEMENT, not by ordering: clingo rejects a repeated
option outright ("multiple occurrences: 'parallel-mode'") rather than letting the last one win, so
build_options() drops an earlier flag when a later one sets the same option.

THREADS. clasp's search thread count is a separate axis from the profile, and it reaches clasp as
--parallel-mode=N. The default is 1, which is what clingo itself defaults to: an explicit
--parallel-mode=1 leaves all 60 configuration keys clingo exposes exactly as passing no flag does,
so requesting one thread is genuinely a no-op rather than a different single-threaded setup. One
thread is the default for COMPARABILITY -- every published LPNMR and ATMOS number was produced by
a single-threaded clasp -- and not because one thread is better. It is not: clasp's parallel mode
is a portfolio, and 4 threads have been measured to prove an optimum in 0.18 s on an instance that
one thread did not close in 25 s.

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

#: clasp search threads. 1 is clingo's own default, so this reproduces every run to date.
DEFAULT_SOLVER_THREADS = 1

SOLVER_THREADS_HELP = (
    "Number of parallel clasp search threads (clingo --parallel-mode). Default 1, which is what "
    "every published result was produced with; clasp runs a PORTFOLIO in parallel mode, so a "
    "higher count changes the search, not just its speed."
)

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
    "the thread count and the profile's flags, and REPLACES either where they set the same "
    "clingo option. Escape hatch for a flag with no profile."
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


#: Short clingo flags that set the same option as a long flag this module generates itself.
#: Only -t matters in practice, because --parallel-mode is the one flag built automatically.
_SHORT_FORMS = {"-t": "parallel-mode"}


def _option_name(flag: str) -> Optional[str]:
    """The clingo option a flag sets: '--opt-strategy=usc,oll' -> 'opt-strategy'.

    None when it cannot be told (a bare value such as the 4 in `-t 4`), in which case the flag is
    kept as-is.
    """
    text = str(flag)
    if text.startswith("--"):
        return text[2:].split("=", 1)[0]
    for short, long_name in _SHORT_FORMS.items():
        if text == short or text.startswith(short):
            return long_name
    return None


def _last_occurrence_wins(flags: Iterable[str]) -> List[str]:
    """Drop an earlier flag when a later one sets the same clingo option.

    clingo raises on a repeated option instead of taking the last value, so this is what makes
    --solver-arg an override rather than a crash.
    """
    kept: List[str] = []
    seen = set()
    for flag in reversed(list(flags)):
        name = _option_name(flag)
        if name is not None:
            if name in seen:
                continue
            seen.add(name)
        kept.append(flag)
    kept.reverse()
    return kept


def thread_options(threads: Optional[int] = None) -> List[str]:
    """The clingo flags for a search thread count.

    None means "say nothing", for a call site that does not manage threads. A count of 1 still
    emits --parallel-mode=1 explicitly: it is indistinguishable from passing nothing (verified
    across every configuration key clingo exposes), and emitting it keeps the argument list an
    honest record of what was asked for.
    """
    if threads is None:
        return []
    count = int(threads)
    if count < 1:
        raise ValueError(f"solver thread count must be at least 1, got {threads!r}")
    return [f"--parallel-mode={count}"]


def build_options(profile: Optional[str] = None,
                  extra: Optional[Iterable[str]] = None,
                  threads: Optional[int] = None) -> List[str]:
    """The full clingo argument list: thread count, then profile flags, then raw --solver-arg.

    Returns [] for the default profile with no extras and no thread count, which is what every
    call site used before profiles existed. `extra` comes last so a raw flag wins over both.
    """
    options = thread_options(threads)
    options.extend(profile_options(profile))
    if extra:
        options.extend(str(a) for a in extra if str(a).strip())
    return _last_occurrence_wins(options)


def describe(profile: Optional[str] = None,
             extra: Optional[Iterable[str]] = None,
             threads: Optional[int] = None) -> str:
    """One-line, log-friendly rendering of a selection."""
    options = build_options(profile, extra, threads)
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


def options_from_args(args, threads: Optional[int] = None) -> List[str]:
    """The clingo argument list implied by a parsed argparse namespace.

    Tolerates a namespace that carries neither option, so a caller that has not yet registered
    them keeps the old behaviour.

    `threads` is passed explicitly rather than read off the namespace, because --number-threads
    does NOT mean clasp threads in every folder that has one: in 01_ASPaeroFlow it is the outer
    Python loop's own processor accounting, and in 04_MIP it is Gurobi's model.Params.Threads.
    Only a call site that knows its --number-threads means clasp threads should pass it here.
    """
    return build_options(getattr(args, "solver_profile", None),
                         getattr(args, "solver_arg", None),
                         threads)
