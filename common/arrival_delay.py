"""Arrival-delay metric, shared by every solver folder in this repository.

The publications define arrival delay as the SIGNED difference

    delta = t_actarr - t_exparr

Historically every implementation here computed max(0, delta) instead, discarding early
arrivals. All three readings are now selectable via --arrival-delay-metric, so a run is explicit
about which definition produced its numbers.

    signed    delta                 the published definition; earliness offsets lateness. DEFAULT.
    floored   max(0, delta)         lateness only; what this code did before the metric existed
    absolute  |delta|               penalises earliness and lateness equally

The default is `signed`, so that the code agrees with the publications out of the box. This was
a deliberate change from the earlier `floored` default; it is safe on the shipped instances
because the three metrics coincide there -- the solvers only ever push a flight later, never
earlier, so no previously published number moves. Pass `--arrival-delay-metric floored` to get
the pre-change behaviour explicitly.

The choice starts to matter as soon as a solver can produce an early arrival -- a reroute onto a
shorter path, for instance. From that point on `signed` lets earliness offset lateness in the
aggregate, exactly as the papers define it.

This module is imported from four folders that each run as their own top-level script, so it
lives at the repository root and is reached through each folder's
`arrival_delay_bootstrap.py` shim.
"""
from __future__ import annotations

import sys
from pathlib import Path

SIGNED = "signed"
FLOORED = "floored"
ABSOLUTE = "absolute"

ARRIVAL_DELAY_METRICS = (SIGNED, FLOORED, ABSOLUTE)

#: The definition used in the publications. `floored` is what the code did before the metric
#: became selectable and remains available explicitly.
DEFAULT_ARRIVAL_DELAY_METRIC = SIGNED

CLI_HELP = (
    "How arrival delay delta = t_actarr - t_exparr is scored: "
    "'signed' = delta (the definition used in the publications; early arrivals count "
    "negatively), 'floored' = max(0, delta) (lateness only), "
    "'absolute' = |delta|. Default: signed, matching the publications."
)


def normalise(metric) -> str:
    """Validate and canonicalise a metric name."""
    if metric is None:
        return DEFAULT_ARRIVAL_DELAY_METRIC
    m = str(metric).strip().lower()
    if m not in ARRIVAL_DELAY_METRICS:
        raise ValueError(
            f"unknown arrival-delay metric {metric!r}; "
            f"expected one of {', '.join(ARRIVAL_DELAY_METRICS)}"
        )
    return m


def apply(delta, metric=DEFAULT_ARRIVAL_DELAY_METRIC):
    """Score a signed delay difference under `metric`.

    Works for a plain int and for a numpy array; returns the same kind it was given.
    """
    m = normalise(metric)
    if m == SIGNED:
        return delta
    if m == FLOORED:
        try:                      # numpy array
            import numpy as np
            return np.maximum(0, delta)
        except ImportError:
            return max(0, delta)
    # ABSOLUTE
    try:
        import numpy as np
        return np.abs(delta)
    except ImportError:
        return abs(delta)


def delay_matrix(t_init, t_final, metric=DEFAULT_ARRIVAL_DELAY_METRIC):
    """Per-flight delay from initial and final arrival timesteps, as the reporting path needs it.

    `t_init` / `t_final` are the last valid position of each flight in the initial and final
    schedule; a flight absent from the initial schedule (t_init < 0) contributes 0 whatever the
    metric, because it has no planned arrival to be measured against.
    """
    import numpy as np
    return np.where(t_init >= 0, apply(t_final - t_init, metric), 0)


def add_cli_argument(parser, default=None):
    """Register --arrival-delay-metric on an argparse parser."""
    parser.add_argument(
        "--arrival-delay-metric",
        type=str,
        choices=list(ARRIVAL_DELAY_METRICS),
        default=normalise(default),
        help=CLI_HELP,
    )
    return parser


def asp_metric_fact(metric=DEFAULT_ARRIVAL_DELAY_METRIC) -> str:
    """The fact that tells an ASP encoding which metric to score.

    Both encodings derive `arrival_delay_scoring/1` from `arrival_delay_metric/1` and fall back
    to the default when no such fact is present, so appending this is all that is required.
    """
    return f"arrival_delay_metric({normalise(metric)}).\n"
