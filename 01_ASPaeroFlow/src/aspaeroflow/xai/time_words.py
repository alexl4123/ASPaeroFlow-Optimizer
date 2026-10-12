"""Times and durations in the words of the interface (K24, the same rules as the frontend's types/time-scale.ts).

A time of the timeline is "time step 37 (09:15)" ("time step 101 (+1d 01:15)" past midnight); "step 6" stays the
optimizer step. A duration (delay, shift, sum of delays) is in minutes and hours: "15 min", "1 h", "1 h 15 min",
signed where it is a difference ("+45 min", "-15 min"). Step 0 is 00:00; a time step lasts 60 / granularity minutes
(run.json "timestep_granularity", 1 when absent). Texts are ASCII only (they travel inside clingo string atoms).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def granularity(run: Optional[Dict[str, Any]]) -> int:
    try:
        g = int((run or {}).get("timestep_granularity") or 1)
    except (TypeError, ValueError):
        g = 1
    return max(1, g)


def minutes(steps: int, g: int) -> int:
    """Whole minutes of a number of time steps (integer arithmetic, as the frontend)."""
    return int(steps) * 60 // g


def clock_text(step: int, g: int) -> str:
    """'09:15', '+1d 01:15'."""
    day, rest = divmod(int(step), 24 * g)
    m = minutes(rest, g)
    text = f"{m // 60:02d}:{m % 60:02d}"
    return f"+{day}d {text}" if day else text


def time_text(step: int, g: int) -> str:
    """'time step 37 (09:15)'."""
    return f"time step {int(step)} ({clock_text(step, g)})"


def minutes_text(m: int) -> str:
    """'0 min', '15 min', '1 h', '1 h 15 min' (no sign; the absolute value)."""
    m = abs(int(m))
    h, r = divmod(m, 60)
    if h and r:
        return f"{h} h {r} min"
    return f"{h} h" if h else f"{r} min"


def duration_text(steps: int, g: int) -> str:
    """A duration of time steps in minutes and hours; negative values get a minus ('-15 min')."""
    m = minutes(steps, g)
    return ("-" if m < 0 else "") + minutes_text(m)


def shift_text(steps: int, g: int) -> str:
    """A signed difference: '+45 min', '-15 min', '0 min'."""
    m = minutes(steps, g)
    return ("+" if m > 0 else "-" if m < 0 else "") + minutes_text(m)


def time_steps(n: int) -> str:
    """A count of time steps (flight durations count occupied time steps): '1 time step', '3 time steps'."""
    return "1 time step" if int(n) == 1 else f"{int(n)} time steps"
