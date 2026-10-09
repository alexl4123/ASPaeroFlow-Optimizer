"""One epsilon-constraint step of an exact (delay, sectors) Pareto front.

The Pareto-front campaign (06_benchmark_start_script/build_pareto_manifest.py) solves the program
of one (instance, variant) many times, once per bound K on the total arrival delay, and minimises
the number of active sectors under that bound:

    sectors-first order  the two weak constraints swap priorities, so sector_number sits at @9 and
                         arrival_delay at @8. Overload stays at @10, above both.
    delay bound K        a hard constraint  :- #sum{DIFF,ID: arrival_delay(ID,DIFF)} > K.
    fixed horizon        02_ASP/main.py does not extend --max-time when overload persists (see the
                         re-solve loop there). A bound below the least delay the fixed horizon
                         allows forces overload up, and the loop would then re-solve until its
                         theorem bound instead of reporting that overload.

All three are OPT-IN (main.py --objective-order / --delay-bound / --fixed-horizon). Without them
the program is byte-for-byte the one main.py has always built.

The swap refuses to guess: it requires that exactly one active weak constraint sits at @9, that it
is the arrival_delay one, that exactly one sits at @8, and that it is the sector_number one. An
encoding that has changed shape fails loudly instead of being solved in a half-swapped order.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

DELAY_FIRST = "delay-first"
SECTORS_FIRST = "sectors-first"
OBJECTIVE_ORDERS = (DELAY_FIRST, SECTORS_FIRST)

# The two weak constraints of 02_ASP/encoding.lp whose priorities a sectors-first step swaps:
#     :~ arrival_delay(ID,DIFF). [DIFF@9,ID]
#     :~ sector_number(T,NUM). [NUM@8,T]
_DELAY_WEAK = re.compile(
    r"^(?P<head>[ \t]*:~[ \t]*arrival_delay\([ \t]*ID[ \t]*,[ \t]*DIFF[ \t]*\)[ \t]*\.[ \t]*"
    r"\[[ \t]*DIFF[ \t]*@[ \t]*)9(?P<tail>[ \t]*,[ \t]*ID[ \t]*\])", re.M)
_SECTOR_WEAK = re.compile(
    r"^(?P<head>[ \t]*:~[ \t]*sector_number\([ \t]*T[ \t]*,[ \t]*NUM[ \t]*\)[ \t]*\.[ \t]*"
    r"\[[ \t]*NUM[ \t]*@[ \t]*)8(?P<tail>[ \t]*,[ \t]*T[ \t]*\])", re.M)
_PRIORITY = re.compile(r"@[ \t]*(-?\d+)")


def _weak_priorities(encoding: str) -> Dict[int, List[str]]:
    """Active weak constraints by the priority they name. A "% ..." comment line never starts with
    ":~", so commented-out weak constraints are not counted."""
    by_priority: Dict[int, List[str]] = {}
    for line in encoding.splitlines():
        text = line.strip()
        if not text.startswith(":~"):
            continue
        match = _PRIORITY.search(text)
        if match:
            by_priority.setdefault(int(match.group(1)), []).append(text)
    return by_priority


def swap_to_sectors_first(encoding: str) -> str:
    """The encoding with sector_number at @9 and arrival_delay at @8, or ValueError."""
    delay = _DELAY_WEAK.findall(encoding)
    sector = _SECTOR_WEAK.findall(encoding)
    priorities = _weak_priorities(encoding)
    problems = []
    if len(delay) != 1:
        problems.append(f"found {len(delay)} arrival_delay weak constraints at @9, expected 1")
    if len(sector) != 1:
        problems.append(f"found {len(sector)} sector_number weak constraints at @8, expected 1")
    for prio in (9, 8):
        if len(priorities.get(prio, [])) != 1:
            problems.append(f"{len(priorities.get(prio, []))} active weak constraints at @{prio}, "
                            f"expected exactly 1: {priorities.get(prio, [])}")
    if problems:
        raise ValueError("--objective-order sectors-first cannot swap the two weak constraints "
                         "of this encoding: " + "; ".join(problems))
    swapped = _DELAY_WEAK.sub(lambda m: m.group("head") + "8" + m.group("tail"), encoding)
    swapped = _SECTOR_WEAK.sub(lambda m: m.group("head") + "9" + m.group("tail"), swapped)
    after = _weak_priorities(swapped)
    if [line for line in after.get(9, []) if "sector_number" not in line] or \
            [line for line in after.get(8, []) if "arrival_delay" not in line]:
        raise ValueError(f"swap produced an unexpected order: @9 {after.get(9)}, @8 {after.get(8)}")
    return swapped


def delay_bound_constraint(bound: int) -> str:
    """The hard constraint total arrival delay <= bound, as the encoding's weak constraint sums it.

    Same tuple (DIFF,ID) as `:~ arrival_delay(ID,DIFF). [DIFF@..,ID]`, so each flight counts once
    and the bound applies to exactly the quantity the objective level minimises, under whichever
    arrival-delay metric the program scores (a signed delay may be negative, and so may K).
    """
    return ("\n% --delay-bound (02_ASP/pareto_step.py): total arrival delay at most K.\n"
            f":- #sum {{ DIFF,ID : arrival_delay(ID,DIFF) }} > {int(bound)}.\n")


def apply(encoding: str, objective_order: Optional[str], delay_bound: Optional[int]) -> str:
    """The encoding one step solves. None/None returns the encoding unchanged."""
    if objective_order not in (None,) + OBJECTIVE_ORDERS:
        raise ValueError(f"unknown objective order {objective_order!r}")
    if objective_order == SECTORS_FIRST:
        encoding = swap_to_sectors_first(encoding)
    if delay_bound is not None:
        encoding = encoding + delay_bound_constraint(delay_bound)
    return encoding


def active(args) -> bool:
    """Whether any Pareto-step option was given on this command line."""
    return (getattr(args, "objective_order", None) is not None
            or getattr(args, "delay_bound", None) is not None
            or bool(getattr(args, "fixed_horizon", False)))


def result_fields(args) -> Dict[str, object]:
    """The keys a Pareto step adds to its result line: which of the options were in force."""
    return {
        "PARETO-OBJECTIVE-ORDER": args.objective_order or DELAY_FIRST,
        "PARETO-DELAY-BOUND": args.delay_bound,
        "PARETO-FIXED-HORIZON": bool(args.fixed_horizon),
    }


def export_header(args) -> str:
    """Extra lines for the --export-program-lp header, empty unless a Pareto option is active."""
    if not active(args):
        return ""
    fields = result_fields(args)
    return (f"% objective order:       {fields['PARETO-OBJECTIVE-ORDER']}\n"
            f"% delay bound:           {fields['PARETO-DELAY-BOUND']}\n"
            f"% fixed horizon:         {fields['PARETO-FIXED-HORIZON']}\n")
