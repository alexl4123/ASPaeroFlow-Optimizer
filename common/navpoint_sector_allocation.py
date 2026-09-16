"""The navpoint-to-sector allocation, shared by every solver folder in this repository.

Every solver works from the same dense ``(|N| x |T|)`` integer array, in which entry
``[nav, t]`` is the sector navpoint ``nav`` belongs to at timestep ``t``. That array has always
been the interface between the input files and the rest of each solver; what it could not
express until now is an allocation that CHANGES over time, because it was filled by broadcasting
a single static assignment across every column.

Two input files feed it:

``navaid_sector_assignment.csv``   ``Navaid_ID,Sector_ID``
    The static allocation. Required. Read exactly as before.

``navaid_sector_schedule.csv``     ``Navaid_ID,Sector_ID,From_Time``
    The time-varying allocation. OPTIONAL. Sparse change-points: a row says "from timestep
    ``From_Time`` onward, ``Navaid_ID`` sits in ``Sector_ID``, until the next row for the same
    navpoint". Rows are applied in ``From_Time`` order on top of the static broadcast.

When the schedule file is absent the array is built exactly as it was before this module
existed, so every previously published instance reproduces its old numbers. A schedule whose
rows all carry ``From_Time = 0`` is likewise indistinguishable from the static case: it
overwrites column 0 onward, which is the whole array.

Three solver folders call ``build_assignment`` from an instance method and the MIP calls it from
a classmethod, so the logic lives here once and each folder reaches it through its own
``navpoint_sector_allocation_bootstrap.py`` shim, the same way ``common/arrival_delay.py`` is
reached.

ASP fact emission
-----------------
``02_ASP/encoding.lp`` takes ``navpoint_sector(NAV,SEC,0)`` facts and broadcasts them across
every timestep. That broadcast is right while the allocation is constant and wrong as soon as it
is not: the T=0 fact and the changed sector would both hold at the same timestep and violate the
one-sector-per-navpoint constraint. ``asp_change_point_facts`` therefore emits
``navpoint_sector_from(NAV,SEC,FROM_T)`` for the navpoints that actually move, and the encoding
withholds its broadcast from exactly those navpoints. On a constant allocation the function
returns nothing at all, so the grounded program is byte-identical to what it was before.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The optional time-varying allocation, looked for beside navaid_sector_assignment.csv.
SCHEDULE_FILENAME = "navaid_sector_schedule.csv"

#: The header the schedule file must carry, in this order.
SCHEDULE_COLUMNS = ("Navaid_ID", "Sector_ID", "From_Time")

#: The ASP predicate carrying a change-point into the encoding.
ASP_CHANGE_POINT_PREDICATE = "navpoint_sector_from"

CLI_HELP = (
    "Optional time-varying navpoint-to-sector allocation, as "
    f"'{','.join(SCHEDULE_COLUMNS)}' change-point rows: from From_Time onward that navpoint "
    "sits in that sector, until its next row. When the file is absent the static "
    "navaid_sector_assignment.csv is used for every timestep, which is the historical "
    f"behaviour. Defaults to {SCHEDULE_FILENAME} beside the static allocation, if that file "
    "exists."
)


class ScheduleError(ValueError):
    """A schedule file that cannot be trusted to mean what it appears to say."""


# ---------------------------------------------------------------------------
# Locating the file
# ---------------------------------------------------------------------------

def schedule_path_beside(navaid_sector_path) -> Optional[Path]:
    """Where a time-varying allocation for this instance would live, if it exists.

    Returns None when there is no such file, which is the ordinary case for every instance
    published so far.
    """
    if navaid_sector_path is None:
        return None
    candidate = Path(navaid_sector_path).with_name(SCHEDULE_FILENAME)
    return candidate if candidate.is_file() else None


def resolve_schedule_path(navaid_sector_path, explicit=None) -> Optional[Path]:
    """Pick the schedule file to read.

    An explicitly named file must exist: naming one that does not is a mistake worth reporting,
    not a silent fall back to the static allocation. A file merely found beside the static
    allocation is optional.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ScheduleError(
                f"navpoint-sector schedule not found: {path}\n"
                "Leave the option unset to use the static navaid_sector_assignment.csv for "
                "every timestep."
            )
        return path
    return schedule_path_beside(navaid_sector_path)


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------

def _fail(path, line_no, message) -> None:
    where = f"{path}:{line_no}" if line_no is not None else str(path)
    raise ScheduleError(
        f"{where}: {message}\n"
        f"Expected rows of '{','.join(SCHEDULE_COLUMNS)}' with integer fields, under that header."
    )


def parse_schedule_rows(rows: Iterable[Sequence[str]], path="<rows>") -> List[Tuple[int, int, int]]:
    """Validate raw CSV rows and return them as (navaid, sector, from_time) triples.

    Sorted by (navaid, from_time), which is the order in which they are applied.
    """
    rows = list(rows)
    if not rows:
        _fail(path, None, "file is empty; it must carry a header row")

    header = [field.strip() for field in rows[0]]
    if [h.lower() for h in header] != [c.lower() for c in SCHEDULE_COLUMNS]:
        _fail(path, 1, f"header is {header}, which is not {list(SCHEDULE_COLUMNS)}")

    parsed: List[Tuple[int, int, int]] = []
    seen = {}
    for offset, row in enumerate(rows[1:]):
        line_no = offset + 2
        fields = [field.strip() for field in row]
        while fields and fields[-1] == "":
            fields.pop()
        if not fields:
            continue                                   # a blank line carries no change-point
        if len(fields) != len(SCHEDULE_COLUMNS):
            _fail(path, line_no, f"has {len(fields)} field(s), not {len(SCHEDULE_COLUMNS)}")

        try:
            navaid, sector, from_time = (int(field) for field in fields)
        except ValueError:
            _fail(path, line_no, f"has a non-integer field in {fields}")

        if navaid < 0:
            _fail(path, line_no, f"Navaid_ID is negative ({navaid})")
        if sector < 0:
            _fail(path, line_no, f"Sector_ID is negative ({sector})")
        if from_time < 0:
            _fail(path, line_no, f"From_Time is negative ({from_time})")

        previous = seen.get((navaid, from_time))
        if previous is not None:
            _fail(
                path, line_no,
                f"navpoint {navaid} already has a change-point at From_Time {from_time} "
                f"(line {previous[0]}, sector {previous[1]}); one row per navpoint per timestep",
            )
        seen[(navaid, from_time)] = (line_no, sector)
        parsed.append((navaid, sector, from_time))

    parsed.sort(key=lambda entry: (entry[0], entry[2]))
    return parsed


def load_schedule(path) -> List[Tuple[int, int, int]]:
    """Read a schedule file into (navaid, sector, from_time) triples."""
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise ScheduleError(f"could not read navpoint-sector schedule {path}: {exc}") from exc
    return parse_schedule_rows(rows, path=path)


def load_schedule_for(navaid_sector_path, explicit=None) -> Optional[List[Tuple[int, int, int]]]:
    """The schedule for this instance, or None when the instance has no schedule file."""
    path = resolve_schedule_path(navaid_sector_path, explicit)
    if path is None:
        return None
    return load_schedule(path)


# ---------------------------------------------------------------------------
# Building the dense (|N| x |T|) array
# ---------------------------------------------------------------------------

def build_assignment(flights: np.ndarray,
                     airplane_flight: np.ndarray,
                     navaid_sector: np.ndarray,
                     max_time: int,
                     time_granularity: int,
                     *,
                     schedule=None,
                     fill_value: int = -1,
                     compress: bool = False) -> np.ndarray:
    """The navpoint-to-sector allocation for every timestep, as an ``(|N| x |T|)`` array.

    With ``schedule=None`` this is the historical ``create_initial_navpoint_sector_assignment``,
    unchanged: the static allocation broadcast across every column, and a self-sector row for
    any navpoint the static file does not mention.

    With a schedule the change-points are applied on top of that broadcast, in ``From_Time``
    order, each one holding until the navpoint's next change-point.

    Assumes IDs are non-negative ints (reasonably dense).
    """

    # --- ensure integer views without copies where possible
    flights = flights.astype(np.int64, copy=False)
    airplane_flight = airplane_flight.astype(np.int64, copy=False)

    # --- build flight -> airplane mapping (array is fastest if IDs are dense)
    fid_map_max = int(max(flights[:, 0].max(), airplane_flight[:, 1].max()))
    flight_to_airplane = np.full(fid_map_max + 1, -1, dtype=np.int64)
    flight_to_airplane[airplane_flight[:, 1]] = airplane_flight[:, 0]

    # --- sort by flight, then time (stable contiguous blocks per flight)
    order = np.lexsort((flights[:, 2], flights[:, 0]))
    f_sorted = flights[order]

    t = f_sorted[:, 2]

    # --- output matrix shape (airplane_id rows, time columns)
    max_time_dim = int(max(t.max() + 1, (max_time + 1) * time_granularity))

    if max_time_dim % time_granularity != 0:
        remainder = max_time_dim % time_granularity
        max_time_dim += time_granularity - remainder

        if max_time_dim % time_granularity != 0:
            print("[ERROR] - Should never occur - failure in maths")
            raise Exception("[ERROR IN COMPUTATION]")

    largest_navaid = navaid_sector[navaid_sector.shape[0] - 1, 0]

    output = np.ones((largest_navaid + 1, max_time_dim), dtype=int) * (-1)

    for index in range(navaid_sector.shape[0]):
        output_index = navaid_sector[index, 0]
        output[output_index, :] = navaid_sector[index, 1]

    for index in range(output.shape[0]):

        if output[index, 0] == -1:
            output[index, :] = index

    if schedule:
        _apply_schedule(output, schedule)

    return output


def _apply_schedule(output: np.ndarray, schedule) -> None:
    """Overwrite `output` in place with the change-points, then check what they imply.

    The checks run only over the entries a schedule actually moved, so an instance without one
    is never subjected to them and cannot start failing because of this function.
    """
    n_navpoints, n_times = output.shape
    before = output[:, 0].copy()

    for navaid, sector, from_time in sorted(schedule, key=lambda entry: (entry[0], entry[2])):
        if navaid >= n_navpoints:
            raise ScheduleError(
                f"navpoint-sector schedule names navpoint {navaid}, but the static allocation "
                f"only reaches navpoint {n_navpoints - 1}"
            )
        if sector >= n_navpoints:
            raise ScheduleError(
                f"navpoint-sector schedule puts navpoint {navaid} into sector {sector}, which "
                f"is not a navpoint of this instance (the largest is {n_navpoints - 1})"
            )
        if from_time >= n_times:
            # Beyond this instance's horizon, so it can never take effect. Harmless, and saying
            # so is better than dropping it without a word.
            # stderr, because stdout carries the JSON lines the benchmark driver parses.
            print(
                f"[navpoint-sector schedule] navpoint {navaid} -> sector {sector} at timestep "
                f"{from_time} is past the end of the horizon ({n_times} timesteps) and has no "
                "effect",
                file=sys.stderr,
            )
            continue
        output[navaid, from_time:] = sector

    # A navpoint may only sit in a sector that is open at that timestep, meaning the sector's own
    # navpoint is assigned to itself. The ASP encoding states this as an integrity constraint
    # (encoding.lp, `not navpoint_sector(SEC,SEC,T)`), so breaking it turns into an unexplained
    # UNSAT; the MIP and ASPaeroFlow would instead read capacities off a sector nothing keeps
    # open. Catch it here, where the file can be named.
    #
    # Whole columns are checked, not just the moved entries: moving a sector's own navpoint away
    # strands the members that did NOT move. Only the columns the schedule changed are checked,
    # which leaves an instance without a schedule untouched.
    changed_columns = np.flatnonzero((output != before[:, None]).any(axis=0))
    for timestep in changed_columns:
        column = output[:, timestep]
        stranded = np.flatnonzero(column[column] != column)
        if stranded.size:
            navaid = int(stranded[0])
            sector = int(column[navaid])
            raise ScheduleError(
                f"navpoint-sector schedule leaves navpoint {navaid} in sector {sector} at "
                f"timestep {timestep}, but sector {sector} is not open then: navpoint {sector} "
                f"is itself assigned to sector {int(column[sector])}. A sector is open only while "
                f"its own navpoint is assigned to it ({stranded.size} navpoint(s) affected at "
                "this timestep)."
            )


def t0_navaid_sector(navaid_sector: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """The static allocation as of timestep 0, read back off the dense array.

    Row order and shape match the input CSV array, so every consumer of the static allocation
    sees the structure it already expects. Without a schedule the result equals the input, so
    routing the static allocation through this function changes nothing.
    """
    out = navaid_sector.copy()
    out[:, 1] = assignment[navaid_sector[:, 0], 0]
    return out


# ---------------------------------------------------------------------------
# Handing the allocation to ASP
# ---------------------------------------------------------------------------

def change_points(assignment: np.ndarray):
    """Per navpoint, the timesteps at which its sector changes.

    Yields ``(navaid, [(from_time, sector), ...])`` for the navpoints that move, starting with
    the ``from_time = 0`` entry so the list describes the whole horizon. Navpoints whose sector
    is constant are skipped, so a static instance yields nothing.
    """
    for navaid in range(assignment.shape[0]):
        row = assignment[navaid]
        breaks = np.flatnonzero(row[1:] != row[:-1]) + 1
        if breaks.size == 0:
            continue
        entries = [(0, int(row[0]))]
        entries += [(int(t), int(row[t])) for t in breaks]
        yield navaid, entries


def asp_change_point_facts(assignment: np.ndarray) -> List[str]:
    """``navpoint_sector_from/3`` facts for the navpoints whose sector changes over time.

    Empty for a constant allocation, which is what keeps every existing instance grounding
    exactly as it did before.
    """
    facts: List[str] = []
    for navaid, entries in change_points(assignment):
        for from_time, sector in entries:
            facts.append(f"{ASP_CHANGE_POINT_PREDICATE}({navaid},{sector},{from_time}).")
    return facts


def describe(assignment: np.ndarray) -> str:
    """One line saying whether the allocation varies over time, for verbose runs."""
    moving = [navaid for navaid, _ in change_points(assignment)]
    if not moving:
        return "navpoint-sector allocation: static (constant across all timesteps)"
    shown = ", ".join(str(navaid) for navaid in moving[:8])
    if len(moving) > 8:
        shown += f", ... (+{len(moving) - 8} more)"
    return (f"navpoint-sector allocation: time-varying, {len(moving)} navpoint(s) change sector "
            f"[{shown}]")
