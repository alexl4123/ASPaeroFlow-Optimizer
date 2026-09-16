"""The navpoint-to-sector allocation, shared by every solver folder in this repository.

Every solver works from the same dense ``(|N| x |T|)`` integer array, in which entry
``[nav, t]`` is the sector navpoint ``nav`` belongs to at timestep ``t``. That array has always
been the interface between the input files and the rest of each solver; what it could not
express until now is an allocation that CHANGES over time, because it was filled by broadcasting
a single static assignment across every column.

Two input files feed it, side by side in the instance bundle:

``navaid_sector_assignment.csv``   ``Navaid_ID,Sector_ID``
    The static allocation. Required. Read exactly as before.

``navaid_sector_schedule.csv``     ``Navaid_ID,Sector_ID,From_Time``
    The time-varying allocation. OPTIONAL. Sparse change-points: a row says "from timestep
    ``From_Time`` onward, ``Navaid_ID`` sits in ``Sector_ID``, until the next row for the same
    navpoint". Rows are applied in ``From_Time`` order on top of the static broadcast.

A solver looks for the schedule beside the static file it was given and nowhere else, which is
where the data generator puts it. When there is no such file the array is built exactly as it
was before this module existed, so every previously published instance reproduces its old
numbers.

The schedule's ``From_Time = 0`` rows must agree with ``navaid_sector_assignment.csv``: the
generator writes them from the same allocation, so a disagreement means the two files were not
produced together, and loading either one silently would be a mis-load. It is an error. A
consequence is that a schedule whose rows all carry ``From_Time = 0`` builds exactly the static
array.

Three solver folders call ``build_assignment`` from an instance method and the MIP calls it from
a classmethod, so the logic lives here once and each folder reaches it through its own
``navpoint_sector_allocation_bootstrap.py`` shim, the same way ``common/arrival_delay.py`` is
reached.

The ASP facts for a time-varying allocation are written by ``02_ASP/translate.py``
(``convert_navaid_sector_schedule``), from this array; a constant allocation adds none.
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


class ScheduleError(ValueError):
    """A schedule file that cannot be trusted to mean what it appears to say."""


# ---------------------------------------------------------------------------
# Locating the file
# ---------------------------------------------------------------------------

def schedule_path_beside(navaid_sector_path) -> Optional[Path]:
    """Where a time-varying allocation for this instance lives, if it exists.

    Returns None when there is no such file, which is the ordinary case for every instance
    published so far.
    """
    if navaid_sector_path is None:
        return None
    candidate = Path(navaid_sector_path).with_name(SCHEDULE_FILENAME)
    return candidate if candidate.is_file() else None


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


def load_schedule_for(navaid_sector_path) -> Optional[List[Tuple[int, int, int]]]:
    """The schedule beside this static allocation, or None when the instance has none."""
    path = schedule_path_beside(navaid_sector_path)
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
                     airports=None,
                     fill_value: int = -1,
                     compress: bool = False) -> np.ndarray:
    """The navpoint-to-sector allocation for every timestep, as an ``(|N| x |T|)`` array.

    With ``schedule=None`` this is the historical ``create_initial_navpoint_sector_assignment``,
    unchanged: the static allocation broadcast across every column, and a self-sector row for
    any navpoint the static file does not mention.

    With a schedule the change-points are applied on top of that broadcast, in ``From_Time``
    order, each one holding until the navpoint's next change-point. ``airports`` (the airport
    navpoints) is used only to check a schedule, never to build the array.

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
        _apply_schedule(output, schedule, navaid_sector, airports)

    return output


def _apply_schedule(output: np.ndarray, schedule, navaid_sector: np.ndarray, airports) -> None:
    """Overwrite `output` in place with the change-points, then check what they imply.

    The checks run only over the entries a schedule actually moved, so an instance without one
    is never subjected to them and cannot start failing because of this function.
    """
    n_navpoints, n_times = output.shape
    static = output[:, 0].copy()
    listed = set(int(navaid) for navaid in navaid_sector[:, 0])

    for navaid, sector, from_time in sorted(schedule, key=lambda entry: (entry[0], entry[2])):
        if navaid not in listed:
            raise ScheduleError(
                f"{SCHEDULE_FILENAME} names navpoint {navaid}, which navaid_sector_assignment.csv "
                f"does not list (it lists {len(listed)} navpoints, the largest "
                f"{n_navpoints - 1})"
            )
        if sector >= n_navpoints:
            raise ScheduleError(
                f"{SCHEDULE_FILENAME} puts navpoint {navaid} into sector {sector}, which is not a "
                f"navpoint of this instance (the largest is {n_navpoints - 1})"
            )
        if from_time == 0 and sector != static[navaid]:
            raise ScheduleError(
                f"{SCHEDULE_FILENAME} puts navpoint {navaid} into sector {sector} at From_Time 0, "
                f"but navaid_sector_assignment.csv puts it into sector {static[navaid]}. The "
                "schedule's From_Time 0 rows must be the static allocation; the two files "
                "disagree, so neither can be trusted."
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

    changed_columns = np.flatnonzero((output != static[:, None]).any(axis=0))

    # A navpoint may only sit in a sector that is open at that timestep, meaning the sector's own
    # navpoint is assigned to itself. The ASP encoding states this as an integrity constraint
    # (encoding.lp, `not navpoint_sector(SEC,SEC,T)`), so breaking it turns into an unexplained
    # UNSAT; the MIP and ASPaeroFlow would instead read capacities off a sector nothing keeps
    # open. Catch it here, where the file can be named.
    #
    # Whole columns are checked, not just the moved entries: moving a sector's own navpoint away
    # strands the members that did NOT move. Only the columns the schedule changed are checked,
    # which leaves an instance without a schedule untouched.
    for timestep in changed_columns:
        column = output[:, timestep]
        stranded = np.flatnonzero(column[column] != column)
        if stranded.size:
            navaid = int(stranded[0])
            sector = int(column[navaid])
            raise ScheduleError(
                f"{SCHEDULE_FILENAME} leaves navpoint {navaid} in sector {sector} at "
                f"timestep {timestep}, but sector {sector} is not open then: navpoint {sector} "
                f"is itself assigned to sector {int(column[sector])}. A sector is open only while "
                f"its own navpoint is assigned to it ({stranded.size} navpoint(s) affected at "
                "this timestep)."
            )

    # Airports are sectors of their own and nothing else joins them: encoding.lp derives
    # navpoint_sector(A,A,T) for every airport A and forbids mixing airports and en-route
    # navpoints in one sector. The Python solvers take a flight's origin and destination SECTOR
    # as its origin and destination navpoint, which is only sound under the same rule.
    if airports is not None and changed_columns.size:
        is_airport = np.zeros(n_navpoints, dtype=bool)
        airport_ids = np.asarray(airports, dtype=np.int64).ravel()
        is_airport[airport_ids[airport_ids < n_navpoints]] = True
        for timestep in changed_columns:
            column = output[:, timestep]
            moved_airports = np.flatnonzero(is_airport & (column != np.arange(n_navpoints)))
            if moved_airports.size:
                navaid = int(moved_airports[0])
                raise ScheduleError(
                    f"{SCHEDULE_FILENAME} moves airport {navaid} into sector {int(column[navaid])} "
                    f"at timestep {timestep}; an airport is always its own sector"
                )
            joined = np.flatnonzero(~is_airport & is_airport[column])
            if joined.size:
                navaid = int(joined[0])
                raise ScheduleError(
                    f"{SCHEDULE_FILENAME} puts en-route navpoint {navaid} into airport sector "
                    f"{int(column[navaid])} at timestep {timestep}; an airport sector holds "
                    "only its airport"
                )


# ---------------------------------------------------------------------------
# Describing the array
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


def epoch_starts(assignment: np.ndarray) -> List[int]:
    """The timesteps at which the allocation as a whole changes, always starting with 0.

    Between two consecutive entries every column of the array is the same partition.
    """
    breaks = np.flatnonzero((assignment[:, 1:] != assignment[:, :-1]).any(axis=0)) + 1
    return [0] + [int(t) for t in breaks]


def is_time_varying(assignment: np.ndarray) -> bool:
    """Whether any navpoint changes sector over the horizon."""
    return bool((assignment[:, 1:] != assignment[:, :-1]).any())


def describe(assignment: np.ndarray) -> str:
    """One line saying whether the allocation varies over time, for verbose runs."""
    moving = [navaid for navaid, _ in change_points(assignment)]
    if not moving:
        return "navpoint-sector allocation: static (constant across all timesteps)"
    shown = ", ".join(str(navaid) for navaid in moving[:8])
    if len(moving) > 8:
        shown += f", ... (+{len(moving) - 8} more)"
    return (f"navpoint-sector allocation: time-varying, {len(moving)} navpoint(s) change sector "
            f"[{shown}], {len(epoch_starts(assignment))} distinct partitions over time")
