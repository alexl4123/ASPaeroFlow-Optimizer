"""Why one ASPaeroFlow step worked on its spot, in checked sentences (the step header of the XAI interface).

A trace record becomes input facts for reasons.lp (plain stratified ASP, its own Control, never together
with encoding.lp); reasons.lp derives the reasons of the step, their arguments and sources, the objective
deltas, and error/1 atoms for every documented rule of the optimizer the record breaks. reasons_text.lp
holds the text templates; this module fills them.

    explain_step(record, previous=..., run=..., trace_folder=...)   one recorded iteration
    explain_start(run)                                               the filed plan (step 0)
    step_context(records_by_iteration, n)                            previous record and rejected attempts before n

Both explain functions return {"iteration", "kept", "run_length", "run_kept", "arrival_delay_metric",
"lines": [{"id", "kind", "level", "order", "text", "refs", "source"}], "deltas": [{"key", "before", "after",
"change", "direction", "hidden"}], "errors": [{"code", "text"}]}.
"""
from __future__ import annotations

import importlib.resources
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import clingo

from .subproblem import Subproblem

#: Objective keys of the trace, in the order of the interface (S0 step-words.ts OBJECTIVE_KEYS).
OBJECTIVE_KEYS = ("OVERLOAD", "ARRIVAL-DELAY", "SECTOR-NUMBER", "REROUTE", "SECTOR-DIFF", "RECONFIG")
#: The one place where trace keys become ASP constants.
ASP_KEY = {"OVERLOAD": "overload", "ARRIVAL-DELAY": "arrival_delay", "SECTOR-NUMBER": "sector_number",
           "REROUTE": "reroute", "SECTOR-DIFF": "sector_diff", "RECONFIG": "reconfig"}
TRACE_KEY = {v: k for k, v in ASP_KEY.items()}
PARAMETERS = ("max_aircraft", "additional_time_increase", "max_delay_per_iteration",
              "number_capacity_management_configs", "failed_attempts_before")
AUDIENCE = "developer"
#: reason_arg keys whose values are flight numbers (each becomes a ref "flight:F").
FLIGHT_KEYS = ("taken", "tied", "longer", "left_out", "changed", "unchanged")

_CONSTANT = re.compile(r"^[a-z][A-Za-z0-9_]*$")
#: lp facts written by optimize_flights.py: paths(F,0..P-1) and actual_flight_operations_start_time(F,T,P).
_PATHS = re.compile(r"\bpaths\((\d+),(\d+)\.\.(-?\d+)\)")
_STARTS = re.compile(r"\bactual_flight_operations_start_time\((\d+),(\d+),(\d+)\)")
#: lp facts flightPlan(F,T,Navpoint): the current trajectory of a solver flight; its first time is the current start.
_PLAN = re.compile(r"\bflightPlan\((\d+),(\d+),(\d+)\)")


def program() -> str:
    """reasons.lp and reasons_text.lp, read from the package (live and replay run from other folders)."""
    folder = importlib.resources.files(__package__)
    return (folder / "reasons.lp").read_text(encoding="utf-8") + "\n" + \
        (folder / "reasons_text.lp").read_text(encoding="utf-8")


# ------------------------------------------------------------------ facts

def _term(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and _CONSTANT.match(value):
        return value
    raise ValueError(f"not an integer or constant: {value!r}")


class Facts:
    def __init__(self):
        self.lines: List[str] = []

    def add(self, name: str, *args: Any) -> None:
        self.lines.append(f"{name}({','.join(_term(a) for a in args)})." if args else f"{name}.")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_facts(facts: Facts, run: Dict[str, Any]) -> None:
    for option in ("sequential_execution", "minimize_number_sectors"):
        value = run.get(option)
        if value is None:
            facts.add("option_assumed", option)
        else:
            if isinstance(value, str):
                value = value.strip().lower() in ("true", "1", "yes")
            facts.add("run_option", option, bool(value))
    composite = run.get("composite_sector_function")
    if composite is None:
        facts.add("option_assumed", "composite_sector_function")
    else:
        facts.add("run_option", "composite_sector_function", "max" if composite == "max" else "other")
    window = _int(run.get("evaluation_window"))
    if window is not None:
        facts.add("evaluation_window", window)
        # written together with evaluation_window, so its initial SECTOR-DIFF is computed (not the old constant 0)
        if _int((run.get("initial_objectives") or {}).get("SECTOR-DIFF")) == 0:
            facts.add("filed_layout_static")


def _hotspot_facts(facts: Facts, n: int, record: Dict[str, Any], full: bool) -> Optional[Tuple[int, int]]:
    h = record.get("hotspot") or {}
    s, t = _int(h.get("sector")), _int(h.get("time"))
    if s is None or t is None:
        return None
    facts.add("hotspot", n, s, t)
    if not full:
        return s, t
    demand, capacity, overload = _int(h.get("demand")), _int(h.get("capacity")), _int(h.get("overload"))
    if demand is not None and capacity is not None:
        facts.add("hotspot_load", n, s, t, demand, capacity)
    if overload is not None:
        facts.add("hotspot_overload", n, s, t, overload)
    if isinstance(h.get("vertices"), list) and h["vertices"]:
        facts.add("hotspot_navpoints", n, s, len(h["vertices"]))
    if isinstance(h.get("flights"), list):
        facts.add("occupants_recorded", n)
        for rank, f in enumerate(h["flights"], start=1):
            facts.add("occupant", n, int(f["id"]), int(f["duration"]), rank)
    if _int(h.get("taken")) is not None:
        facts.add("taken_count", n, int(h["taken"]))
    return s, t


def _param_facts(facts: Facts, n: int, record: Dict[str, Any]) -> None:
    p = record.get("parameters") or {}
    for name in PARAMETERS:
        value = _int(p.get(name))
        if value is not None:
            facts.add("param", n, name, value)


def _objective_facts(facts: Facts, n: int, which: str, objectives: Optional[Dict[str, Any]]) -> None:
    for key, name in ASP_KEY.items():
        value = _int((objectives or {}).get(key))
        if value is not None:
            facts.add("objective", n, which, name, value)


def _instance_facts(facts: Facts, n: int, record: Dict[str, Any], trace_folder: Optional[Path]) -> Optional[Subproblem]:
    subs = record.get("subproblems") or []
    if len(subs) > 1:
        raise ValueError(f"iteration {n} has {len(subs)} sub-problems; the explanation reads exactly one")
    if trace_folder is None or not subs or not subs[0].get("instance_file"):
        return None
    path = Path(trace_folder) / subs[0]["instance_file"]
    if not path.exists():
        return None
    sub = Subproblem.from_instance(path.read_text(encoding="utf-8"))
    facts.add("instance_read", n)
    for f in sub.decision_flights:
        facts.add("subproblem_flight", n, int(f))
    for leg, parent in sorted(sub.parent.items()):
        facts.add("later_leg", n, int(leg), int(parent))
    # departure options of the solver flights: number of candidate paths, distinct start times among them
    decision = {int(f) for f in sub.decision_flights}
    path_count: Dict[int, int] = {}
    for f, lo, hi in _PATHS.findall(sub.instance):
        if int(f) in decision:
            path_count[int(f)] = path_count.get(int(f), 0) + max(0, int(hi) - int(lo) + 1)
    starts: Dict[int, set] = {}
    for f, t, _ in _STARTS.findall(sub.instance):
        if int(f) in decision:
            starts.setdefault(int(f), set()).add(int(t))
    for f in sorted(path_count):
        facts.add("offered_paths", n, f, path_count[f])
    current_start: Dict[int, int] = {}
    for f, t, _ in _PLAN.findall(sub.instance):
        if int(f) in decision:
            current_start[int(f)] = min(int(t), current_start.get(int(f), int(t)))
    for f in sorted(starts):
        facts.add("offered_starts", n, f, len(starts[f]))
        if f in current_start:          # the delays offered: start time minus the current start
            facts.add("offered_delay", n, f, min(starts[f]) - current_start[f], max(starts[f]) - current_start[f])
    for c, config in sorted(sub.configs.items()):
        if config.outside_overload is not None:          # config/2: the layouts the encoding can choose
            facts.add("layout_option", n, int(c))
    return sub


def step_facts(record: Dict[str, Any], *, previous: Optional[Dict[str, Any]], run: Dict[str, Any],
               trace_folder: Optional[Path], rejected_before: Sequence[Dict[str, Any]] = (),
               run_length: Optional[int] = None, run_kept: Optional[int] = None) -> str:
    """The input facts of reasons.lp for one record."""
    facts = Facts()
    n = int(record["iteration"])
    accepted = bool(record.get("accepted"))
    facts.add("step", n)
    facts.add("kept" if accepted else "rejected", n)
    _hotspot_facts(facts, n, record, full=True)
    _param_facts(facts, n, record)
    _instance_facts(facts, n, record, trace_folder)
    if accepted and isinstance(record.get("flight_changes"), dict):
        facts.add("flights_recorded", n)
        for f in sorted(int(k) for k in record["flight_changes"]):
            facts.add("changed_flight", n, f)
    sectors = record.get("sector_changes") or {}
    prev = sectors.get("prev_sector_config") or {}
    if accepted and prev:
        facts.add("sector_record", n)
        for s, part in prev.items():
            for v in (part or {}).get("vertices") or []:
                facts.add("part_before", n, int(s), int(v))
        for p, part in (sectors.get("post_sector_config") or {}).items():
            for v in (part or {}).get("vertices") or []:
                facts.add("part_after", n, int(p), int(v))
    if previous is not None:
        _objective_facts(facts, n, "before", previous.get("objectives"))
    elif n == 1:
        _objective_facts(facts, n, "before", run.get("initial_objectives"))
    _objective_facts(facts, n, "after", record.get("objectives"))
    before = _int((record.get("objectives") or {}).get("OVERLOAD-BEFORE"))
    if before is not None:
        facts.add("recorded_before", n, "overload", before)
    for r in rejected_before:
        rn = int(r["iteration"])
        facts.add("rejected_before", n, rn)
        _hotspot_facts(facts, rn, r, full=False)
        _param_facts(facts, rn, r)
    _run_facts(facts, run)
    if run_length is not None:
        facts.add("run_length", int(run_length))
    if run_kept is not None:
        facts.add("run_kept", int(run_kept))
    return facts.text()


def start_facts(run: Dict[str, Any]) -> str:
    facts = Facts()
    overload = _int(run.get("initial_overload"))
    if overload is None:
        overload = _int((run.get("initial_objectives") or {}).get("OVERLOAD"))
    if overload is not None:
        facts.add("start_overload", overload)
    for s, o in sorted((run.get("initial_sector_overload") or {}).items(), key=lambda x: int(x[0])):
        facts.add("start_sector_overload", int(s), int(o))
    _run_facts(facts, run)
    return facts.text()


# ------------------------------------------------------------------ solving

def solve(facts: str) -> Tuple[List[clingo.Symbol], Dict[str, str], Dict[str, str]]:
    """Shown atoms of the one model, the developer templates and the error texts."""
    ctl = clingo.Control(["--warn=none", "0"])
    ctl.add("base", [], program())
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])
    models: List[List[clingo.Symbol]] = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(list(model.symbols(shown=True)))
    if len(models) != 1:
        raise RuntimeError(f"reasons.lp must have exactly one model, it has {len(models)}")
    return models[0], *_texts(ctl)


def _texts(ctl: clingo.Control) -> Tuple[Dict[str, str], Dict[str, str]]:
    templates = {str(a.symbol.arguments[0]): a.symbol.arguments[2].string
                 for a in ctl.symbolic_atoms.by_signature("template", 3)
                 if str(a.symbol.arguments[1]) == AUDIENCE}
    errors = {str(a.symbol.arguments[0]): a.symbol.arguments[1].string
              for a in ctl.symbolic_atoms.by_signature("error_text", 2)}
    return templates, errors


def _value(sym: clingo.Symbol) -> Any:
    if sym.type == clingo.SymbolType.Number:
        return sym.number
    if sym.type == clingo.SymbolType.Function and sym.name == "":
        return tuple(_value(a) for a in sym.arguments)
    if sym.type == clingo.SymbolType.String:
        return sym.string
    return str(sym)


# ------------------------------------------------------------------ text

def join(items: Iterable[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def periods(n: int) -> str:
    return "1 time period" if n == 1 else f"{n} time periods"


def _format(key: str, values: List[Any], templates: Dict[str, str]) -> str:
    if key == "limits":
        return "".join(templates[v] for v in values)
    if key == "legs":                       # [(leg, parent)] grouped by parent: "26 (after 19), 47 and 58 (after 18)"
        groups: List[Tuple[int, List[int]]] = []
        for leg, parent in values:
            if groups and groups[-1][0] == parent:
                groups[-1][1].append(leg)
            else:
                groups.append((parent, [leg]))
        return ", ".join(f"{join(str(l) for l in legs)} (after {parent})" for parent, legs in groups)
    if key == "taken_durations":            # "19 (3 time periods) and 18 (4)"
        return join(f"{f} ({periods(d)})" if i == 0 else f"{f} ({d})" for i, (f, d) in enumerate(values))
    if key == "longer":                     # "; 11 has a longer one" (optional part of the tie lines)
        return "; " + join(str(f) for f in values) + (" has a longer one" if len(values) == 1 else " have longer ones")
    if key == "left_out_durations":         # "17 has 6 and 20 has 7"
        return join(f"{f} has {d}" for f, d in values)
    if key == "sectors":                    # "6 (44), 0 (29) and 21 (14)"
        return join(f"{s} ({o})" for s, o in values)
    return join(str(v) for v in values)


def _refs(key: str, values: List[Any], n: int, start: bool) -> List[Tuple[str, str]]:
    if key in FLIGHT_KEYS:
        return [(f"flight:{v}", str(v)) for v in values]
    if key in ("taken_durations", "left_out_durations"):
        return [(f"flight:{f}", str(f)) for f, _ in values]
    if key == "legs":
        return [r for leg, parent in values for r in ((f"flight:{leg}", str(leg)), (f"flight:{parent}", str(parent)))]
    if key == "sector":
        return [(f"sector:{v}@{n}:before", str(v)) for v in values]
    if key == "part_ids":
        return [(f"sector:{v}@{n}:after", str(v)) for v in values]
    if key == "sectors" and start:
        return [(f"sector:{s}@0:initial", str(s)) for s, _ in values]
    return []


def _render(atoms: List[clingo.Symbol], templates: Dict[str, str], n: int) -> List[Dict[str, Any]]:
    kinds: Dict[str, str] = {}
    shown: Dict[str, Tuple[str, int]] = {}
    args: Dict[str, Dict[str, List[Tuple[int, Any]]]] = {}
    sources: Dict[str, List[str]] = {}
    for a in atoms:
        if a.name == "reason_kind":
            kinds[str(a.arguments[0])] = str(a.arguments[1])
        elif a.name == "show" and str(a.arguments[1]) == AUDIENCE:
            shown[str(a.arguments[0])] = (str(a.arguments[2]), a.arguments[3].number)
        elif a.name == "reason_arg":
            args.setdefault(str(a.arguments[0]), {}).setdefault(str(a.arguments[1]), []).append(
                (a.arguments[2].number, _value(a.arguments[3])))
        elif a.name == "reason_source":
            sources.setdefault(str(a.arguments[0]), []).append(str(a.arguments[1]))
    lines = []
    for rid, (level, order) in shown.items():
        kind = kinds[rid]
        template = templates[kind]
        values = {key: [v for _, v in sorted(items, key=lambda x: x[0])] for key, items in args.get(rid, {}).items()}
        fill = {key: _format(key, vs, templates) for key, vs in values.items()}
        fill.setdefault("limits", "")
        fill.setdefault("longer", "")
        text = template.format(**fill)
        refs: List[Dict[str, str]] = []
        seen = set()
        for _, key, _, _ in string.Formatter().parse(template):
            if key is None or key not in values:
                continue
            for ref, ref_text in _refs(key, values[key], n, rid in ("start", "start_rule")):
                if ref not in seen:
                    seen.add(ref)
                    refs.append({"ref": ref, "text": ref_text})
        lines.append({"id": rid, "kind": kind, "level": level, "order": order, "text": text, "refs": refs,
                      "source": sorted(sources.get(rid, []))})
    lines.sort(key=lambda line: (0 if line["level"] == "visible" else 1, line["order"]))
    return lines


def _deltas(atoms: List[clingo.Symbol], after: Dict[str, Any], hidden_keys: Iterable[str]) -> List[Dict[str, Any]]:
    found = {}
    for a in atoms:
        if a.name == "delta":
            key = TRACE_KEY[str(a.arguments[1])]
            b, v = a.arguments[2].number, a.arguments[3].number
            found[key] = {"key": key, "before": b, "after": v, "change": v - b, "direction": str(a.arguments[4])}
    hidden = set(hidden_keys)
    out = []
    for key in OBJECTIVE_KEYS:
        if key in found:
            out.append(found[key] | {"hidden": key in hidden})
        elif _int(after.get(key)) is not None:
            out.append({"key": key, "before": None, "after": int(after[key]), "change": None, "direction": None,
                        "hidden": key in hidden})
    return out


def _errors(atoms: List[clingo.Symbol], error_texts: Dict[str, str]) -> List[Dict[str, Any]]:
    errors = sorted((a.arguments[0] for a in atoms if a.name == "error"), key=str)
    return [{"code": e.name, "text": error_texts.get(e.name, e.name)} for e in errors]


def _result(n: int, kept, run: Dict[str, Any], run_length, run_kept, lines, deltas, errors) -> Dict[str, Any]:
    return {"iteration": n, "kept": kept, "run_length": run_length, "run_kept": run_kept,
            "arrival_delay_metric": run.get("arrival_delay_metric"),
            "lines": lines, "deltas": deltas, "errors": errors}


def _finish(n: int, atoms, templates, error_texts):
    """Lines and errors; a record error replaces every line, a template problem gives render_error."""
    errors = _errors(atoms, error_texts)
    if errors:
        first = next(a.arguments[0] for a in sorted((a for a in atoms if a.name == "error"), key=lambda a: str(a.arguments[0])))
        lines = [{"id": f"record_error({n})", "kind": "record_error", "level": "visible", "order": 1,
                  "text": templates["record_error"].format(text=errors[0]["text"]), "refs": [],
                  "source": [f"error({first})"]}]
        return lines, errors
    try:
        return _render(atoms, templates, n), errors
    except (KeyError, IndexError, ValueError):
        lines = [{"id": f"render_error({n})", "kind": "render_error", "level": "visible", "order": 1,
                  "text": templates.get("render_error", "The explanation text of this step could not be produced."),
                  "refs": [], "source": []}]
        return lines, [{"code": "render_error", "text": error_texts.get("render_error", "render_error")}]


def explain_step(record: Dict[str, Any], *, previous: Optional[Dict[str, Any]], run: Dict[str, Any],
                 trace_folder: Optional[Path], rejected_before: Sequence[Dict[str, Any]] = (),
                 run_length: Optional[int] = None, run_kept: Optional[int] = None) -> Dict[str, Any]:
    """The step header of one recorded iteration (see the module docstring for the shape)."""
    n = int(record["iteration"])
    facts = step_facts(record, previous=previous, run=run, trace_folder=trace_folder,
                       rejected_before=rejected_before, run_length=run_length, run_kept=run_kept)
    atoms, templates, error_texts = solve(facts)
    lines, errors = _finish(n, atoms, templates, error_texts)
    codes = {e["code"] for e in errors}
    hidden = set(OBJECTIVE_KEYS) if "missing_previous" in codes else ({"OVERLOAD"} if "before_mismatch" in codes else set())
    deltas = _deltas(atoms, record.get("objectives") or {}, hidden)
    return _result(n, bool(record.get("accepted")), run, run_length, run_kept, lines, deltas, errors)


def explain_start(run: Dict[str, Any], *, run_length: Optional[int] = None,
                  run_kept: Optional[int] = None) -> Dict[str, Any]:
    """The filed plan before step 1: its total overload and the overloaded sectors."""
    atoms, templates, error_texts = solve(start_facts(run))
    lines, errors = _finish(0, atoms, templates, error_texts)
    initial = run.get("initial_objectives") or {}
    deltas = [{"key": key, "before": None, "after": int(initial[key]), "change": None, "direction": None,
               "hidden": False} for key in OBJECTIVE_KEYS if _int(initial.get(key)) is not None]
    return _result(0, None, run, run_length, run_kept, lines, deltas, errors)


def step_context(records_by_iteration: Dict[int, Dict[str, Any]], n: int):
    """(previous record by iteration number or None, the consecutive rejected records right before n, oldest first)."""
    previous = records_by_iteration.get(n - 1)
    rejected: List[Dict[str, Any]] = []
    m = n - 1
    while m in records_by_iteration and not records_by_iteration[m].get("accepted"):
        rejected.append(records_by_iteration[m])
        m -= 1
    return previous, list(reversed(rejected))
