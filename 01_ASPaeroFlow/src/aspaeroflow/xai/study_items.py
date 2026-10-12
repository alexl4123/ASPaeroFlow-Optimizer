"""Study items with answer keys from a recorded run (K32b): forward prediction, counterfactual, claim verification
and reason items for the replay study, written as a list the task panel can read and a table for the review.

    python -m src.aspaeroflow.xai.study_items <trace folder> --run A [--out DIR]     (from 01_ASPaeroFlow)

Every key is derived from the trace and its keep contrasts (xai/keep.py), never from a solver tie-break:
- forward (step n): which of the hotspot's decision flights the step changes; only steps where exactly one decision
  flight changed and the change is not a tie (keep contrast equal on every criterion);
- counterfactual (step n, flight F): the total overload after the step if F had been kept as it was (higher / the
  same; "lower" cannot be right, the step's overload is optimal); the sub-problem overload must equal the run's;
- claim (step n, flight F): a feasible plan that keeps F and is worse on exactly one criterion and equal on the others
  ("dominated"); the claim "this plan is better" is false. Whether it stays dominated after the rest of the run is
  not checked here (a forced-alternative continuation, open);
- reason (step n, flight F): why the step changed F, from the row reasons (xai/reasons.py explain_rows) and the keep
  contrast: the first criterion on which keeping F is worse, a later flight of the same aircraft, or a tie.
Items are proposed in a pool; five are selected per run (one forward, one counterfactual, one claim, two reason), in
step order, so that the step a participant may see (`cap`) never goes back within a block.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .keep import read_keep
from .plan import build_plan, clock
from .time_words import minutes_text

#: the minimised criteria in priority order (encoding.lp: @10 overload, @9 arrival delay, @8 open sectors, @7 changed)
CRITERIA = [("overload", "total overload"), ("delay", "arrival delay"), ("sectors", "open sectors"),
            ("changed", "flights sent off their filed plan")]
CANNOT_TELL = {"id": "z", "text": "Cannot tell from what the tool shows"}


def _costs(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not entry or not entry.get("feasible"):
        return None
    return {k: int(entry["costs"][k]) for k, _ in CRITERIA if k in entry.get("costs", {})}


def _diff(factual: Dict[str, int], other: Dict[str, int]) -> Dict[str, int]:
    return {k: other[k] - factual[k] for k, _ in CRITERIA if k in factual and k in other}


def _first_worse(d: Dict[str, int]) -> Optional[str]:
    """The first criterion in priority order on which the other plan differs, if it is worse there."""
    for k, _ in CRITERIA:
        if d.get(k, 0) != 0:
            return k if d[k] > 0 else None
    return None


class Run:
    """The trace of one run with what the items need: records, keep contrasts, row reasons, plan."""

    def __init__(self, folder: Path):
        from .session import ReplaySession
        self.folder = Path(folder)
        self.session = ReplaySession(self.folder)
        self.records = [r for r in self.session.records if r.get("accepted")]
        self.keep = read_keep(self.folder)
        self.plan = build_plan(self.folder)
        self.mps = self.plan["run"]["minutes_per_step"]
        self.per_day = self.plan["run"]["steps_per_day"]
        self._rows: Dict[int, Dict[str, Dict[str, Any]]] = {}

    def time(self, t: int) -> str:
        return f"time step {t} ({clock(t, self.mps, self.per_day)})"

    def rows(self, n: int) -> Dict[str, Dict[str, Any]]:
        if n not in self._rows:
            out = self.session.reasons(n)
            self._rows[n] = {r["ref"]: r for r in (out.get("rows") or [])} if isinstance(out, dict) else {}
        return self._rows[n]

    def factual(self, n: int) -> Optional[Dict[str, int]]:
        return _costs(self.keep.get(n, {}).get("factual"))

    def decision(self, n: int) -> List[int]:
        """The flights the step's solver was given (the hotspot's flights it took, xai/subproblem.py)."""
        return sorted(int(f) for f in self.session.explainer(n).sub.decision_flights)

    def keep_of(self, n: int, f: int) -> Optional[Dict[str, Any]]:
        return (self.keep.get(n, {}).get("flights") or {}).get(str(f))


def _hotspot_flights(record: Dict[str, Any]) -> Optional[List[int]]:
    flights = (record.get("hotspot") or {}).get("flights")
    return [int(x["id"]) for x in flights] if flights is not None else None


def _changed(record: Dict[str, Any]) -> List[int]:
    return sorted(int(f) for f in record.get("flight_changes") or {})


def forward_items(run: Run) -> List[Dict[str, Any]]:
    out = []
    for r in run.records:
        n, h = r["iteration"], r.get("hotspot") or {}
        decision = run.decision(n)
        changed = [f for f in _changed(r) if f in decision]
        if len(decision) < 2 or len(changed) != 1:
            continue
        shown = _hotspot_flights(r) or decision        # the options: the hotspot's flights (older traces: the decision flights)
        row = run.rows(n).get(f"flight:{changed[0]}") or {}
        if row.get("class") == "tie":
            continue
        options = [{"id": f"f{f}", "text": f"Flight {f}"} for f in sorted(shown)]
        options += [{"id": "none", "text": "None of these flights"}, CANNOT_TELL]
        out.append({
            "kind": "forward", "step": n, "cap": n - 1, "visible_in_e_on": False,
            "prompt": (f"Next comes step {n}. It deals with sector {h.get('sector')} at {run.time(h.get('time'))}, "
                       f"where {h.get('demand')} flights meet a capacity of {h.get('capacity')}. "
                       f"Which of these flights will the optimizer change in step {n}?"),
            "options": options, "key": f"f{changed[0]}",
            "source": f"step {n}: hotspot flights {sorted(shown)}, given to the solver {decision}, changed {changed}; "
                      f"row class {row.get('class')}",
        })
    return out


def counterfactual_items(run: Run) -> List[Dict[str, Any]]:
    out = []
    for r in run.records:
        n = r["iteration"]
        factual = run.factual(n)
        if not factual or factual.get("overload") != int(r["objectives"]["OVERLOAD"]):
            continue          # the sub-problem's overload must be the run's total overload
        for f in _changed(r):
            k = run.keep_of(n, f) or {}
            keep = _costs(k.get("keep"))
            if not keep:
                continue
            d = _diff(factual, keep)
            key = "higher" if d["overload"] > 0 else "same" if d["overload"] == 0 else None
            if key is None:
                continue
            out.append({
                "kind": "counterfactual", "step": n, "cap": n, "visible_in_e_on": True, "flight": f,
                "prompt": (f"In step {n} the optimizer changed flight {f}. Suppose flight {f} had been kept as it was "
                           f"before the step and step {n} had been solved again. The total overload after the step "
                           f"would be ..."),
                "options": [{"id": "higher", "text": "higher"}, {"id": "same", "text": "the same"},
                            {"id": "lower", "text": "lower"}, CANNOT_TELL],
                "key": key,
                "source": f"keep contrast step {n} flight {f}: overload {factual['overload']} -> {keep['overload']}",
            })
    return out


def claim_items(run: Run) -> List[Dict[str, Any]]:
    out = []
    last = run.records[-1]["iteration"] if run.records else 0
    for r in run.records:
        n = r["iteration"]
        factual = run.factual(n)
        if not factual:
            continue
        for f in _changed(r):
            keep = _costs((run.keep_of(n, f) or {}).get("keep"))
            if not keep:
                continue
            d = _diff(factual, keep)
            worse = [k for k, v in d.items() if v > 0]
            if len(worse) != 1 or any(v < 0 for v in d.values()):
                continue
            label = dict(CRITERIA)[worse[0]]
            unit = (f"{minutes_text(int(d[worse[0]] * run.mps))} more" if worse[0] == "delay" else f"{d[worse[0]]} more")
            same = [lbl for k, lbl in CRITERIA if k != worse[0] and k in d]
            equal = same[0] if len(same) == 1 else ", ".join(same[:-1]) + " and " + same[-1]
            out.append({
                "kind": "claim", "step": n, "cap": last, "visible_in_e_on": False, "flight": f,
                "prompt": (f"Consider plan X for step {n}: flight {f} is kept as it was before the step, and the step "
                           f"is otherwise solved as well as possible. Plan X has the same {equal} as the optimizer's "
                           f"choice and {unit} {label}. Claim: plan X is better than what the optimizer chose."),
                "options": [{"id": "true", "text": "True"}, {"id": "false", "text": "False"}, CANNOT_TELL],
                "key": "false",
                "source": f"keep contrast step {n} flight {f}: differences {d} (worse only on {worse[0]}); "
                          f"final-objective check open",
            })
    return out


REASON_OPTIONS = [
    {"id": "overload", "text": "Keeping it would have left more overload"},
    {"id": "delay", "text": "Keeping it would have caused more arrival delay, with the same overload"},
    {"id": "sectors", "text": "Keeping it would have needed more open sectors, with the same overload and delay"},
    {"id": "aircraft", "text": "It is a later flight of an aircraft whose earlier flight was moved"},
    {"id": "tie", "text": "Another flight could have changed instead; both choices were equally good"},
    CANNOT_TELL,
]


def reason_items(run: Run) -> List[Dict[str, Any]]:
    out = []
    for r in run.records:
        n = r["iteration"]
        factual = run.factual(n)
        for f in _changed(r):
            row = run.rows(n).get(f"flight:{f}") or {}
            cls = row.get("class")
            key = None
            if cls == "rule":
                key = "aircraft"
            elif cls == "tie":
                key = "tie"
            elif factual:
                keep = _costs((run.keep_of(n, f) or {}).get("keep"))
                worse = _first_worse(_diff(factual, keep)) if keep else None
                key = worse if worse in ("overload", "delay", "sectors") else None
            if key is None:
                continue
            out.append({
                "kind": "reason", "step": n, "cap": n, "visible_in_e_on": True, "flight": f,
                "prompt": f"In step {n}, why did the optimizer change flight {f} instead of keeping it as it was?",
                "options": REASON_OPTIONS, "key": key,
                "source": f"row class {cls}; " + (row.get("text") or "")[:160],
            })
    return out


def select(pool: Dict[str, List[Dict[str, Any]]], run_id: str) -> List[Dict[str, Any]]:
    """One forward, one counterfactual, one claim, two reason items; spread over the run; caps in step order; the
    counterfactual's key alternates between 'higher' and 'same' by run; no flight twice."""
    rnd = random.Random(f"study-items-{run_id}")
    used: set = set()
    picked: List[Dict[str, Any]] = []

    def take(items: List[Dict[str, Any]], prefer=None) -> None:
        items = [i for i in items if i.get("flight") not in used and i not in picked]
        if prefer:
            items = [i for i in items if prefer(i)] or items
        if not items:
            return
        i = rnd.choice(items)
        picked.append(i)
        if i.get("flight") is not None:
            used.add(i["flight"])
        if i["kind"] == "forward":
            used.add(int(i["key"][1:]))

    # the scarce kinds first, so that a later pick cannot take the only candidate's flight
    take(pool["claim"])
    take(pool["forward"])
    want = "higher" if sum(map(ord, run_id)) % 2 == 0 else "same"
    take(pool["counterfactual"], lambda i: i["key"] == want)
    take(pool["reason"], lambda i: i["key"] in ("overload", "delay", "sectors"))
    take(pool["reason"], lambda i: i["key"] in ("aircraft", "tie"))
    picked.sort(key=lambda i: (i["cap"], i["step"]))
    return picked


def missing_kinds(items: List[Dict[str, Any]]) -> List[str]:
    """The kinds of the planned mix (one forward, counterfactual, claim, two reason) that could not be filled."""
    want = {"forward": 1, "counterfactual": 1, "claim": 1, "reason": 2}
    have = {k: sum(1 for i in items if i["kind"] == k) for k in want}
    return [k for k, n in want.items() for _ in range(n - have[k])]


def build_items(folder: Path, run_id: str) -> Dict[str, Any]:
    run = Run(folder)
    pool = {"forward": forward_items(run), "counterfactual": counterfactual_items(run),
            "claim": claim_items(run), "reason": reason_items(run)}
    items = select(pool, run_id)
    for k, i in enumerate(items, 1):
        i["id"] = f"{run_id}{k}"
        i["type"] = "single_choice"
    return {"run": run_id, "trace": str(folder), "items": items, "missing": missing_kinds(items),
            "pool": {k: len(v) for k, v in pool.items()}, "pool_items": pool}


def review_table(result: Dict[str, Any]) -> str:
    lines = [f"# Study items, run {result['run']}", "", f"Trace: `{result['trace']}`. Pool sizes: "
             + ", ".join(f"{k} {v}" for k, v in result["pool"].items()) + "."
             + (f" **Not filled: {', '.join(result['missing'])}.**" if result.get("missing") else ""), ""]
    for i in result["items"]:
        key = next(o["text"] for o in i["options"] if o["id"] == i["key"])
        lines += [f"## {i['id']} ({i['kind']}, step {i['step']}, may see up to step {i['cap']}, "
                  f"answer visible in E-on: {'yes' if i['visible_in_e_on'] else 'no'})", "", i["prompt"], ""]
        lines += [f"- {'**' if o['id'] == i['key'] else ''}{o['text']}{'** (key)' if o['id'] == i['key'] else ''}"
                  for o in i["options"]]
        lines += ["", f"Key: {key}. Derived from: {i['source']}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = build_items(Path(a.trace), a.run)
    out = Path(a.out) if a.out else Path(a.trace)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"study_items_{a.run}.json").write_text(json.dumps(result, indent=1))
    (out / f"study_items_{a.run}.md").write_text(review_table(result))
    print(f"wrote {out / f'study_items_{a.run}.json'} and the review table; pool {result['pool']}; "
          f"selected {[(i['id'], i['kind'], i['step']) for i in result['items']]}")


if __name__ == "__main__":
    main()
