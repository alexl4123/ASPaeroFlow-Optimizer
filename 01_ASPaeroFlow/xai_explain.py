#!/usr/bin/env python3
"""Ask an ASPaeroFlow iteration trace questions (written with main.py --xai-trace-dir).

    python 01_ASPaeroFlow/xai_explain.py TRACE --iteration 6 --hotspot --flight 18 --sectors
    python 01_ASPaeroFlow/xai_explain.py TRACE --iteration 6 --alternatives
    python 01_ASPaeroFlow/xai_explain.py TRACE --iteration 6 --what-if "keep 18" "avoid 22"
    python 01_ASPaeroFlow/xai_explain.py TRACE --summary          # every iteration, one paragraph each
Add --json for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.aspaeroflow.xai.contrastive import IterationExplainer, Lock  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402


def _print(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"\n## {result.get('question', '')}")
    print(result.get("answer", ""))
    for c in result.get("contrasts", []):
        print(f"  - {c['text']}")
        if c.get("ladder"):
            for row in c["ladder"]:
                mark = "  <- decides" if row["decides"] else ""
                print(f"      {row['text']:<52} {row['chosen']:>5} vs {row['alternative']:>5}{mark}")
    if "ladder" in result and result.get("ladder"):
        for row in result["ladder"]:
            mark = "  <- decides" if row["decides"] else ""
            print(f"      {row['text']:<52} {row['chosen']:>5} vs {row['alternative']:>5}{mark}")
    for row in result.get("alternatives", []):
        what = f"flight {row['flight']} via {'-'.join(map(str, row['route']))}" if row["kind"] == "route" \
            else f"configuration {row['config']} ({row['number_sectors']} sectors)"
        costs = "infeasible" if row["costs"] is None else " ".join(f"{k}={v}" for k, v in row["costs"].items())
        print(f"  {'*' if row['chosen'] else ' '} {what:<45} {costs}"
              + (f"  (worse on {row['deciding_level']})" if row.get("deciding_level") else ""))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--hotspot", action="store_true")
    p.add_argument("--flight", type=int, action="append", default=[])
    p.add_argument("--sectors", action="store_true")
    p.add_argument("--tie", action="store_true")
    p.add_argument("--alternatives", action="store_true")
    p.add_argument("--what-if", nargs="+", default=None, metavar="LOCK")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    trace = TraceReader(args.trace)
    if args.summary:
        for record in trace:
            ex = IterationExplainer(trace, record["iteration"])
            h = ex.why_hotspot()
            status = "accepted" if record["accepted"] else "rejected (overload did not drop)"
            print(f"\n### Iteration {record['iteration']}: {status}")
            print(h["answer"])
            if record["accepted"]:
                for f in ex.sub.decision_flights:
                    print("  " + ex.why_flight(f)["answer"])
                print("  " + ex.why_sectors()["answer"])
                tie = ex.tie_check()
                if tie["tie"]:
                    print("  Tie: an equally good answer exists (" + "; ".join(tie["equally_good"]) + ").")
        return

    if args.iteration is None:
        p.error("--iteration is required unless --summary is given")
    ex = IterationExplainer(trace, args.iteration)
    if args.hotspot:
        _print(ex.why_hotspot(), args.json)
    for f in args.flight:
        _print(ex.why_flight(f), args.json)
    if args.sectors:
        _print(ex.why_sectors(), args.json)
    if args.tie:
        print(json.dumps(ex.tie_check(), indent=2))
    if args.alternatives:
        _print(ex.alternatives(), args.json)
    if args.what_if:
        _print(ex.what_if([Lock.parse(t) for t in args.what_if]), args.json)
    if not args.json:
        print("\n" + ex.why_flight(ex.sub.decision_flights[0])["scope"])


if __name__ == "__main__":
    main()
