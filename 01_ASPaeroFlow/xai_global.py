#!/usr/bin/env python3
"""A whole run summarised from its local explanations (trace written with main.py --xai-trace-dir).

    python 01_ASPaeroFlow/xai_global.py TRACE [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.aspaeroflow.xai.global_summary import summarize  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    s = summarize(TraceReader(a.trace))
    if a.json:
        a.json.write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"{s['accepted_steps']} accepted steps, {s['rejected_steps']} rejected, {s['tied_steps']} tied")
    print("actions:", ", ".join(f"{k} {v}" for k, v in sorted(s["actions"].items(), key=lambda x: -x[1])))
    for lever, v in s["levers"].items():
        if v["used"]:
            print(f"{lever}: used in {v['used']} steps, needed in {v['needed']}"
                  + (" (without it: " + ", ".join(f"{k} worse {n}x" for k, n in v["decided_by"].items()) + ")" if v["needed"] else ""))
    print(f"{s['flights_moved']} flights moved, {s['flights_moved_more_than_once']} of them more than once")
    print("arrival shift caused by hotspot (sector, time):")
    for h in s["delay_by_hotspot"][:8]:
        print(f"  sector {h['sector']:>3} at t={h['time']:>3}: {h['arrival_shift']:+d} steps")
    for st in s["steps"]:
        lev = "; ".join(f"{k}: " + (f"needed ({v['level']} +{v['margin']})" if v["needed"] else "replaceable")
                        for k, v in st["levers"].items())
        print(f"  step {st['iteration']:>3} {st['action']:<22} {'tie' if st['tie'] else '   '} overload -{st['overload_removed']:<3} {lev}")


if __name__ == "__main__":
    main()
