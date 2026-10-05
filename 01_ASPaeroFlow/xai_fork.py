#!/usr/bin/env python3
"""Fork-and-continue experiment: replace step k's answer by the best other answer, run on, compare.

    python 01_ASPaeroFlow/xai_fork.py --data-dir INSTANCE --work-dir DIR [--option max_number_sectors=100000 ...]

Prints the base run's final objectives and, for every step, the final objectives when that step
took the best answer other than its own (marked "tie" when that answer was equally good).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.aspaeroflow.xai.fork import experiment  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="main.py long option without dashes, e.g. max_number_sectors=100000")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    options = dict(o.split("=", 1) for o in a.option)
    result = experiment(a.data_dir, a.work_dir, options)
    if a.json:
        a.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    base = result["base"]["final"]
    print("base:", " ".join(f"{k}={v}" for k, v in base.items()))
    for f in result["forks"]:
        fin = f["final"]
        kind = "tie " if f.get("tie") else ("next" if f.get("forked") else "none")
        diffs = " ".join(f"{k}={fin[k]}({fin[k] - base[k]:+d})" for k in ("ARRIVAL-DELAY", "SECTOR-NUMBER", "REROUTE")
                         if fin.get(k) is not None and base.get(k) is not None)
        print(f"step {f['fork_at']:>3} {kind} overload={fin['OVERLOAD']} {diffs} iterations={fin['iterations']}")


if __name__ == "__main__":
    main()
