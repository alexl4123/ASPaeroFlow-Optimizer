#!/usr/bin/env python3
"""Exposure of a saved ASPaeroFlow plan to capacity loss (prototype, fixed plan, no re-optimisation).

    python 01_ASPaeroFlow/main.py --data-dir=INSTANCE --save-results=true --results-root=OUT
    python 01_ASPaeroFlow/xai_robustness.py --results OUT/<instance name> --instance INSTANCE \\
        --model storms --scenarios 300 [--json report.json]

Models: storms (moving cells; --cells --radius --loss --life) or independent (--p --loss --window).
See src/aspaeroflow/xai/robustness.py for the measures.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.aspaeroflow.xai.robustness import Plan, analyse, coordinates_of, save  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--instance", type=Path, required=True)
    p.add_argument("--model", choices=["storms", "independent"], default="storms")
    p.add_argument("--scenarios", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cells", type=int, default=2)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--life", type=int, default=6)
    p.add_argument("--p", type=float, default=0.1)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--loss", type=float, default=0.75)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)

    plan = Plan.load(a.results, a.instance)
    if a.model == "storms":
        kw = dict(cells=a.cells, radius=a.radius, loss=a.loss, life=a.life)
        result = analyse(plan, a.scenarios, "storms", a.seed, coords=coordinates_of(a.instance), **kw)
    else:
        kw = dict(p=a.p, loss=a.loss, window=a.window)
        result = analyse(plan, a.scenarios, "independent", a.seed, **kw)
    if a.json:
        save(result, a.json)
    n, o = result["nominal"], result["overload"]
    print(f"nominal overload {n['overload']}; {n['tight_cells']} of {n['occupied_cells']} occupied sector-time cells have no slack")
    print(f"{a.scenarios} {a.model} scenarios: overload in {o['p_any']:.0%} of them, mean {o['mean']:.2f}, "
          f"mean of the worst 10% {o['cvar90']:.2f}, max {o['max']:.0f}")
    print("flights carrying most overload (expected share, probability of flying through reduced capacity):")
    for f in result["flights"]:
        print(f"  flight {f['flight']:>4}  {f['expected_overload_share']:.3f}  {f['p_exposed']:.2f}")
    print("sectors overloaded most often (probability, expected overload):")
    for s in result["sectors"]:
        print(f"  sector {s['sector']:>4}  {s['p_overloaded']:.3f}  {s['expected_overload']:.3f}")


if __name__ == "__main__":
    main()
