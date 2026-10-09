#!/usr/bin/env python3
"""Merge the step files of a Pareto-front campaign folder and certify every front.

    ./merge_pareto_steps.py --folder 20260928_PARETO
    ./merge_pareto_steps.py --folder 20260928_PARETO_PILOT        # also checks the laptop fronts

Reads output/<FOLDER>/steps/<label>/<round>_<step>.jsonl (written by run_pareto_steps.slurm, or
imported from usc27 by build_pareto_manifest.py) and writes, into output/<FOLDER>/:

    fronts/front_<label>.csv    one line per step, in exactly the five-column format of
                                bsc_student_stuff/20260923/pareto/plot_fronts.py:
                                    label,K|none,status,time,c1 c2 c3 c4 c5 c6
                                costs in sectors-first order (overload, sectors, delay, sector_diff,
                                reroute, reconfig); status OPTIMUM FOUND, SATISFIABLE, UNKNOWN,
                                UNSATISFIABLE, or KILLED for a run without a result line. The proven
                                delay-first end is written as a proven row at K = D_min.
    bounds/bounds_<label>.csv   per step: K, status, the incumbent and the lower-bound vector, by
                                level name (a sixth column would make plot_fronts.py skip the row)
    pareto_steps.csv            one row per step: outcome, times, program size, peak RSS, host
    pareto_fronts.csv           one row per front: its class (exact / partial / endpoints / none),
                                o*, the points (proven marked *), the certified stretches and share,
                                the lower-bound staircase, and the solving cost
    pareto_summary.md           class counts per (group, flights) and per variant, and, for the
                                pilot folder, the comparison with the laptop fronts

Certification is pareto_lib.certify() (draft Section 3.4, lower bounds included); read its module
docstring for the rules. Nothing here needs matplotlib: plot_pareto_fronts.py draws the figure.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pareto_lib as pl  # noqa: E402

LEVEL_NAMES = {"o": "overload", "s": "sectors", "d": "delay", "sd": "sector_diff",
               "rr": "reroute", "rc": "reconfig"}


def _fmt(value):
    if value is None:
        return ""
    if value == pl.INF:
        return "inf"
    return str(value)


def staircase_text(points) -> str:
    return " ".join(f"({_fmt(k)},{_fmt(v)})" for k, v in points)


def step_rows(label: str, steps: List[pl.Step]) -> List[Dict]:
    info = pl.parse_label(label)
    rows = []
    for s in steps:
        final = s.final or {}
        row = {"label": label, "region": info["region"], "flights": info["flights"],
               "seed": info["seed"], "variant": info["variant"], "metric": info["metric"],
               "round": s.round, "step": s.key, "order": s.order, "bound": _fmt(s.bound),
               "status": s.status, "exhausted": s.exhausted, "stopped_at_deadline": s.stopped,
               "has_model": s.has_model, "models": len(s.models)}
        for level in pl.LEVELS:
            row[LEVEL_NAMES[level]] = _fmt(s.incumbent[level]) if s.incumbent else ""
        for level in pl.LEVELS:
            row["lb_" + LEVEL_NAMES[level]] = _fmt(s.lower[level]) if s.lower else ""
        row.update({
            "grounding_s": final.get("GROUNDING-TIME"),
            "search_started_s": final.get("SOLVER-SEARCH-STARTED-S"),
            "search_ended_s": final.get("SOLVER-SEARCH-ENDED-S"),
            "wall_s": s.prov.get("wall_s", ""), "exit_code": s.prov.get("exit_code", ""),
            "outcome": s.prov.get("outcome", ""),
            "peak_rss_mb": final.get("PEAK-RSS-MB"),
            "program_atoms": final.get("PROGRAM-ATOMS"), "program_rules": final.get("PROGRAM-RULES"),
            "clingo": final.get("CLINGO-VERSION"), "host": s.prov.get("host", ""),
            "commit": s.prov.get("optimizer_commit", ""),
            "time_limit_s": s.prov.get("time_limit_s", ""),
            "imported_from": s.prov.get("imported_from", ""),
            "problems": "; ".join(s.problems),
            "file": str(s.path),
        })
        rows.append(row)
    return rows


def front_row(label: str, front: pl.Front, steps: List[pl.Step]) -> Dict:
    info = pl.parse_label(label)
    corners = front.corners
    points = " ".join(f"({d},{s}){'*' if ok else ''}"
                      for (d, s), ok in zip(corners, front.point_proven or [False] * len(corners)))
    share = front.certified_share
    walls = [s.wall_s for s in steps if s.wall_s is not None]
    rss = [s.final.get("PEAK-RSS-MB") for s in steps if s.final and s.final.get("PEAK-RSS-MB")]
    rules = [s.final.get("PROGRAM-RULES") for s in steps if s.final and s.final.get("PROGRAM-RULES")]
    return {
        "label": label, "region": info["region"], "flights": info["flights"], "seed": info["seed"],
        "variant": info["variant"], "group": pl.group_of(info["variant"]), "metric": info["metric"],
        "class": front.cls, "o_star": _fmt(front.o_star), "o_known": front.o_known,
        "points": len(corners), "proven_points": sum(front.point_proven),
        "stretches": len(front.stretch_ok), "certified_stretches": sum(front.stretch_ok),
        "left_end": front.left_ok, "right_end": front.right_ok,
        "certified_share": "" if share is None else round(share, 4),
        "df_status": front.df.status if front.df else "MISSING",
        "sf_status": front.sf.status if front.sf else "MISSING",
        "steps": len(steps),
        "steps_stopped": sum(1 for s in steps if s.stopped),
        "steps_no_model": sum(1 for s in steps if s.final is not None and not s.has_model
                              and not s.exhausted),
        "steps_unsat": sum(1 for s in steps if s.status == "UNSATISFIABLE"),
        "steps_killed": sum(1 for s in steps if s.final is None),
        "wall_s_total": round(sum(walls), 1),
        "peak_rss_mb_max": max(rss) if rss else "",
        "program_rules": max(rules) if rules else "",
        "front": points,
        "lower_staircase": staircase_text(front.lower_staircase()),
        "notes": "; ".join(front.notes),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pilot_check(fronts: Dict[str, pl.Front]) -> List[str]:
    """The laptop fronts (all proven there, BSc encoding = this one under `floored`)."""
    out = ["", "## Pilot: laptop fronts (floored; must match exactly)", "",
           "| front | laptop | here | class | match |", "|---|---|---|---|---|"]
    for label, expected in pl.PILOT_EXPECTED.items():
        front = fronts.get(label)
        if front is None:
            out.append(f"| {label} | {' '.join(map(str, expected))} | (no steps) | | NO |")
            continue
        here = [c for c, ok in zip(front.corners, front.point_proven) if ok]
        match = here == expected and front.cls == "exact"
        out.append(f"| {label} | {' '.join(map(str, expected))} | "
                   f"{' '.join(map(str, front.corners))} | {front.cls} | "
                   f"{'yes' if match else 'NO'} |")
    out += ["", "## Pilot: signed against floored", "",
            "| front | floored | signed |", "|---|---|---|"]
    for label, front in sorted(fronts.items()):
        if pl.parse_label(label)["metric"] != "floored":
            continue
        twin = fronts.get(label[:-len("__floored")])
        out.append(f"| {label[:-len('__floored')]} | {' '.join(map(str, front.corners))} "
                   f"({front.cls}) | "
                   f"{' '.join(map(str, twin.corners)) + f' ({twin.cls})' if twin else '(none)'} |")
    return out


def rounds_md(step_table: List[Dict]) -> List[str]:
    """Status counts per round, and the P2 steps one by one (when did each close, if at all)."""
    statuses = ("OPTIMUM FOUND", "SATISFIABLE", "UNKNOWN", "UNSATISFIABLE", "KILLED")
    out = ["", "Steps per round (status of the result line; KILLED = none):", "",
           "| round | steps | " + " | ".join(statuses) + " |",
           "|---|---|" + "---|" * len(statuses)]
    by_round: Dict = defaultdict(lambda: defaultdict(int))
    for row in step_table:
        by_round[row["round"]][row["status"]] += 1
    for name, counts in sorted(by_round.items()):
        out.append(f"| {name} | {sum(counts.values())} | " +
                   " | ".join(str(counts[st]) for st in statuses) + " |")
    p2 = [row for row in step_table if row["round"] == "P2"]
    if p2:
        out += ["", "P2 (known-hard steps, one run each): the search closed at SOLVER-SEARCH-ENDED-S "
                    "when the status is OPTIMUM FOUND.", "",
                "| front | K | status | search ended (s) | incumbent sectors | lower bound sectors |",
                "|---|---|---|---|---|---|"]
        for row in sorted(p2, key=lambda r: (r["label"], int(r["bound"]))):
            out.append(f"| {row['label']} | {row['bound']} | {row['status']} | "
                       f"{row['search_ended_s']} | {row['sectors']} | {row['lb_sectors']} |")
    return out


def summary_md(folder: str, rows: List[Dict], fronts: Dict[str, pl.Front], pilot: bool,
               step_table: List[Dict]) -> str:
    classes = ("exact", "partial", "endpoints", "none")
    lines = [f"# Pareto fronts: {folder}", "",
             f"{len(rows)} fronts. Class per (group, flights):", "",
             "| group | flights | fronts | " + " | ".join(classes) + " |",
             "|---|---|---|" + "---|" * len(classes)]
    cells: Dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cells[(row["group"], row["flights"])][row["class"]] += 1
    for (group, flights), counts in sorted(cells.items()):
        lines.append(f"| {group} | {flights} | {sum(counts.values())} | " +
                     " | ".join(str(counts[c]) for c in classes) + " |")
    lines += ["", "Class per variant:", "",
              "| variant | fronts | " + " | ".join(classes) + " |",
              "|---|---|" + "---|" * len(classes)]
    by_variant: Dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_variant[row["variant"]][row["class"]] += 1
    for variant in pl.VARIANTS:
        if variant in by_variant:
            counts = by_variant[variant]
            lines.append(f"| {variant} | {sum(counts.values())} | " +
                         " | ".join(str(counts[c]) for c in classes) + " |")
    flagged = [r for r in rows if r["o_known"] and r["o_star"] not in ("", "0")]
    lines += ["", f"Fronts with least overload > 0 at the fixed horizon (V2 would have extended "
                  f"it): {len(flagged)}"]
    noted = [r for r in rows if "step ignored" in r["notes"]]
    if noted:
        lines += [f"Fronts with ignored steps (see notes in pareto_fronts.csv): {len(noted)}"]
    lines += rounds_md(step_table)
    if pilot:
        lines += pilot_check(fronts)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()

    base = args.output_root / args.folder
    all_steps = pl.load_all(args.output_root, args.folder)
    if not all_steps:
        print(f"[merge] no step files under {base / 'steps'}")
        return 1
    step_table, front_table, fronts = [], [], {}
    for label, steps in all_steps.items():
        try:
            pl.parse_label(label)
        except ValueError:
            print(f"[warn] {label}: not a front label, skipped", file=sys.stderr)
            continue
        front = pl.certify(label, steps)
        fronts[label] = front
        (base / "fronts").mkdir(parents=True, exist_ok=True)
        (base / "fronts" / f"front_{label}.csv").write_text(
            "\n".join(pl.front_csv_rows(label, steps)) + "\n")
        bound_rows = [{"step": s.key, "round": s.round, "order": s.order, "K": _fmt(s.bound),
                       "status": s.status, "exhausted": s.exhausted,
                       **{LEVEL_NAMES[lv]: _fmt(s.incumbent[lv]) if s.incumbent else ""
                          for lv in pl.LEVELS},
                       **{"lb_" + LEVEL_NAMES[lv]: _fmt(s.lower[lv]) if s.lower else ""
                          for lv in pl.LEVELS}}
                      for s in sorted(steps, key=pl._step_sort_key)]
        write_csv(base / "bounds" / f"bounds_{label}.csv", bound_rows)
        step_table += step_rows(label, steps)
        front_table.append(front_row(label, front, steps))

    write_csv(base / "pareto_steps.csv", step_table)
    write_csv(base / "pareto_fronts.csv", front_table)
    pilot = any(label in pl.PILOT_EXPECTED for label in fronts)
    (base / "pareto_summary.md").write_text(summary_md(args.folder, front_table, fronts, pilot,
                                                       step_table))

    counts = defaultdict(int)
    for row in front_table:
        counts[row["class"]] += 1
    print(f"[merge] {len(step_table)} steps, {len(front_table)} fronts: " +
          ", ".join(f"{k} {counts[k]}" for k in ("exact", "partial", "endpoints", "none")))
    print(f"[merge] wrote {base}/fronts/, bounds/, pareto_steps.csv, pareto_fronts.csv, "
          f"pareto_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
