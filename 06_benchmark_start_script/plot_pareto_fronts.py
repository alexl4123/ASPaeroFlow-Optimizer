#!/usr/bin/env python3
"""Draw the certified (delay, sectors) fronts of a Pareto campaign folder, with hypervolumes.

    python plot_pareto_fronts.py --folder 20260928_PARETO_PILOT [--labels REGEX] [--out fronts.png]

An extension of bsc_student_stuff/20260923/pareto/plot_fronts.py (which stays as it is, for the BSc
figure) to the campaign's step files and to the certification with lower bounds of pareto_lib:

  * one panel per instance (<region>_<flights>_<seed>, plus the metric when it is not `signed`),
    one staircase per variant through the corners of the feasible-point staircase U;
  * a stretch is solid where it is certified (L = U on it), dashed where it is not; a point is
    filled where it is proven to lie on the front, hollow where not;
  * where L < U the band between the two staircases is shaded: the true front lies inside it;
  * the right end is pinned only by an unbounded sectors-first step (proven, or with a lower bound
    equal to the last point), as in plot_fronts.py.

Hypervolume, as in plot_fronts.py: 2-D, minimisation, in the space normalised by the ideal and
nadir of the union of the variants' corners on the instance, reference point (1.1, 1.1). Per front
it writes the interval [HV of U, HV of the lower staircase L on the same delay range] to
<folder>/pareto_hv.csv; the two agree exactly when the front is exact. Needs matplotlib (the
laptop's plot environment); merge_pareto_steps.py does not.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pareto_lib as pl  # noqa: E402

REF = 1.1
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# Colour follows the variant, never its position in a panel. Slots 1-3 of the reference
# categorical palette (validated all-pairs) for the three variants most often compared; the rest
# take the later slots, and the group-B check pair shares full rerouting's hue with its own marker.
STYLE = {
    "rp_d_sp": ("#2a78d6", "o"), "r_d_sp": ("#eb6834", "o"), "rp_dp_sp": ("#1baf7a", "o"),
    "rp_nd_sp": ("#eda100", "D"), "nr_d_sp": ("#e87ba4", "s"), "nr_dp_sp": ("#008300", "s"),
    "rp_d_s": ("#4a3aa7", "^"), "rp_dp_s": ("#e34948", "^"),
    "r_dp_sp": ("#eb6834", "s"), "r_nd_sp": ("#eb6834", "D"),
}


def instance_of(label: str) -> str:
    info = pl.parse_label(label)
    key = f"{info['region']}, {info['flights']} flights, seed {info['seed']}"
    return key if info["metric"] == pl.CAMPAIGN_METRIC else f"{key} ({info['metric']})"


def hypervolume(points, ideal, nadir) -> float:
    span = [max(nadir[i] - ideal[i], 1) for i in range(2)]
    z = sorted(((d - ideal[0]) / span[0], (s - ideal[1]) / span[1]) for d, s in points)
    hv = 0.0
    for i, (x, y) in enumerate(z):
        x_next = z[i + 1][0] if i + 1 < len(z) else REF
        hv += max(0.0, x_next - x) * max(0.0, REF - y)
    return hv


def lower_points(front: pl.Front):
    """Corners of the lower staircase L on [first corner, last corner] (finite values only)."""
    pts, best = [], pl.INF
    lo, hi = front.corners[0][0], front.corners[-1][0]
    for k in range(lo, hi + 1):
        value = front.L(k)
        if value < best:
            pts.append((k, value))
            best = value
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--labels", default=None, help="Only fronts whose label matches this regex")
    parser.add_argument("--out", type=Path, default=None,
                        help="Figure path (default <folder>/pareto_fronts.png, and .pdf)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    base = args.output_root / args.folder
    fronts = {}
    for label, steps in pl.load_all(args.output_root, args.folder).items():
        if args.labels and not re.search(args.labels, label):
            continue
        front = pl.certify(label, steps)
        if front.corners and front.o_known:
            fronts[label] = front
    if not fronts:
        print("[plot] nothing to draw")
        return 1

    panels = defaultdict(list)
    for label in sorted(fronts):
        panels[instance_of(label)].append(label)

    hv_rows = []
    for panel, labels in panels.items():
        union = [c for label in labels for c in fronts[label].corners]
        ideal = (min(d for d, _ in union), min(s for _, s in union))
        nadir = (max(d for d, _ in union), max(s for _, s in union))
        for label in labels:
            front = fronts[label]
            hv_u = hypervolume(front.corners, ideal, nadir)
            hv_l = hypervolume(lower_points(front), ideal, nadir) if front.corners else None
            hv_rows.append({"instance": panel, "label": label,
                            "variant": pl.parse_label(label)["variant"], "class": front.cls,
                            "ideal": f"{ideal}", "nadir": f"{nadir}",
                            "hv_feasible": round(hv_u, 4),
                            "hv_bound": round(hv_l, 4) if hv_l is not None else "",
                            "front": " ".join(map(str, front.corners))})
    with (base / "pareto_hv.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(hv_rows[0]))
        writer.writeheader()
        writer.writerows(hv_rows)

    plt.rcParams.update({"font.size": 8})
    ncol = 2 if len(panels) > 1 else 1
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.6 * nrow), squeeze=False,
                             facecolor=SURF)
    seen_variants, any_gap, any_unproven, any_band = [], False, False, False
    for ax, (panel, labels) in zip(axes.flat, panels.items()):
        ax.set_facecolor(SURF)
        classes = []
        for k, label in enumerate(labels):
            front = fronts[label]
            variant = pl.parse_label(label)["variant"]
            color, marker = STYLE.get(variant, (INK2, "o"))
            if variant not in seen_variants:
                seen_variants.append(variant)
            classes.append(f"{variant} {front.cls}")
            xs = [d for d, _ in front.corners]
            ys = [s for _, s in front.corners]
            lw, ms = (3.0, 8) if k == 0 else (1.4, 5)
            # band between the feasible staircase U and the lower staircase L, where they differ
            lo, hi = xs[0], xs[-1]
            ks = list(range(lo, hi + 1))
            upper = [front.U(q) for q in ks]
            lower = [min(front.L(q), front.U(q)) for q in ks]
            if any(l < u for l, u in zip(lower, upper)):
                ax.fill_between(ks, lower, upper, step="post", color=color, alpha=0.14, lw=0,
                                zorder=1)
                any_band = True
            for i in range(len(xs) - 1):
                ok = front.stretch_ok[i]
                ax.plot([xs[i], xs[i + 1], xs[i + 1]], [ys[i], ys[i], ys[i + 1]], color=color,
                        lw=lw if ok else max(lw * 0.5, 1.0), ls="-" if ok else (0, (2, 2)),
                        zorder=2 + k, solid_joinstyle="round")
                any_gap |= not ok
            for (x, y), proven in zip(front.corners, front.point_proven):
                ax.plot(x, y, marker, ms=ms, mfc=color if proven else SURF, mec=color,
                        mew=1.2, zorder=3 + k)
                any_unproven |= not proven
            if not front.right_ok:     # the right end is not pinned: the front may continue
                ax.annotate("", xy=(xs[-1] + 0.6, ys[-1]), xytext=(xs[-1], ys[-1]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.0), zorder=2)
        ax.set_title(panel, fontsize=7.5, color=INK, loc="left")
        # the lower-left corner (little delay, few sectors) is empty for every minimisation front
        ax.text(0.02, 0.03, "\n".join(classes), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.5, color=INK2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=7)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("total arrival delay (time steps)", color=INK2, fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("active sectors, summed over time", color=INK2, fontsize=7.5)

    handles = [plt.Line2D([], [], color=STYLE.get(v, (INK2, "o"))[0], lw=1.6,
                          marker=STYLE.get(v, (INK2, "o"))[1], ms=5, label=v)
               for v in seen_variants]
    if any_unproven:
        handles.append(plt.Line2D([], [], color=MUTED, lw=0, marker="o", ms=5, mfc=SURF, mew=1.2,
                                  label="point not proven"))
    if any_gap:
        handles.append(plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (2, 2)),
                                  label="stretch not certified"))
    if any_band:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25,
                                     label="between feasible and lower bound"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.06 * (1 + (len(handles) - 1) // 4) / nrow))
    out = args.out or (base / "pareto_fronts.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURF)
    print(f"[plot] {len(fronts)} fronts in {len(panels)} panels -> {out} (+ .pdf), "
          f"{base / 'pareto_hv.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
