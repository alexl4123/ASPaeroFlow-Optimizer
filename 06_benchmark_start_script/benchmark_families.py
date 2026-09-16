#!/usr/bin/env python3
"""Which solver systems a problem runs, and why -- the rules run_all_benchmarks.slurm encodes.

run_all_benchmarks.slurm decides a task's system list in bash, from the manifest's
capacity_level column and from RUN_MIP. The per-unit path has to make the SAME decision in two
more places -- when the worklist is enumerated, and again when a unit is executed -- so the rule
lives here once and both ask for it.

The flag lists below are character-for-character the ones the bash script builds. That is the
point: a unit's command line differs from the monolithic one only by --only-instance,
--only-system and the output directory, so any disagreement about what a family is would show up
immediately as "no such enabled system" rather than as quietly different results.

THE THREE FAMILIES (from run_all_benchmarks.slurm, whose comments are the authority)

  small   capacity_level == "NONE" in the manifest. That family has no PCAP sweep because
          cap-enroute is already 1. It runs EVERYTHING: heuristics, baselines, MIP and all 27
          exact-ASP variants -> 39 systems per instance. The 27 exist so exact methods can be
          measured where they can actually close instances.

  large   any other capacity_level. Heuristics, baselines, MIP, and the two best-performing
          exact configurations -> 12 systems per instance:
              05_ASP_rp_dp_sp  =  ASP(r_p, d_p, s_p)   partial rerouting, partial delay, partial DAC
              05_ASP_rp_d_sp   =  ASP(r_p, d,   s_p)   partial rerouting, full delay,    partial DAC
          The other 25 would time out essentially every time on a 31,622-flight TG=60 instance,
          at 1800 seconds per run of predictable non-information.

  MIP only  RUN_MIP=only, whatever the capacity level. Exists because the Gurobi-licensed nodes
          (coppernode25-28) are in sunnycove while the campaign has to stay on broadwell.

RUN_MIP, unchanged from the bash:

  auto  probe the node for a usable licence; run MIP only if one is found (default)
  yes   force MIP on, alongside every other solver
  no    force MIP off -- the sweep across the whole partition
  only  run NOTHING BUT MIP

`auto` is the one mode that cannot be resolved when the worklist is built, because it is a
property of the NODE a task lands on and the worklist is built on the login node. So enumeration
treats auto like yes (the units exist) and the runner re-decides per task, recording a skip
rather than a failure where no licence is found -- which is exactly what auto means in the bash:
"MIP skipped, not failed". If that bothers you, the two-submission pattern the slurm comments
recommend avoids the question entirely: one worklist with --run-mip=no, one with --run-mip=only.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

RUN_MIP_MODES = ("auto", "yes", "no", "only")

#: The switches the MIP-only pass turns off: everything that is not 04_MIP.
_MIP_ONLY_OFF: Tuple[str, ...] = (
    "--experiment-asp-aero-flow=0", "--experiment-asp-aero-flow-no-convex=0",
    "--experiment-asp-aero-flow-nr-nd=0", "--experiment-asp-aero-flow-nr-d=0",
    "--experiment-asp-aero-flow-r-nd=0",
    "--experiment-casa=0", "--experiment-route-delay=0", "--experiment-route=0",
    "--experiment-delay=0",
    "--experiment-all-asp-variants=0",
    "--experiment-asp-rp-dp-sp=0", "--experiment-asp-rp-d-sp=0",
)

#: What the large family switches off. Only the first line does any work:
#: --experiment-all-asp-variants=0 drops all 27 exact-ASP variants in one go. The twelve
#: --experiment-asp-<r>-<d>-<s>=0 flags after it are INERT -- start_benchmark_caller.py parses
#: them and build_system_config() never reads them. They are reproduced here anyway, because the
#: aim is a command line identical to the monolithic one, and because dropping them from this
#: list while leaving them in the bash would make the two look like they disagreed.
_LARGE_OFF: Tuple[str, ...] = (
    "--experiment-all-asp-variants=0",
    "--experiment-asp-r-d-s=0", "--experiment-asp-r-d-ns=0",
    "--experiment-asp-r-nd-s=0", "--experiment-asp-nr-d-s=0",
    "--experiment-asp-r-nd-ns=0", "--experiment-asp-nr-nd-s=0",
    "--experiment-asp-nr-d-ns=0", "--experiment-asp-nr-nd-ns=0",
    "--experiment-asp-r-d-sp=0", "--experiment-asp-nr-d-sp=0",
    "--experiment-asp-r-nd-sp=0", "--experiment-asp-nr-nd-sp=0",
)

FAMILY_MIP_ONLY = "MIP only"
FAMILY_SMALL = "small (everything: heuristics + MIP + 27 exact-ASP variants)"
FAMILY_LARGE = "large (heuristics + MIP + ASP rp_dp_sp and rp_d_sp)"


def mip_enabled(run_mip: str, licence_found: bool = True) -> bool:
    """Whether 04_MIP runs. `licence_found` is consulted for auto and ignored otherwise."""
    if run_mip not in RUN_MIP_MODES:
        raise ValueError(f"RUN_MIP must be one of {RUN_MIP_MODES} (got: {run_mip})")
    if run_mip == "no":
        return False
    if run_mip in ("yes", "only"):
        return True
    return bool(licence_found)


def family_name(capacity_level: str, run_mip: str) -> str:
    if run_mip == "only":
        return FAMILY_MIP_ONLY
    return FAMILY_SMALL if capacity_level == "NONE" else FAMILY_LARGE


def experiment_flags(capacity_level: str, run_mip: str, licence_found: bool = True) -> List[str]:
    """The --experiment-* flags run_all_benchmarks.slurm would pass for this problem."""
    mip_flag = f"--experiment-mip={1 if mip_enabled(run_mip, licence_found) else 0}"
    if run_mip == "only":
        return [mip_flag, *_MIP_ONLY_OFF]
    if capacity_level == "NONE":
        return [mip_flag]
    return [mip_flag, *_LARGE_OFF]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capacity-level", required=True,
                    help="the manifest's capacity_level; NONE marks the small-scaling family")
    ap.add_argument("--run-mip", default="auto", choices=list(RUN_MIP_MODES))
    ap.add_argument("--mip-licence", default="yes", choices=["yes", "no"],
                    help="for --run-mip=auto: whether this node has a usable Gurobi licence")
    ap.add_argument("--print", dest="what", default="flags", choices=["flags", "family"])
    a = ap.parse_args()
    if a.what == "family":
        print(family_name(a.capacity_level, a.run_mip))
    else:
        for flag in experiment_flags(a.capacity_level, a.run_mip, a.mip_licence == "yes"):
            print(flag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
