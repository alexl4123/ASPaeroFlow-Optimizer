# ASPaeroFlow-Optimizer

**ASPaeroFlow-Optimizer** is a research codebase for solving the **Strategic Air Traffic Flow and Capacity Management (ATFCM)** problem with a combination of **Answer Set Programming (ASP)** and heuristic optimization. It provides algorithms to jointly optimize **air traffic flow management measures** (ground/en-route delays and reroutes) together with **Dynamic Airspace Configuration (DAC)** decisions (sector capacity adjustments) under a unified model. The repository includes a novel **ASPaeroFlow** heuristic that iteratively resolves sector overloads using ASP sub-solvers, as well as exact and baseline solvers for comparison.

## Features

- **Simultaneous Optimization:** Jointly optimizes flow actions (delay, rerouting) and airspace reconfiguration (DAC)
- **Instance-Space Decomposition:** Scalable heuristic that solves local overloads with ASP
- **ASP-Based Core Solver:** Lexicographic soft objectives (safety, delay, stability, changes)
- **Open Data Compatibility:** Accepts input from generator based on OpenSky Network data
- **Baselines Included:** MILP solver, delay-only heuristic, exact ASP model for small instances
- **Benchmarking Tools:** Batch runner and CSV output for experiments

## Repository Structure

```
├── 01_ASPaeroFlow/              # Main heuristic solver
├── 02_ASP/                      # Exact ASP solver (small instances)
├── 04_MIP/                      # MILP solver baseline (Gurobi)
├── 06_benchmark_start_script/   # Scripts to benchmark across solvers
```

## Installation

Requirements:
- Python 3.9+
- Dependencies: `numpy`, `pandas`, `scipy`, `networkx`, `joblib`, `psutil`, `clingo`
- ASP solver: [Clingo](https://potassco.org/clingo/)
- (Optional) Gurobi for MIP solver (`gurobipy`)

Install with pip:

```bash
pip install numpy pandas scipy networkx joblib psutil clingo
```

Ensure `clingo` is available via the Python API or command line.

## Usage

Ensure you have an example in `example_instance` from ASPaeroFlow-DataGenerator.

### Run ASPaeroFlow Heuristic

```bash
python 01_ASPaeroFlow/main.py \
  --data-dir example_instance
  --encoding-path 01_ASPaeroFlow/encoding.lp \
  --wandb-enabled=false \
  --verbosity 1
```

Output shows overload resolution progress and final objective.

### Run Exact ASP Solver (Small Instances)

```bash
python 02_ASP/main.py \
  --data-dir example_instance
  --encoding-path 02_ASP/encoding.lp --verbosity 1
```

Prints delay, reroutes, and final schedule.

#### Exporting the ASP program

`02_ASP` can write out the logic program it builds, so that another group can run their own
solver against exactly this problem without reimplementing the CSV-to-ASP translation.

```bash
# 1. the translated instance facts only
python 02_ASP/main.py --data-dir example_instance \
  --export-instance-lp out/instance.lp --export-only

# 2. a SELF-CONTAINED program: encoding + arrival-delay metric fact + instance facts
python 02_ASP/main.py --data-dir example_instance \
  --export-program-lp out/program.lp --export-only

clingo out/program.lp            # runs directly, no further arguments needed
```

| option | effect |
|---|---|
| `--export-instance-lp FILE` | write the translated instance facts (one atom per line) |
| `--export-program-lp FILE` | write encoding + metric fact + facts, runnable as `clingo FILE` |
| `--export-only` | write the file(s) and exit without solving |

Without `--export-only` the file is written and solving continues as normal; neither option
changes the search.

The exported program depends on the options that shape the translation, so pass the same ones you
would use for solving — in particular `--timestep-granularity`, `--sector-capacity-factor`,
`--arrival-delay-metric` (`signed` | `floored` | `absolute`, default `signed`, which is the
definition used in the publications) and the regulation switches. `--export-program-lp` records all of them in
a comment header, so an exported file states the settings it was built under:

```prolog
% ASPaeroFlow instance, exported by 02_ASP/main.py
% encoding:              02_ASP/encoding.lp
% arrival-delay metric:  signed
% timestep granularity:  60
% max time:              1440
% sector capacity factor:6
% regulations: ground-delay=True rerouting=True dynamic-sectorization=False
% Self-contained: run with `clingo program.lp`.
```

Note that `sectors.csv::Capacity` is a **per-timestep** capacity, not per hour — see the dataset
README shipped with the instances.

#### Time horizon of the exact ASP solver

`02_ASP` solves inside a finite time domain: timesteps `0 .. max_time × T_gran` (`--max-time`,
in hours, default 24, i.e. the instance's day), extended to the last planned timestep if a filed
plan ends later. **Every flight position must lie inside this domain**: a delayed or rerouted
flight cannot land after it. (Before commit `cf90cea`, the no-rerouting and restricted-rerouting
variants did not enforce this; a flight could fly past the domain, where no capacity was checked,
so its load went uncounted.) What happens when the best solution still has overload depends on
the delay option:

| ground delay | horizon |
|---|---|
| full (`d`, `--regulation-ground-delay-active=2`) | re-solve loop: when a search ends with overload > 0, solve again with `max_time` one hour longer, up to a theoretical bound. A search stopped at the time limit (or at `--solve-deadline`) is not extended. `--fixed-horizon` switches the loop off. |
| bounded (`dp`) or none (`nd`) | **fixed** at `max_time`: no re-solve. |

**This differs from the formal model** (LPNMR 2026), whose time axis continues after the
instance's day with the last sector configuration and its capacity. With a fixed horizon, a flight
cannot be delayed so far that it would land after the day, so the optimum of a bounded-delay
variant can be worse than the formal model's. This is accepted: bounded delay is itself a
restriction of the model (at most a few timesteps per flight), and delay 0 always fits, so no
instance becomes unsatisfiable. The other solvers (ASPaeroFlow, MIP, the baselines) are not bound
to the day; their solutions are validated over the whole time axis.

### Run Baselines

- **MILP (flow-only):** `04_MIP/main.py` (requires Gurobi)

### Batch Benchmarking

`06_benchmark_start_script/start_benchmark_caller.py` runs every enabled solver over every
instance in a directory and writes performance CSVs per solver and instance.

For a whole benchmark campaign, `run_all_benchmarks.slurm` drives it as a single SLURM job array
— one array task per problem directory — indexed by a manifest that
`ASPaeroFlow-DataGenerator/expand_instances_for_benchmark.sh` produces:

```bash
# 1. materialise the capacity sweep into the layout the caller expects
cd ../ASPaeroFlow-DataGenerator
./expand_instances_for_benchmark.sh . ../ASPaeroFlow-Optimizer/05_instances --clean

# 2. launch everything (the expansion script prints the exact array range)
cd ../ASPaeroFlow-Optimizer/06_benchmark_start_script
sbatch --array=1-325%40 run_all_benchmarks.slurm

# 3. check progress at any time -- pure stdlib, no conda environment needed
./benchmark_progress.py                # overall
./benchmark_progress.py --detail       # per task
./benchmark_progress.py --failures     # what failed, by solver
```

Two options matter for large campaigns:

| option | why |
|---|---|
| `--results-format csv.gz` | the per-run result matrices are ~38 MB uncompressed and compress about 254x. Across a full grid that is the difference between ~2.9 TB and ~11 GB. |
| `--wandb-enabled False` | Weights & Biases is on by default. At tens of thousands of runs it means as many logged runs and network traffic from every compute node — and it is a hard failure if `wandb.key` is absent on the nodes. |

`run_all_benchmarks.slurm` sets both, and records them alongside the optimizer commit in
`output/<FOLDER>/run_provenance_<PROBLEM>.txt`.

## Citation


## License

MIT license with attribution (for details see license.md).

## Contributions

We welcome academic contributions and extensions. Please contact the authors for collaboration or open a pull request.

---

*For questions or collaborations, please reach out to the original authors.*
*This README was created with the help of generative AI.*

