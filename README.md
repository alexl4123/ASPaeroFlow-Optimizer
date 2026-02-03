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

### Run Baselines

- **MILP (flow-only):** `04_MIP/main.py` (requires Gurobi)

### Batch Benchmarking

Provided in `06_benchmark_start_script/start_benchmarks.py`
Generates performance CSVs per solver and instance.

## Citation


## License

MIT license with attribution (for details see license.md).

## Contributions

We welcome academic contributions and extensions. Please contact the authors for collaboration or open a pull request.

---

*For questions or collaborations, please reach out to the original authors.*
*This README was created with the help of generative AI.*

