# ASPaeroFlow-Optimizer

**ASPaeroFlow-Optimizer** is a research codebase for solving the **Strategic Air Traffic Flow and Capacity Management (ATFCM)** problem with a combination of **Answer Set Programming (ASP)** and heuristic optimization. It provides algorithms to jointly optimize **air traffic flow management measures** (ground/en-route delays and reroutes) together with **Dynamic Airspace Configuration (DAC)** decisions (sector capacity adjustments) under a unified model. The repository includes a novel **ASPaeroFlow** heuristic that iteratively resolves sector overloads using ASP sub-solvers, as well as exact and baseline solvers for comparison.

## Features

- **Simultaneous Optimization:** Jointly optimizes flow actions (delay, rerouting) and airspace reconfiguration (DAC)
- **Instance-Space Decomposition:** Scalable heuristic that solves local overloads with ASP
- **ASP-Based Core Solver:** Hard safety constraints, lexicographic soft objectives (delay, stability, changes)
- **Open Data Compatibility:** Accepts input from generator based on OpenSky Network data
- **Baselines Included:** MILP solver, delay-only heuristic, exact ASP model for small instances
- **Benchmarking Tools:** Batch runner and CSV output for experiments

## Repository Structure

```
├── 00_instance_generator/        # Grid-based instance generator
├── 01_ASPaeroFlow/              # Main heuristic solver
├── 02_ASP/                      # Exact ASP solver (small instances)
├── 03_Delay/                    # Delay-only baseline solver
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

### Generate an Example Instance

```bash
python 00_instance_generator/generator_v2.py --out example_instance --flights 5
```

Creates a folder like `example_instance/0000005/` with:
- `edges.csv`, `airports.csv`, `capacity.csv`, `instance.csv`

### Run ASPaeroFlow Heuristic

```bash
python 01_ASPaeroFlow/main.py \
  --path-graph example_instance/0000005/edges.csv \
  --path-capacity example_instance/0000005/capacity.csv \
  --path-instance example_instance/0000005/instance.csv \
  --airport-vertices-path example_instance/0000005/airports.csv \
  --encoding-path 01_ASPaeroFlow/encoding.lp \
  --verbosity 1
```

Output shows overload resolution progress and final objective.

### Run Exact ASP Solver (Small Instances)

```bash
python 02_ASP/main.py \
  --path-graph ... --path-capacity ... --path-instance ... --airport-vertices-path ... \
  --encoding-path 02_ASP/encoding.lp --verbosity 1
```

Prints delay, reroutes, and final schedule.

### Run Baselines

- **Delay only:** `03_Delay/main.py`
- **MILP (flow-only):** `04_MIP/main.py` (requires Gurobi)

### Batch Benchmarking

```bash
python 06_benchmark_start_script/start_benchmarks.py \
  path/to/instances --time-limit 1800 --memory-limit 5000
```

Generates performance CSVs per solver and instance.

## Citation

If you use this code, please cite:

> Alexander Beiser, Nysret Musliu, Stefan Woltran (2026).  
> *Complexity Analysis and ASP Modeling for Strategic ATFCM*.  
> Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning (KR 2026).

> Alexander Beiser et al. (2026).  
> *ASPaeroFlow: Instance-Space Decomposition and Answer Set Programming for Simultaneous Flow and Airspace Optimization in Strategic ATFCM*.  
> Second US-Europe Air Transportation Research and Development Symposium (ATRDS 2026).

## License

MIT license with attribution (for details see license.md).

## Contributions

We welcome academic contributions and extensions. Please contact the authors for collaboration or open a pull request.

---

*For questions or collaborations, please reach out to the original authors.*
*This README was created with the help of generative AI.*

