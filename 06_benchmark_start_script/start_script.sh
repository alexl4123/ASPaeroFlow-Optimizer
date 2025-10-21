#!/bin/bash
TIME_GRANULARITY=60
SEED=11904657
FOLDER=20251021
CPU_RANGE_START=22
CPU_RANGE_END=23
GRID_WIDTH=019
GRID_HEIGHT=019
#

INSTANCE=0000003_MINI
PROBLEM=MINI

mkdir -p output/${FOLDER}
mkdir -p logs/${FOLDER}

./start_benchmark_caller.py ../05_instances/${PROBLEM} --output-root=output --output-dir=${FOLDER}/output_${INSTANCE}/ --timestep-granularity=${TIME_GRANULARITY} 

