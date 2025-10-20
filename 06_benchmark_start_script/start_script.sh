#!/bin/bash
TIME_GRANULARITY=60
SEED=11904657
FOLDER=20250913
CPU_RANGE_START=22
CPU_RANGE_END=23
GRID_WIDTH=019
GRID_HEIGHT=019
#

INSTANCE=${GRID_WIDTH}_${GRID_HEIGHT}_TG${TIME_GRANULARITY}_SEED${SEED}
PROBLEM=${FOLDER}/instances_${INSTANCE}

mkdir -p output/${FOLDER}
mkdir -p logs/${FOLDER}

nohup taskset --cpu-list ${CPU_RANGE_START}-${CPU_RANGE_END} ./start_benchmark_caller.py ../05_instances/${PROBLEM} --output-dir=output/${FOLDER}/output_${INSTANCE}/ --timestep-granularity=${TIME_GRANULARITY} &> logs/${PROBLEM}.log &

