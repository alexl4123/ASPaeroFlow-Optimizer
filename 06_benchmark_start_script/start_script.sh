#!/bin/bash
INSTANCE=019_019_TG1_SEED11904657
FOLDER=20250737
PROBLEM=${FOLDER}/instances_${INSTANCE}

mkdir -p output/${FOLDER}
mkdir -p logs/${FOLDER}

nohup taskset --cpu-list 28-29 ./start_benchmark_caller.py ../05_instances/${PROBLEM} --output-dir=output/${FOLDER}/output_${INSTANCE}/ &> logs/${PROBLEM}.log &

