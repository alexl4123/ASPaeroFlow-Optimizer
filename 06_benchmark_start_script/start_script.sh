#!/bin/bash
PROBLEM=instances_005_005

nohup taskset --cpu-list 2-3 ./start_benchmark_caller.py &> logs/${PROBLEM}.log &

