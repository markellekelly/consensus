#!/bin/bash

start_points=(0 250 500)
dataset="imagenet"
suffix=""
shuffle_num=("" 1 2 3)

for shuf in "${shuffle_num[@]}"; do
	for sp in "${start_points[@]}"; do
		nohup python3 run_experiments.py "$dataset" "$sp" "$suffix" "$shuf" &> "out/${dataset}${sp}s${shuf}.out" &
	done
done
