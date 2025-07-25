#!/usr/bin/env bash

data_folders="data/VITA-1.5/subjective_metrics data/Moshi-output/subjective_metrics data/Freeze-omni-output/subjective_metrics"

for data in $data_folders; do
    find $data/ -name "conversation_rounds.txt" | while read line; do
        sbatch -o $line.log --gres=gpu:1 -c 24 ./Llama3/CPPL.py --input_file $line
    done
done