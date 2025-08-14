#!/usr/bin/env bash

export PYTHONPATH=./:$PYTHONPATH
port=$1
python3 bin/server.py \
    --ip 127.0.0.1 \
    --port $port \
    --max_users 50 \
    --llm_exec_nums 1 \
    --timeout 80 \
    --model_path ./checkpoints \
    --llm_path ./Qwen2-7B-Instruct \
    --top_p 0.8 \
    --top_k 20 \
    --temperature 0.8