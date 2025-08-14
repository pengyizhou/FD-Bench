#!/usr/bin/env bash

port=$1

python -m web_demo.server --model_path demo_VITA_ckpt --ip 0.0.0.0 --port $port
