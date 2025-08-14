#!/usr/bin/env bash

port=8086
sbatch -o ./log/noise_client_$port.log -c 5 ./freeze-omni/start_clients_single.sh "data/cosyvoice2-single-round-combine-easy" $port

port=8087
sbatch -o ./log/noise_client_$port.log -c 5 ./freeze-omni/start_clients_single.sh "data/cosyvoice2-single-round-combine-easy-noisy-gap-0dB" $port
