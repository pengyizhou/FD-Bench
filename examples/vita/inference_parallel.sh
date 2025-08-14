#!/usr/bin/env bash

port=8081
sbatch -o ./log/client_$port.log -c 5 ./vita/start_clients_single.sh "data/chattts-single-round-combine-easy data/chattts-single-round-combine-med" $port

port=8082
sbatch -o ./log/client_$port.log -c 5 ./vita/start_clients_single.sh "data/chattts-single-round-combine-hard data/cosyvoice2-single-round-combine-easy" $port

port=8083
sbatch -o ./log/client_$port.log -c 5 ./vita/start_clients_single.sh "data/cosyvoice2-single-round-combine-med data/cosyvoice2-single-round-combine-hard" $port

port=8084
sbatch -o ./log/client_$port.log -c 5 ./vita/start_clients_single.sh "data/f5tts-combine-easy data/f5tts-combine-med data/f5tts-combine-hard" $port
