#!/usr/bin/env bash 

ports=`seq 8081 8084`

for port in $ports; do
    sbatch -o ./log/server_$port.log --gres=gpu:2 -c 24 \
        ./server.sh $port
done