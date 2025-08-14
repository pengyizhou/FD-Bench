#!/usr/bin/env bash 

ports=`seq 8086 8087`

for port in $ports; do
    sbatch -o ./log/server_$port.log --gres=gpu:1 -c 12 ./server.sh $port
done