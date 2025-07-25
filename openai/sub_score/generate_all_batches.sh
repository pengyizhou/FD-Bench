#!/usr/bin/env bash

find ./data/Moshi-output/subjective_metrics -name "conversation_rounds.txt" | grep dB | grep cosyvoice | while read line; do
    ./4o-batch-gen.py --input $line 
done

find ./data/VITA-1.5/subjective_metrics -name "conversation_rounds.txt" | grep dB | grep cosyvoice | while read line; do
    ./4o-batch-gen.py --input $line 
done

find ./data/Freeze-omni-output/subjective_metrics -name "conversation_rounds.txt" | grep dB | grep cosyvoice | while read line; do
    ./4o-batch-gen.py --input $line 
done

find ./data/Moshi-output/subjective_metrics -name "conversation_rounds.txt" | grep -v dB | grep cosyvoice | while read line; do
    ./4o-batch-gen.py --input $line 
done

find ./data/VITA-1.5/subjective_metrics -name "conversation_rounds.txt" | grep -v dB | grep cosyvoice | while read line; do
    ./4o-batch-gen.py --input $line 
done

find ./data/Freeze-omni-output/subjective_metrics -name "conversation_rounds.txt" | grep -v dB | grep cosyvoice | while read line; do
    ./4o-batch-gen.py --input $line 
done