#!/usr/bin/env bash 

audio_folders=$1
port=$2

echo $audio_folders 
for audio_folder in $audio_folders; do
    echo $audio_folder
    find $audio_folder/ -type f -name "*.wav" | sort | while read audio_file; do
        echo $audio_file
        ./freeze-omni/bin/client-audio.py --port $port --audio-file $audio_file
    done
done
