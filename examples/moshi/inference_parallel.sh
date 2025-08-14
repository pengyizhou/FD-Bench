#!/usr/bin/env bash

mkdir -p ./data/S-C-VAD0.5/moshi_output
mkdir -p ./data/S-C-VAD0.5/moshi_input

python ./moshi/client_audiofile.py --port 8991 --audio-folder "chattts-single-round-combine-easy chattts-single-round-combine-med"> ./log/new-moshi-client-8991.log 2>&1 &

python ./moshi/client_audiofile.py --port 8992 --audio-folder "chattts-single-round-combine-hard f5tts-single-round-combine-easy" > ./log/new-moshi-client-8992.log 2>&1 &

python ./moshi/client_audiofile.py --port 8993 --audio-folder "f5tts-single-round-combine-med f5tts-single-round-combine-hard" > ./log/new-moshi-client-8993.log 2>&1 &

python ./moshi/client_audiofile.py --port 8994 --audio-folder "cosyvoice2-single-round-combine-easy cosyvoice2-single-round-combine-med cosyvoice2-single-round-combine-hard" > ./log/new-moshi-client-8994.log 2>&1
