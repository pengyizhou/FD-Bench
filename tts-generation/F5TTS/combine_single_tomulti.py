#!/usr/bin/env python

# We are going to combine the wave forms in a given folder into a single wave file.
# Random silence will be added in between

from operator import length_hint
from typing import List, Dict
import torch
import torchaudio
import os
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

import json

interrupt_tokens = ['Further Inquiry', 'Topic Shift', 'Affirmative Acknowledgment', 'Third-Party Noise', 'Denial and Discontent', 'Affirmative Acknowledgment + Topic Shift', 'Affirmative Acknowledgment + Further Inquiry', 'Denial and Discontent + Topic Shift']

                    
def silero_vad(vad_model, output_audio_file):
    output_audio = read_audio(output_audio_file)
    
    output_speech_timestamps = get_speech_timestamps(
        output_audio,
        vad_model,
        return_seconds=False,
        threshold=0.5,
        min_silence_duration_ms=1500,
    )
    return output_speech_timestamps

def combine_single_tomulti_easy(idx: int, data: Dict, folder: str, output: str, vad_model):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    print(interrupt_token)
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    speech_timestamps = list()
    blank_len = 0
    segment_len = 0
    current_start_time = 0
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            current_start_time += segment_len + blank_len # in seconds format
            speech_segment = torchaudio.load(os.path.join(folder, file))
            # apply VAD first, to get absolute timestamps
            speech_timestamp = silero_vad(vad_model, os.path.join(folder, file))
            if len(speech_timestamp) > 0 :
                start_time = speech_timestamp[0].get('start') + int(current_start_time * 16000)
                end_time = speech_timestamp[-1].get('end') + int(current_start_time * 16000)
            elif len(speech_timestamp) == 0:
                start_time = 0 + int(current_start_time * 16000)
                end_time = 0 + int((float(speech_segment[0].shape[1] / 24000) + current_start_time) * 16000)
                print("VAD failed")
                
            speech_timestamps.append({'start': start_time, 'end': end_time})
            wavs.append(speech_segment[0])
            segment_len = float(speech_segment[0].shape[1] / 24000) # in seconds format
            # Add silence range from 6-10s in between
            
            if round < len(interrupt_token):
                print(interrupt_token[round])
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment' or interrupt_token[round] == "Third-Party Noise":
                    # add some blank
                    
                    blank_dur = int(torch.randint(24000*8, 24000*10, (1,)).item())
                    blank_len = float(blank_dur / 24000)
                    blank_segment = torch.zeros(1, blank_dur)

                    wavs.append(blank_segment)
                    
                else:
                    blank_dur = int(torch.randint(24000*6, 24000*8, (1,)).item())
                    blank_len = float(blank_dur / 24000)
                    blank_segment = torch.zeros(1, blank_dur)

                    wavs.append(blank_segment)

    wavs = torch.cat(wavs, dim=1)
    # torchaudio.save(f'{output}-clean.wav', wavs, 24000)

    torchaudio.save(f'{output}.wav', wavs, 24000)
    with open(f'{output}.timestamps', 'w') as f:
        json.dump(speech_timestamps, f)
    
def combine_single_tomulti_medium(idx: int, data: Dict, folder: str, output: str, vad_model):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    print(interrupt_token)
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    speech_timestamps = list()
    blank_len = 0
    segment_len = 0
    current_start_time = 0
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            current_start_time += segment_len + blank_len # in seconds format
            speech_segment = torchaudio.load(os.path.join(folder, file))
            # apply VAD first, to get absolute timestamps
            speech_timestamp = silero_vad(vad_model, os.path.join(folder, file))
            if len(speech_timestamp) > 0 :
                start_time = speech_timestamp[0].get('start') + int(current_start_time * 16000)
                end_time = speech_timestamp[-1].get('end') + int(current_start_time * 16000)
            elif len(speech_timestamp) == 0:
                start_time = 0 + int(current_start_time * 16000)
                end_time = 0 + int((float(speech_segment[0].shape[1] / 24000) + current_start_time) * 16000)
                print("VAD failed")
            speech_timestamps.append({'start': start_time, 'end': end_time})
            wavs.append(speech_segment[0])
            segment_len = float(speech_segment[0].shape[1] / 24000) # in seconds format
            # Add silence range from 6-10s in between
            
            if round < len(interrupt_token):
                print(interrupt_token[round])
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment' or interrupt_token[round] == "Third-Party Noise":
                    # add some blank
                    
                    blank_dur = int(torch.randint(24000*5, 24000*6, (1,)).item())
                    blank_len = float(blank_dur / 24000)
                    blank_segment = torch.zeros(1, blank_dur)

                    wavs.append(blank_segment)
                    
                else:
                    blank_dur = int(torch.randint(24000*4, 24000*5, (1,)).item())
                    blank_len = float(blank_dur / 24000)
                    blank_segment = torch.zeros(1, blank_dur)

                    wavs.append(blank_segment)

    wavs = torch.cat(wavs, dim=1)
    # torchaudio.save(f'{output}-clean.wav', wavs, 24000)

    torchaudio.save(f'{output}.wav', wavs, 24000)
    with open(f'{output}.timestamps', 'w') as f:
        json.dump(speech_timestamps, f)

def combine_single_tomulti_hard(idx: int, data: Dict, folder: str, output: str, vad_model):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    print(interrupt_token)
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    speech_timestamps = list()
    blank_len = 0
    segment_len = 0
    current_start_time = 0
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            current_start_time += segment_len + blank_len # in seconds format
            speech_segment = torchaudio.load(os.path.join(folder, file))
            # apply VAD first, to get absolute timestamps
            speech_timestamp = silero_vad(vad_model, os.path.join(folder, file))
            if len(speech_timestamp) > 0 :
                start_time = speech_timestamp[0].get('start') + int(current_start_time * 16000)
                end_time = speech_timestamp[-1].get('end') + int(current_start_time * 16000)
            elif len(speech_timestamp) == 0:
                start_time = 0 + int(current_start_time * 16000)
                end_time = 0 + int((float(speech_segment[0].shape[1] / 24000) + current_start_time) * 16000)
                print("VAD failed")
            speech_timestamps.append({'start': start_time, 'end': end_time})
            wavs.append(speech_segment[0])
            segment_len = float(speech_segment[0].shape[1] / 24000) # in seconds format
            # Add silence range from 6-10s in between
            
            if round < len(interrupt_token):
                print(interrupt_token[round])
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment' or interrupt_token[round] == "Third-Party Noise":
                    # add some blank
                    
                    blank_dur = int(torch.randint(24000*3, 24000*4, (1,)).item())
                    blank_len = float(blank_dur / 24000)
                    blank_segment = torch.zeros(1, blank_dur)

                    wavs.append(blank_segment)
                    
                else:
                    blank_dur = int(torch.randint(24000*2, 24000*3, (1,)).item())
                    blank_len = float(blank_dur / 24000)
                    blank_segment = torch.zeros(1, blank_dur)

                    wavs.append(blank_segment)

    wavs = torch.cat(wavs, dim=1)
    # torchaudio.save(f'{output}-clean.wav', wavs, 24000)

    torchaudio.save(f'{output}.wav', wavs, 24000)
    with open(f'{output}.timestamps', 'w') as f:
        json.dump(speech_timestamps, f)

   
def main():
    data_folder = "F5TTS/data/f5tts-single-round"
    # get easier output
    output_easy = f"F5TTS/data/f5tts-single-round-combine-easy"
    output_med = f"F5TTS/data/f5tts-single-round-combine-med"
    output_hard = f"F5TTS/data/f5tts-single-round-combine-hard"

    # create these three folders
    os.makedirs(output_easy, exist_ok=True)
    os.makedirs(output_med, exist_ok=True)
    os.makedirs(output_hard, exist_ok=True)
    vad_model = load_silero_vad()
    json_file = "F5TTS/data/conversation_round.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    # look through the data folder
    for idx, folder in enumerate(sorted(os.listdir(data_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))):
        if os.path.isdir(os.path.join(data_folder, folder)):
            print(f"Processing {folder}")
            # combine the wave files in the folder
            combine_single_tomulti_easy(idx, data, os.path.join(data_folder, folder), os.path.join(output_easy, folder), vad_model)
            combine_single_tomulti_medium(idx, data, os.path.join(data_folder, folder), os.path.join(output_med, folder), vad_model)
            combine_single_tomulti_hard(idx, data, os.path.join(data_folder, folder), os.path.join(output_hard, folder), vad_model)
            
if __name__ == "__main__":
    main()