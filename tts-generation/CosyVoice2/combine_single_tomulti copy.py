#!/usr/bin/env python3
# We are going to combine the wave forms in a given folder into a single wave file.
# Random silence will be added in between

from operator import length_hint
from typing import List, Dict
import torch
import torchaudio
import os

import json

interrupt_tokens = ['Further Inquiry', 'Topic Shift', 'Affirmative Acknowledgment', 'Third-Party Noise', 'Denial and Discontent', 'Affirmative Acknowledgment + Topic Shift', 'Affirmative Acknowledgment + Further Inquiry', 'Denial and Discontent + Topic Shift']

with open("CosyVoice2/data/MUSAN/noise.list", 'r') as f:
    noise_files = [line.replace("\n", "") for line in f.readlines()]

def sample_noise(noise_file, time):
    noise, _ = torchaudio.load(noise_files[noise_file])
    # get noise length in seconds
    # randomly sample from 2.5 seconds of noise
    noise_samples = int(24000 * time)
    if noise_samples > noise.shape[1]:
        noise_samples = noise.shape[1] - 10
    max_start = noise.shape[1] - noise_samples
    start = torch.randint(0, max_start, (1,)).item()
    noise_segment = noise[:, start:start + noise_samples]
    # reduce the noise volume by 10%
    noise_segment = noise_segment * 0.05
    return noise_segment
                    


def combine_single_tomulti_easy(idx: int, data: Dict, folder: str, output: str):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    print(interrupt_token)
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            wavs.append(torchaudio.load(os.path.join(folder, file))[0])
            # Add silence range from 6-10s in between
            
            if round < len(interrupt_token):
                print(interrupt_token[round])
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment':
                    wavs.append(torch.zeros(1, int(torch.randint(24000*8, 24000*10, (1,)).item())))
                elif interrupt_token[round] == 'Third-Party Noise':
                    # add some noise after some silence
                    noise_file = torch.randint(0, len(noise_files), (1,))
                    noise = sample_noise(noise_file, 6)
                    wavs.append(torch.zeros(1, int(torch.randint(24000*1, 24000*2, (1,)).item())))
                    wavs.append(noise)
                    wavs.append(torch.zeros(1, int(torch.randint(24000*1, 24000*2, (1,)).item())))
                    
                else:
                    wavs.append(torch.zeros(1, int(torch.randint(24000*6, 24000*8, (1,)).item())))

    wavs = torch.cat(wavs, dim=1)
    torchaudio.save(f'{output}.wav', wavs, 24000)
    
def combine_single_tomulti_medium(idx: int, data: Dict, folder: str, output: str):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            wavs.append(torchaudio.load(os.path.join(folder, file))[0])
            # Add silence range from 4-6s in between
            if round < len(interrupt_token) - 1:
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment':
                    wavs.append(torch.zeros(1, int(torch.randint(24000*5, 24000*6, (1,)).item())))
                elif interrupt_token[round] == 'Third-Party Noise':
                    # add some noise after some silence
                    noise_file = torch.randint(0, len(noise_files), (1,))
                    noise = sample_noise(noise_file, 4)
                    
                    wavs.append(torch.zeros(1, int(torch.randint(int(24000*0.8), int(24000*1.5), (1,)).item())))
                    wavs.append(noise)
                    wavs.append(torch.zeros(1, int(torch.randint(int(24000*0.8), int(24000*1.5), (1,)).item())))
                else:
                    wavs.append(torch.zeros(1, int(torch.randint(24000*4, 24000*5, (1,)).item())))

    wavs = torch.cat(wavs, dim=1)
    torchaudio.save(f'{output}.wav', wavs, 24000)
    
def combine_single_tomulti_hard(idx: int, data: Dict, folder: str, output: str):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            wavs.append(torchaudio.load(os.path.join(folder, file))[0])
            # Add silence range from 2-4s in between
            if round < len(interrupt_token) - 1:
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment':
                    wavs.append(torch.zeros(1, int(torch.randint(24000*3, 24000*4, (1,)).item())))
                elif interrupt_token[round] == 'Third-Party Noise':
                    # add some noise after some silence
                    noise_file = torch.randint(0, len(noise_files), (1,))
                    noise = sample_noise(noise_file, 2.5)
                    wavs.append(torch.zeros(1, int(torch.randint(int(24000*0.5), 24000, (1,)).item())))
                    wavs.append(noise)
                    wavs.append(torch.zeros(1, int(torch.randint(int(24000*0.5), 24000, (1,)).item())))
                else:
                    wavs.append(torch.zeros(1, int(torch.randint(24000*2, 24000*3, (1,)).item())))

    wavs = torch.cat(wavs, dim=1)
    torchaudio.save(f'{output}.wav', wavs, 24000)
    
    
def main():
    data_folder = "CosyVoice2/data/cosyvoice2-single-round"
    # get easier output
    output_easy = "CosyVoice2/data/cosyvoice2-single-round-combine-easy"
    output_med = "CosyVoice2/data/cosyvoice2-single-round-combine-med"
    output_hard = "CosyVoice2/data/cosyvoice2-single-round-combine-hard"

    # create these three folders
    os.makedirs(output_easy, exist_ok=True)
    os.makedirs(output_med, exist_ok=True)
    os.makedirs(output_hard, exist_ok=True)

    json_file = "CosyVoice2/data/conversation_round.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    # look through the data folder
    for idx, folder in enumerate(sorted(os.listdir(data_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))):
        if os.path.isdir(os.path.join(data_folder, folder)):
            print(f"Processing {folder}")
            # combine the wave files in the folder
            combine_single_tomulti_easy(idx, data, os.path.join(data_folder, folder), os.path.join(output_easy, folder))
            combine_single_tomulti_medium(idx, data, os.path.join(data_folder, folder), os.path.join(output_med, folder))
            combine_single_tomulti_hard(idx, data, os.path.join(data_folder, folder), os.path.join(output_hard, folder))
            
if __name__ == "__main__":
    main()