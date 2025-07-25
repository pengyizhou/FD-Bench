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

with open("CosyVoice2/data/MUSAN/noise.list", 'r') as f:
    noise_files = [line.replace("\n", "") for line in f.readlines()]

def sample_noise(speech_file, noise_file, noise_dur, snr):
    # Load the noise and speech signals.
    noise, sr_noise = torchaudio.load(noise_file)
    speech, sr_speech = torchaudio.load(speech_file)
    
    # Ensure that the sampling rates are the same.
    if sr_noise != sr_speech:
        raise ValueError("Sampling rates of noise and speech must match!")
    
    # Calculate the number of samples corresponding to the desired segment length.
    segment_length = noise_dur
    
    # Ensure the noise file is long enough.
    if segment_length > noise.shape[1]:
        raise ValueError("Noise file is shorter than the desired segment length.")
    
    # Randomly select a starting index for the noise segment.
    # Note: max_start is set such that the segment fits within the noise signal.
    max_start = noise.shape[1] - segment_length
    start_idx = torch.randint(0, max_start + 1, (1,)).item()  # +1 to make the range inclusive.
    noise_segment = noise[:, start_idx:start_idx + segment_length]
    
    # Compute the average power of the speech segment.
    # Power is calculated as the mean squared value.
    signal_power = speech.pow(2).mean()
    
    # Compute the average power of the noise segment.
    noise_power = noise_segment.pow(2).mean()
    
    # Compute the desired noise power to achieve the target SNR (in dB).
    # Using the formula: SNR (dB) = 10 * log10(P_signal / P_noise)
    desired_noise_power = signal_power / (10 ** (snr / 10))
    
    # Calculate the scaling factor to adjust the noise power.
    scaling_factor = torch.sqrt(desired_noise_power / noise_power)
    
    # Scale the noise segment accordingly.
    noise_segment_scaled = noise_segment * scaling_factor
    
    return noise_segment_scaled
                    
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

def combine_single_tomulti_easy(idx: int, data: Dict, folder: str, output: str, snr: int, vad_model):
    wavs = []
    # create the folder first
    # os.makedirs(output, exist_ok=True)
    user_stream_text = data[str(idx+1)]['user'].replace("<", "|").replace(">", "|")
    user_stream = user_stream_text.split("|")
    interrupt_token = [stream for stream in user_stream if stream in interrupt_tokens]
    print(interrupt_token)
    assert len(interrupt_token) + 1 == len(os.listdir(folder))
    speech_timestamps = list()
    noise_len = 0
    segment_len = 0
    current_start_time = 0
    for round, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".wav"):
            print(f"Loading {file}")
            current_start_time += segment_len + noise_len # in seconds format
            speech_segment = torchaudio.load(os.path.join(folder, file))
            # apply VAD first, to get absolute timestamps
            speech_timestamp = silero_vad(vad_model, os.path.join(folder, file))
            start_time = speech_timestamp[0].get('start') + int(current_start_time * 16000)
            end_time = speech_timestamp[-1].get('end') + int(current_start_time * 16000)
            speech_timestamps.append({'start': start_time, 'end': end_time})
            wavs.append(speech_segment[0])
            segment_len = float(speech_segment[0].shape[1] / 24000) # in seconds format
            # Add silence range from 6-10s in between
            
            if round < len(interrupt_token):
                print(interrupt_token[round])
                if interrupt_token[round] == 'Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment + Further Inquiry' or interrupt_token[round] == 'Affirmative Acknowledgment' or interrupt_token[round] == "Third-Party Noise":
                    # add some noise
                    noise_segment = None
                    while noise_segment is None:
                        try:
                            noise_file = torch.randint(0, len(noise_files), (1,))
                            noise_dur = int(torch.randint(24000*8, 24000*10, (1,)).item())
                            noise_len = float(noise_dur / 24000)
                            noise_segment = sample_noise(os.path.join(folder, file), noise_files[noise_file], noise_dur, snr)
                        except ValueError:
                            print("retrying...")
                            continue
                    wavs.append(noise_segment)
                    
                else:
                    noise_segment = None
                    while noise_segment is None:
                        try:
                            noise_file = torch.randint(0, len(noise_files), (1,))
                            noise_dur = int(torch.randint(24000*6, 24000*8, (1,)).item())
                            noise_len = float(noise_dur / 24000)
                            noise_segment = sample_noise(os.path.join(folder, file), noise_files[noise_file], noise_dur, snr)
                        except ValueError:
                            print("retrying...")
                            continue
                    wavs.append(noise_segment)

    wavs = torch.cat(wavs, dim=1)
    torchaudio.save(f'{output}.wav', wavs, 24000)
    with open(f'{output}.timestamps', 'w') as f:
        json.dump(speech_timestamps, f)
    
    
def main():
    data_folder = "CosyVoice2/data/cosyvoice2-single-round"
    # get easier output
    snr = 20
    output_easy = f"CosyVoice2/data/cosyvoice2-single-round-combine-easy-noisy-gap-{snr}dB"

    # create these three folders
    os.makedirs(output_easy, exist_ok=True)
    # os.makedirs(output_med, exist_ok=True)
    # os.makedirs(output_hard, exist_ok=True)
    vad_model = load_silero_vad()
    json_file = "CosyVoice2/data/conversation_round.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    # look through the data folder
    for idx, folder in enumerate(sorted(os.listdir(data_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))):
        if os.path.isdir(os.path.join(data_folder, folder)):
            print(f"Processing {folder}")
            # combine the wave files in the folder
            combine_single_tomulti_easy(idx, data, os.path.join(data_folder, folder), os.path.join(output_easy, folder), snr, vad_model)
            
if __name__ == "__main__":
    main()