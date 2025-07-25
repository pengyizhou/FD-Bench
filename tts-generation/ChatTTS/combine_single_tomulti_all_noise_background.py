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

with open("F5TTS/data/MUSAN/noise.list", 'r') as f:
    noise_files = [line.replace("\n", "") for line in f.readlines()]

def add_noise_to_audio(audio, noise_file, snr):
    """
    Adds background noise to an audio tensor at a specified SNR.

    Parameters:
        audio (Tensor): Audio tensor of shape (channels, samples).
        noise_file (str): Path to the noise audio file.
        snr (float): Desired signal-to-noise ratio in dB.

    Returns:
        Tensor: The noisy audio tensor.
    """
    # Load the noise audio
    noise, noise_sr = torchaudio.load(noise_file)
    
    # Optional: Check if sampling rates match. If not, consider resampling.
    # For this example, we assume the sampling rates match.
    
    # Determine the number of samples in the audio tensor.
    audio_length = audio.shape[1]
    
    # Ensure the noise file is long enough; if not, handle appropriately.
    if noise.shape[1] < audio_length:
        raise ValueError("Noise file must be at least as long as the audio signal.")
    
    # Randomly select a starting point for a noise segment matching the audio length.
    max_start = noise.shape[1] - audio_length
    start_idx = torch.randint(0, max_start + 1, (1,)).item()  # +1 to include max_start
    noise_segment = noise[:, start_idx:start_idx + audio_length]
    
    # Compute the average power of the audio (signal) and the noise segment.
    audio_power = audio.pow(2).mean()
    noise_power = noise_segment.pow(2).mean()
    
    # Compute the desired noise power for the target SNR.
    desired_noise_power = audio_power / (10 ** (snr / 10))
    
    # Calculate the scaling factor for the noise segment.
    scaling_factor = torch.sqrt(desired_noise_power / noise_power)
    noise_segment_scaled = noise_segment * scaling_factor
    
    # Mix the scaled noise with the original audio.
    noisy_audio = audio + noise_segment_scaled
    
    return noisy_audio

                    
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
            start_time = speech_timestamp[0].get('start') + int(current_start_time * 16000)
            end_time = speech_timestamp[-1].get('end') + int(current_start_time * 16000)
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
    noisy_audio = None
    while noisy_audio is None:
        try:
            noise_file = torch.randint(0, len(noise_files), (1,))
            print(f"Adding noise from {noise_files[noise_file]}")
            noisy_audio = add_noise_to_audio(wavs, noise_files[noise_file], snr)
        except ValueError:
            print("retrying..., ")
            continue
    torchaudio.save(f'{output}.wav', noisy_audio, 24000)
    with open(f'{output}.timestamps', 'w') as f:
        json.dump(speech_timestamps, f)
    
    
    
def main():
    data_folder = "ChatTTS/data/chattts-single-round"
    # get easier output
    snr = 20
    output_easy = f"ChatTTS/data/chattts-single-round-combine-easy-noisy-bg-{snr}dB"

    # create these three folders
    os.makedirs(output_easy, exist_ok=True)
    # os.makedirs(output_med, exist_ok=True)
    # os.makedirs(output_hard, exist_ok=True)
    vad_model = load_silero_vad()
    json_file = "ChatTTS/data/conversation_round.json"
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