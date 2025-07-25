#!/home/yizhou/miniconda3/envs/f5-tts/bin/python


import json
import os
import re
from importlib.resources import files
from pathlib import Path
import ipdb
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT
import pyloudnorm as pyln
config = tomli.load(open(os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"), "rb"))

vocoder_name = "vocos"
model_cls = DiT
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
if vocoder_name == "vocos":
    repo_name = "F5-TTS"
    exp_name = "F5TTS_Base"
    ckpt_step = 1200000
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path
elif vocoder_name == "bigvgan":
    repo_name = "F5-TTS"
    exp_name = "F5TTS_Base_bigvgan"
    ckpt_step = 1250000
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))

ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file="")
vocoder = load_vocoder(vocoder_name=vocoder_name)

def main_process(ref_audio, ref_text, text_gen, model_obj, mel_spec_type, remove_silence, speed, output_dir, wave_path):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("Voice:", voice)
        print("Ref_audio:", voices[voice]["ref_audio"])
        print("Ref_text:", voices[voice]["ref_text"])

    generated_audio_segments = []
    chunks = text_gen.split("|")
    for text in chunks:
        if not text.strip():
            continue
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        gen_text = text.strip()
        ref_audio = voices[voice]["ref_audio"]
        ref_text = voices[voice]["ref_text"]
        print(f"Voice: {voice}")
        audio, final_sample_rate, spectragram = infer_process(
            ref_audio, ref_text, gen_text, model_obj, vocoder, mel_spec_type=mel_spec_type, speed=speed
        )
        generated_audio_segments.append(audio)

    if generated_audio_segments:
        for idx, audio_seg in enumerate(generated_audio_segments):
            # output to wave files and save to output_dir according to the order
            output = os.path.join(output_dir, f"output_{idx}.wav")
            # normalize the volume of the audio to -14 LUFS

            meter = pyln.Meter(final_sample_rate)  # create BS.1770 meter
            loudness = meter.integrated_loudness(audio_seg)  # measure loudness
            normalized_audio = pyln.normalize.loudness(audio_seg, loudness, -14.0)  # normalize audio to -14 LUFS

            sf.write(output, normalized_audio, final_sample_rate)

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def read_wavlist(wavlist_dir: str):
    # read the wav.scp and text file
    wavscp = {}
    texts = {}
    wavlist = {}
    with open(os.path.join(wavlist_dir, "wav.scp"), 'r') as f:
        for line in f:
            idx, path = line.strip().split(" ", 1)
            wavscp[idx] = path
    with open(os.path.join(wavlist_dir, "text"), 'r') as f:
        for line in f:
            idx, text = line.strip().split(" ", 1)
            texts[idx] = text.lower()
    
    for idx in wavscp.keys():
        wavlist[idx] = {"audio": wavscp[idx], "text": texts[idx]}
    return wavlist


def generate_single_turn(ref: dict, data, output_dir):
    for idx, value in data.items():
        # create the folder 
        folder = os.path.join(output_dir, "conversation_{}".format(str(idx)))
        os.makedirs(folder, exist_ok=True)
        
        # get userstream
        user_stream = value["user"]
        
        # randomly select one of the reference audio and text
        ref_idx = np.random.choice(list(ref.keys()))
        ref_audio = ref[ref_idx]["audio"]
        ref_text = ref[ref_idx]["text"]
        main_process(ref_audio, ref_text, user_stream, ema_model, vocoder_name, remove_silence=False, speed=1.0, output_dir=folder, wave_path=os.path.join(folder, "output.wav"))
        
        
        
    

def main():
    
    data = "F5TTS/data/conversation_round_no_interruption_label.json"
    data = read_json(data)
    ref_data = "F5TTS/data/LibriSpeech/dev_clean"
    ref = read_wavlist(ref_data)
    generate_single_turn(ref, data, "F5TTS/data/f5tts-single-round")
    # main_process(ref_audio, ref_text, gen_text, ema_model, vocoder_name, remove_silence=False, speed=1.0, output_dir="output", wave_path="output/output.wav")


if __name__ == "__main__":
    main()
