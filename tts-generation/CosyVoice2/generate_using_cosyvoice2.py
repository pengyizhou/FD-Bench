#!/usr/bin/env python3

import sys, os
sys.path.append('CosyVoice2/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from rich import progress_bar
import json
import numpy as np
import torchaudio.transforms as T


def test():
    cosyvoice = CosyVoice2('CosyVoice2/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    prompt_speech_16k = load_wav('./1462-170138-0000.wav', 16000)
    for i, j in enumerate(cosyvoice.inference_zero_shot('Any advice on starting a meditation practice?', "HE HAD WRITTEN A NUMBER OF BOOKS HIMSELF AMONG THEM A HISTORY OF DANCING A HISTORY OF COSTUME A KEY TO SHAKESPEARE'S SONNETS A STUDY OF THE POETRY OF ERNEST DOWSON ET CETERA", prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

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


def main_process(cosyvoice, prompt_speech_16k, ref_text, user_stream, output_dir=None, Transform=None):
    for idx, user_text in enumerate(user_stream.split("|")):
        
        for _, j in enumerate(cosyvoice.inference_zero_shot(user_text, ref_text, prompt_speech_16k, stream=False, text_frontend=False)):
            # resample the audio to 24KHz
            output = os.path.join(output_dir, f"round_{idx}.wav")
            torchaudio.save(output, j['tts_speech'], cosyvoice.sample_rate)

def generate_single_turn(ref: dict, data, output_dir):
    # Transform = T.Resample(orig_freq=16000, new_freq=24000)
    cosyvoice = CosyVoice2('CosyVoice2/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

    for idx, value in data.items():
        # create the folder                                                                                                                                 
        folder = os.path.join(output_dir, "conversation_{}".format(str(idx)))
        os.makedirs(folder, exist_ok=True)

        # get userstream                                                                                                                                    
        user_stream = value["user"]
        length_user = np.max([len(user_part) for user_part in user_stream.split("|")])

        # randomly select one of the reference audio and text
        length_ref = length_user * 2 + 1
        loop_idx = 0
        while length_ref > length_user * 2 and loop_idx < 10:
            ref_idx = np.random.choice(list(ref.keys()))
            ref_audio = ref[ref_idx]["audio"]
            ref_text = ref[ref_idx]["text"]
            length_ref = len(ref_text.split(" "))
            loop_idx += 1
        prompt_speech_16k = load_wav(ref_audio, 16000)
        main_process(cosyvoice, prompt_speech_16k, ref_text, user_stream, output_dir=folder, Transform=None)

def main():
    data = "CosyVoice2/data/conversation_round_no_interruption_label.json"
    data = read_json(data)
    ref_data = "CosyVoice2/data/LibriSpeech/dev_clean"
    ref = read_wavlist(ref_data)
    output_dir = os.path.join("CosyVoice2", "data", "cosyvoice2-single-round")
    generate_single_turn(ref, data, output_dir)


if __name__ == "__main__":
    main()