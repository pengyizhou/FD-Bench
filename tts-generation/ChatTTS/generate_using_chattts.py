#!/usr/bin/env python3
import ChatTTS
from typing import List
import torch
import torchaudio
import json
import os
from rich import progress_bar

def test():
    chat = ChatTTS.Chat()
    chat.load(compile=False) # Set to True for better performance
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_7]',
    )
    print(params_refine_text.prompt)

    text = "Any advice on starting a meditation practice.[lbreak][lbreak][lbreak] Can meditation help with stress relief."
    # text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
    rand_spk = chat.sample_random_speaker()
    # print(rand_spk) # save it for later timbre recovery

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = .3,   # using custom temperature
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
    )
    wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code, do_text_normalization=False)
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
    exit()

    ###################################
    # Sample a speaker from Gaussian.

    rand_spk = chat.sample_random_speaker()
    print(rand_spk) # save it for later timbre recovery

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = .3,   # using custom temperature
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
    )

    ###################################
    # For sentence level manual control.

    # use oral_(0-9), laugh_(0-2), break_(0-7) 
    # to generate special token in text to synthesize.
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )

    wavs = chat.infer(
        texts,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )

    ###################################
    # For word level manual control.

    text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
    wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
    """
    In some versions of torchaudio, the first line works but in other versions, so does the second line.
    """
    try:
        torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
    except:
        torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)
        
        
def run_tts_single_round(chat: ChatTTS.Chat, texts: List, output_dir: str, idx: int):
    # create a folder for each conversation
    output_dir = output_dir + "/conversation_" + str(idx)
    os.makedirs(output_dir, exist_ok=True)
    
    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = .3,   # using custom temperature
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
    )
    wavs = chat.infer(texts, skip_refine_text=True, params_infer_code=params_infer_code, do_text_normalization=False)
    # save wavs to output_dir
    for i, wav in enumerate(wavs):
        torchaudio.save(output_dir + "/round_" + str(i) + ".wav", torch.from_numpy(wav).unsqueeze(0), 24000)
        
        
def run_tts_multi_round(chat: ChatTTS.Chat, texts: List, output_dir: str, idx: int):
    
    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = .3,   # using custom temperature
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[break_4]',
    )
    wavs = chat.infer(texts, skip_refine_text=True, params_infer_code=params_infer_code, do_text_normalization=False, params_refine_text=params_refine_text)
    # save wavs to output_dir
    torchaudio.save(output_dir + "/conversation_{}.wav".format(str(idx)), torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
        

def test_blank(chat: ChatTTS.Chat):
    texts = ["Any advice on starting a meditation practice. [uv_break]Can meditation help with stress relief. [uv_break]Oh, also, what's a good app for guided meditation? [uv_break]By the way, do you know if I should meditate in the morning or evening?"]
    params_refine_texts = [ ChatTTS.Chat.RefineTextParams(
        prompt=f'[break_{i}]',
    ) for i in range(8) ]
    wavs = list()
    for params_refine_text in params_refine_texts:
        wavs.append(chat.infer(texts, skip_refine_text=True, do_text_normalization=False, params_refine_text=params_refine_text)[0])
    for i, wav in enumerate(wavs):
        torchaudio.save("data/chattts-blank/blank_output_" + str(i+8) + ".wav", torch.from_numpy(wav).unsqueeze(0), 24000)
    

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def generate_single_round(chat, data, output_dir):

    for idx, value in data.items():
        print("start processing conversation {}".format(str(idx)))
        user_stream = value["user"].split("|")
        print(user_stream)
        run_tts_single_round(chat, user_stream, output_dir, idx)


def generate_multi_round(chat, data, output_dir):
    for idx, value in data.items():
        print("start processing conversation {}".format(str(int(idx))))
        user_stream: str = value["user"].replace("|", "[lbreak] [lbreak]")
        print(user_stream)
        run_tts_multi_round(chat, [user_stream], output_dir, int(idx))
        



def main():
    chat = ChatTTS.Chat()
    chat.load(compile=False) # Set to True for better performance
    # test_blank(chat)
    output_dir = "ChatTTS/data/chattts-single-round"
    data = read_json("conversation_round_no_interruption_label.json")
    generate_single_round(chat, data, output_dir)
    
    
if __name__ == "__main__":
    main()