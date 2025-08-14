import json
import torch
import threading
import queue
import subprocess
import concurrent.futures

import numpy as np
import torchaudio.compliance.kaldi as k

from silero_vad.model import load_silero_vad
from silero_vad.utils_vad import VADIterator

class VAD:
    def __init__(self, cache_history=10):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.cache_history = cache_history
        self.in_dialog = False

        with torch.no_grad():
            self.load_vad()
            self.reset_vad()
    
    def get_chunk_size(self):
        return self.CHUNK

    def load_vad(self):
        self.vad_model = load_silero_vad()
        self.vad_model.eval()
        # generate vad itertator
        print("##################### Generating VAD iterator..., threshold=0.5")
        self.vad_iterator = VADIterator(self.vad_model, 
                                        threshold=0.8, 
                                        sampling_rate=16000, 
                                        min_silence_duration_ms=1000, 
                                        speech_pad_ms=30)
        self.vad_iterator.reset_states()

    def reset_vad(self):
        # reset all parms
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap , 1])
        self.history = torch.zeros([self.cache_history, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.vad_iterator.reset_states()
        self.in_dialog = False
    
    def run_vad_iterator(self, audio):
        speech_dict_out = None
        # split into chunk with 512
        for i in range(len(audio) // 512):
            speech_dict = self.vad_iterator(audio[i * 512: (i + 1) * 512], return_seconds=True)
            if speech_dict is not None:
                speech_dict_out = speech_dict
        return speech_dict_out
    
    def predict(self,
                audio: torch.Tensor):
        """
        Predict the Voice Activity Detection (VAD) status and return related features.

        Parameters:
        - audio (torch.Tensor): A 1D or 2D tensor representing the input audio signal.

        Returns:
        - return_dict (dict): A dictionary containing the VAD status and related features.
            - 'status' (str): The current VAD status, which can be 'sl' (speech start), 
                              'cl' (speech continue), or 'el' (speech end).
            - 'feature_last_chunk' (list of list of float): The feature of the last chunks.
            - 'feature' (list of list of float): The feature of the current chunk of audio.
            - 'history_feature' (list of list of list of float): The cached features of previous chunks.
        
        """

        # 1. Converts the input audio tensor to the appropriate format.
        # 2. Computes the filter bank features (fbank) for the audio.
        # 3. Updates the input chunk and history based on the new audio segment.
        # 4. Determines the VAD status by running the VAD iterator on the audio.
        # 5. Populates the return dictionary with the VAD status and related features.

        return_dict = {}
        return_dict['status'] = None
        with torch.no_grad():
            # get fbank feature
            audio = torch.tensor(audio)
            sample_data = audio.reshape(1, -1, 1)[:, :, :1] * 32768
            self.input_sample[:, :self.frame_overlap , :] = self.input_sample[:, -self.frame_overlap:, :].clone()
            self.input_sample[:, self.frame_overlap:, :] = sample_data
            # compute kaldi style feature
            xs = k.fbank(waveform = self.input_sample.squeeze(-1), dither=0, 
                        frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            self.input_chunk[:, :self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap:, :].clone()
            self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)

            # get vad status
            if self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "end" in speech_dict:
                    return_dict['status'] = 'el'
                    # reset state
                    self.vad_iterator.reset_states()
                else:
                    return_dict['status'] = 'cl'
            if not self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "start" in speech_dict:
                    return_dict['status'] = 'sl'
                    self.in_dialog = True
                    # self.vad_iterator.reset_states()
                else:  
                    # cache fbank feature
                    self.history[:-1] = self.history[1:].clone()
                    self.history[-1:] = self.input_chunk

            # return dict
            if return_dict['status'] == 'sl':
                # copy last 6 chunks
                return_dict['feature_last_chunk'] = self.history[-6:].unsqueeze(1).numpy().tolist()
                return_dict['feature'] = self.input_chunk.numpy().tolist()
                return_dict['history_feature'] = self.history.numpy().tolist()
            elif return_dict['status'] == 'cl' or return_dict['status'] == 'el':
                return_dict['feature_last_chunk'] = None
                return_dict['feature'] = self.input_chunk.numpy().tolist()
                return_dict['history_feature'] = self.history.numpy().tolist()

        return return_dict
