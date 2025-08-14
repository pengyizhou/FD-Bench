# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Client with file-based audio I/O for Moshi server."""

import argparse
import asyncio
import queue
import sys
import torch
import aiohttp
import numpy as np
import soundfile as sf
import sphn
import os
import ast
from moshi.client_utils import AnyPrinter, Printer, RawPrinter

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

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


class Connection:
    def __init__(
        self,
        printer: AnyPrinter,
        websocket: aiohttp.ClientWebSocketResponse,
        audio_file: str,
        new_audio_file: str,
        sample_rate: float = 24000,
        channels: int = 1,
        frame_size: int = 1920,
    ) -> None:
        self.printer = printer
        self.websocket = websocket
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self.audio_file = audio_file
        self.new_audio_file = new_audio_file

        self._done = False
        self.output_buffer = []
        
        # Load input audio file
        audio_data, sr = sf.read(audio_file, dtype='float32')
        silence_start = torch.zeros(int(3 * sr), dtype=torch.float32)
        self.silence_samples_start_16000 = int(len(silence_start) / 1.5)
        silence_end = torch.zeros(int(10 * sr), dtype=torch.float32)
        self.audio_data = torch.cat((silence_start, torch.tensor(audio_data), silence_end)).numpy()
        
        # save the audio file first
        sf.write(self.new_audio_file, self.audio_data, self.sample_rate)
        self.audio_position = 0
        
        self._opus_writer = sphn.OpusStreamWriter(sample_rate)
        self._opus_reader = sphn.OpusStreamReader(sample_rate)
        self.all_tokens = list()
        self.text_tokens = list()

    async def _audio_streamer(self) -> None:
        """Simulate real-time audio input streaming"""
        start_time = asyncio.get_event_loop().time()
        
        while self.audio_position < len(self.audio_data) and not self._done:
            expected_time = start_time + (self.audio_position / self.sample_rate)
            current_time = asyncio.get_event_loop().time()
            
            if (remaining := expected_time - current_time) > 0:
                await asyncio.sleep(remaining)
            
            end = self.audio_position + self.frame_size
            chunk = self.audio_data[self.audio_position:end]
            
            if len(chunk) < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - len(chunk)), 'constant')
            
            self._opus_writer.append_pcm(chunk)
            self.audio_position += self.frame_size
            await asyncio.sleep(0)
        await asyncio.sleep(1)
        self._lost_connection()
        await self.websocket.close()

    async def _queue_loop(self) -> None:
        """Send encoded audio to server"""
        while not self._done:
            await asyncio.sleep(0.001)
            msg = self._opus_writer.read_bytes()
            if msg:
                try:
                    await self.websocket.send_bytes(b"\x01" + msg)
                except Exception as e:
                    self.printer.log("error", str(e))
                    self._lost_connection()

    async def _decoder_loop(self) -> None:
        """Collect received audio into buffer"""
        all_pcm_data = np.array([])
        while not self._done:
            await asyncio.sleep(0.001)
            pcm = self._opus_reader.read_pcm()
            if pcm is None:
                continue
                
            all_pcm_data = np.concatenate([all_pcm_data, pcm]) if all_pcm_data.size else pcm
            
            while len(all_pcm_data) >= self.frame_size:
                chunk = all_pcm_data[:self.frame_size]
                self.output_buffer.append(chunk)
                all_pcm_data = all_pcm_data[self.frame_size:]

        # Add remaining audio
        if all_pcm_data.size > 0:
            self.output_buffer.append(all_pcm_data)

    async def _recv_loop(self) -> None:
        """Handle incoming messages"""
        try:
            async for message in self.websocket:
                if message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    self.printer.log("error", f"{self.websocket.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.BINARY:
                    data = message.data
                    if data[0] == 1:  # Audio
                        self._opus_reader.append_bytes(data[1:])
                    elif data[0] == 2:  # Text
                        self.printer.print_token(data[1:].decode())
                        self.text_tokens.append(data[1:].decode())
                    elif data[0] == 3:  # All Tokens
                        self.all_tokens.append(int(data[1:].decode()))
        except Exception as e:
            self.printer.log("error", str(e))
        finally:
            self._lost_connection()

    def _lost_connection(self) -> None:
        """Handle connection termination"""
        if not self._done:
            self._done = True
            self.printer.log("info", "Connection closed")

    async def run(self) -> None:
        """Main processing loop"""
        await asyncio.gather(
            self._recv_loop(),
            self._decoder_loop(),
            self._queue_loop(),
            self._audio_streamer()
        )
        
    async def disconnect(self):
        await self.websocket.close()
        self.printer.print("Disconnected from server")

    def save_output(self, output_file: str) -> None:
        """Save collected audio to file"""
        if not self.output_buffer:
            self.printer.log("warning", "No audio data received to save")
            return
            
        full_audio = np.concatenate(self.output_buffer)
        sf.write(output_file, full_audio, self.sample_rate)
        self.printer.log("info", f"Saved output audio to {output_file}")


async def run(printer: AnyPrinter, args):
    # URI construction remains unchanged from original
    if args.url is None:
        proto = "wss" if args.https else "ws"
        uri = f"{proto}://{args.host}:{args.port}/api/chat"
    else:
        uri = args.url if "://" in args.url else f"wss://{args.url}"
        
    data_root = "data"
    vad_model = load_silero_vad()
    # audio_folders = ["chattts-single-round", "f5tts-single-round", "cosyvoice2-single-round"]
    audio_folders = args.audio_folder.split()
    # audio_folders = ["test"]
    for audio_folder in audio_folders:
        moshi_output_token_file = os.path.join(data_root, "S-C-VAD0.5", "moshi_output", audio_folder + ".txt")
        moshi_output_folder = os.path.join(data_root, "S-C-VAD0.5", "moshi_output", audio_folder)
        moshi_input_folder = os.path.join(data_root, "S-C-VAD0.5", "moshi_input", audio_folder)
        # mkdir input and output folder
        os.makedirs(moshi_output_folder, exist_ok=True)
        os.makedirs(moshi_input_folder, exist_ok=True)
        with open(moshi_output_token_file, "w") as f:
            audio_folder_path = os.path.join(data_root, audio_folder)
            for conversation in os.listdir(audio_folder_path):
                if conversation.endswith(".timestamps"):
                    continue
                conversation_path = os.path.join(audio_folder_path, conversation)
                input_timestamps_path = os.path.join(audio_folder_path, conversation.replace(".wav", ".timestamps"))
                output_conv_path = os.path.join(moshi_output_folder, conversation)
                new_input_file = os.path.join(moshi_input_folder, conversation)
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(uri) as ws:
                        printer.log("info", f"Connected to {uri}")
                        connection = Connection(printer, ws, conversation_path, new_input_file)
                        await connection.run()
                        connection.save_output(output_conv_path)
                        if os.path.exists(input_timestamps_path):
                            print(f"Using existing timestamps for {conversation}")
                            with open(input_timestamps_path, "r") as ts:
                                input_time_stamp_original = ast.literal_eval(ts.read())
                                input_time_stamp = list()
                                for i in input_time_stamp_original:
                                    input_time_stamp.append({'start': int(i.get("start")) + connection.silence_samples_start_16000, 'end': int(i.get("end")) + connection.silence_samples_start_16000})
                        else:
                            input_time_stamp = silero_vad(vad_model, new_input_file)
                        output_time_stamp = silero_vad(vad_model, output_conv_path)
                        # conversation_177.wav || [3, 3, 3, 3, 0, ..., 5287, 3, 3, 3] || [' Hello', ',', ' how', ' can', ' I', ' help', ' you', '?', ... ' Okay', '.', ' Sure', ',', ' what', ' do', ' I', ' need', '?', ' Sure', ',', ' I', ' recommend', ' starting', ' with', ' a', ' basic', ' photography', ' course', ' online', ' or', ' in', ' person', '.', ' will', ' give', ' you', ' a', ' good', ' foundation'] || [{'start': 53280, 'end': 93664}, {'start': 218656, 'end': 250848}, {'start': 397856, 'end': 422368}, {'start': 565280, 'end': 597472}, {'start': 752160, 'end': 794592}] || [{'start': 7200, 'end': 36320}, {'start': 78368, 'end': 83936}, {'start': 142368, 'end': 322528}, {'start': 344096, 'end': 488928}, {'start': 573984, 'end': 655840}, {'start': 770592, 'end': 804320}, {'start': 848416, 'end': 904160}, {'start': 935456, 'end': 953600}]
                        f.write(f"{conversation} || {connection.all_tokens} || {connection.text_tokens} || {input_time_stamp} || {output_time_stamp}\n")
                        print(f"{conversation} || {connection.all_tokens} || {connection.text_tokens} || {input_time_stamp} || {output_time_stamp}")  
                            
                    # print(connection.all_tokens) 
                    # print(connection.text_tokens)


def main():
    parser = argparse.ArgumentParser("file_client")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8998, help="Server port")
    parser.add_argument("--https", action="store_true", help="Use HTTPS")
    parser.add_argument("--url", help="Direct server URL")
    parser.add_argument("--audio-folder", default="cosyvoice2-single-round-combine-easy-noisy-bg-20dB", help="Input audio file path")
    args = parser.parse_args()

    printer = Printer() if sys.stdout.isatty() else RawPrinter()
    
    try:
        asyncio.run(run(printer, args))
    except KeyboardInterrupt:
        printer.log("warning", "Interrupted by user")
    printer.log("info", "Client shutdown complete")


if __name__ == "__main__":
    main()