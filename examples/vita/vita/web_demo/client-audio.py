#!/usr/bin/env python

import time
import json
import threading
import numpy as np
import socketio
import soundfile as sf
from collections import deque
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import os
from datetime import datetime
import ast

import argparse



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


# Configuration
FIXED_SAMPLE_RATE = 24000             # Same as your server logic
CHUNK_SIZE = 4800                      # Number of samples in a small chunk
BYTES_PER_SAMPLE = 2
REAL_TIME_INTERVAL = CHUNK_SIZE / float(FIXED_SAMPLE_RATE)
REAL_TIME_STREAMING = True
ref_time_begin = None
all_tts_times = list()

# ----------------------------------------------------
# Global state
# ----------------------------------------------------
sio = socketio.Client()

# A buffer that accumulates raw bytes from the server
server_audio_buffer = bytearray()
server_audio_time = []

# A buffer that stores exactly what was "played" in real time (including silence)
final_received_data = bytearray()

# A lock so we can safely read/write server_audio_buffer from different threads
audio_buffer_lock = threading.Lock()

# A flag to stop the real-time streaming thread
stop_streaming = False

# ----------------------------------------------------
# Socket.IO event handlers
# ----------------------------------------------------
@sio.event
def connect():
    print("Connected to server.")

@sio.event
def disconnect():
    if len(server_audio_time) > 0:
        now = datetime.now() - ref_time_begin
        start_time = server_audio_time[0] - ref_time_begin
        all_tts_times.append({'start': int(start_time.total_seconds() * 16000), 'end': int(now.total_seconds() * 16000)})
    print("Disconnected from server.")

@sio.on('audio')
def on_audio(data):
    """
    Called when the server sends audio data.
    'data' should be raw binary bytes for int16 samples, or you might need to decode JSON
    if your server is sending a JSON structure.

    We simply append the raw bytes to server_audio_buffer so the streaming thread
    can read them in CHUNK_SIZE * 2 increments.
    """
    with audio_buffer_lock:
        if len(server_audio_buffer) == 0:
            server_audio_time.append(datetime.now())
        server_audio_buffer.extend(data)
    # print(f"Received {len(data)} bytes of audio from server. Buffer size now = {len(server_audio_buffer)} bytes.")

@sio.on('stop_tts')
def on_stop_tts():
    """
    When the server indicates TTS should stop, we clear any pending audio data
    so no more old data is "played".
    """
    global server_audio_time
    global ref_time_begin
    global all_tts_times
    print("Server says: Stop TTS. Clearing server_audio_buffer.")
    with audio_buffer_lock:
        if len(server_audio_time) > 0:
            now = datetime.now() - ref_time_begin
            start_time = server_audio_time[0] - ref_time_begin
            all_tts_times.append({'start': int(start_time.total_seconds() * 16000), 'end': int(now.total_seconds() * 16000)})
            print("###################### Start time {}, end time {}".format(server_audio_time[0], now))
            server_audio_time = []
        server_audio_buffer.clear()

@sio.on('too_many_users')
def on_too_many_users(data):
    print("Server says: Too many users connected.")

@sio.on('out_time')
def on_out_time():
    print("Server says: Connect time out.")

@sio.on('prompt_success')
def on_prompt_success(data):
    print("Server says: Prompt set success.")

# ----------------------------------------------------
# Real-time streaming simulation
# ----------------------------------------------------
def simulate_real_time_stream():
    """
    Runs in a background thread. On each iteration:
      1) Attempt to read exactly CHUNK_SIZE*BYTES_PER_SAMPLE bytes.
      2) If not enough data, substitute with silence.
      3) Append that chunk (real data or silence) to final_received_data.
      4) Sleep for the duration that chunk represents (CHUNK_SIZE / SAMPLE_RATE).
    """
    global stop_streaming

    while not stop_streaming:
        with audio_buffer_lock:
            if len(server_audio_buffer) >= CHUNK_SIZE * BYTES_PER_SAMPLE:
                # Grab one chunk from the buffer
                chunk = server_audio_buffer[:CHUNK_SIZE*BYTES_PER_SAMPLE]
                del server_audio_buffer[:CHUNK_SIZE*BYTES_PER_SAMPLE]
            else:
                # Not enough data => produce silence
                chunk = b'\x00' * (CHUNK_SIZE * BYTES_PER_SAMPLE)

        # Append this chunk (data or silence) to final buffer
        final_received_data.extend(chunk)

        # Simulate real time
        time.sleep(REAL_TIME_INTERVAL)

    print("Real-time streaming thread stopped.")

# -----------------------------------
# Main function
# -----------------------------------
def run(server_url, prompt_text, audio_file, save_path):
    global stop_streaming
    global server_audio_time
    global ref_time_begin
    global all_tts_times
    # 1) Connect to the server
    sio.connect(server_url)
    sio.emit('reset_state') 

    

    # 3) Send some prompt text to the server
    # print(f"Sending prompt text: {prompt_text}")
    # sio.emit('prompt_text', prompt_text)

    # 4) Send 'recording-started'
    audio_data, sr = sf.read(audio_file, dtype='int16')
    audio_data = audio_data.astype(np.float32)

    # If not already at FIXED_SAMPLE_RATE, resample here (you can use libraries like librosa or resampy)
    # For simplicity, assume audio_data is already at 16 kHz or you do something like:
    # audio_data = resample_if_needed(audio_data, sr, FIXED_SAMPLE_RATE)

    # Convert float samples to int16 if needed
    # (soundfile 'int16' read is actually returning int16, but let's ensure we pass int16 for clarity)
    audio_data_int16 = audio_data.astype(np.int16)

    # Send chunks to simulate real-time streaming
    total_samples = len(audio_data_int16)
    idx = 0 

    print("Emitting 'recording-started'")
    sio.emit('recording-started')
    time.sleep(2)
    # 5) Simulate sending local audio (optional). 
    #    For demonstration, we'll just sleep or do something else.
    #    If you had local audio to send in real time, you could do that here.
    # Start the background streaming thread
    playback_thread = threading.Thread(target=simulate_real_time_stream, daemon=True)
    playback_thread.start()
    
    ref_time_begin = datetime.now()
    # Let’s wait some time so we can receive server TTS or audio.
    # Adjust as needed for your scenario.
    
    while idx < total_samples:
        # Take a chunk of 4800 (or CHUNK_SIZE) samples
        chunk = audio_data_int16[idx:idx + CHUNK_SIZE]
        idx += CHUNK_SIZE

        # Convert to bytes
        chunk_bytes = chunk.tobytes()

        # Create a JSON payload similar to the JS code
        payload = {
            'sample_rate': FIXED_SAMPLE_RATE,
            'audio': list(chunk_bytes)  # or possibly base64, or raw bytes depending on server
        }
        # In the JS code, it used JSON.stringify, which means the server expects a JSON-serialized array
        # We can do similarly:
        sio.emit('audio', json.dumps(payload))

        # Draw waveform? This is purely visual in JS; not typically done in Python console.
        # We'll skip that here.

        # If we want to simulate real time, sleep the duration of the chunk
        # CHUNK_SIZE samples at 16kHz => CHUNK_SIZE / 16000 seconds
        if REAL_TIME_STREAMING:
            time.sleep(CHUNK_SIZE / float(FIXED_SAMPLE_RATE))

    # Stop "recording"
    print("Stopping recording after 10 seconds delay for the model to finish the last question...")
    time.sleep(10)
    sio.emit('recording-stopped')

    # # 7) Optionally, wait more for server to finish TTS
    # print("Waiting 2 more seconds to receive any final data from server ...")
    # time.sleep(2)

    # 8) Now we stop streaming
    stop_streaming = True
    playback_thread.join()  # Wait for background thread to finish

    # 9) Disconnect from server
    sio.disconnect()
    time.sleep(5)

    # 10) final_received_data now contains the byte stream that was "played" in real-time
    #     (including silence).
    #     If you'd like, save it to a WAV file:
    if final_received_data:
        # Convert from raw bytes to int16
        received_int16 = np.frombuffer(final_received_data, dtype=np.int16)
        # Save it
        
        sf.write(save_path, received_int16, FIXED_SAMPLE_RATE)

    print("Done.")


def main(args):

    vad_model = load_silero_vad()
    server_url = f"http://{args.host}:{args.port}"
    audio_file = args.audio_file
    
    # get the data folder name
    input_folder = os.path.basename(os.path.dirname(audio_file))
    audio_file_name = os.path.basename(audio_file)
    input_timestamps_path = os.path.join("data", input_folder, audio_file_name.replace(".wav", ".timestamps"))

    # input: data/chattts-single-round-combine-easy/conversation_1.wav
    # output: output/omni-output/chattts-single-round-combine-easy/conversation_1.wav
    output_folder = os.path.join("output", "vita-output", input_folder)
    output_audio_file = os.path.join(output_folder, audio_file_name)
    os.makedirs(output_folder, exist_ok=True)
    output_text_file = output_folder + ".txt"
    run(server_url, "Hello, you are my personal AI assistant.", audio_file, output_audio_file)
    
    if os.path.exists(input_timestamps_path):
        print(f"Using existing timestamps for {audio_file_name} adding silence")
        with open(input_timestamps_path, "r") as ts:
            input_time_stamp_original = ast.literal_eval(ts.read())
            original_timestamp = list()
            for i in input_time_stamp_original:
                original_timestamp.append({'start': int(i.get("start")), 'end': int(i.get("end"))})
    else:
        original_timestamp = silero_vad(vad_model, audio_file)
    # original_timestamp = silero_vad(vad_model, audio_file)
    
    with open(output_text_file, 'a+') as f:
        # conversation_177.wav || [3, 3, 3, 3, 0, ..., 5287, 3, 3, 3] || [' Hello', ',', ' how', ' can', ' I', ' help', ' you', '?', ... ' Okay', '.', ' Sure', ',', ' what', ' do', ' I', ' need', '?', ' Sure', ',', ' I', ' recommend', ' starting', ' with', ' a', ' basic', ' photography', ' course', ' online', ' or', ' in', ' person', '.', ' will', ' give', ' you', ' a', ' good', ' foundation'] || [{'start': 53280, 'end': 93664}, {'start': 218656, 'end': 250848}, {'start': 397856, 'end': 422368}, {'start': 565280, 'end': 597472}, {'start': 752160, 'end': 794592}] || [{'start': 7200, 'end': 36320}, {'start': 78368, 'end': 83936}, {'start': 142368, 'end': 322528}, {'start': 344096, 'end': 488928}, {'start': 573984, 'end': 655840}, {'start': 770592, 'end': 804320}, {'start': 848416, 'end': 904160}, {'start': 935456, 'end': 953600}]
        f.write(f"{audio_file_name} || [] || [] || {original_timestamp} || {all_tts_times}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("file_client")
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--audio-file", required=True, help="Input audio file path")
    args = parser.parse_args()

    main(args)