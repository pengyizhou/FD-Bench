# Moshi Module for FD-Bench

This directory contains a modified version of the Moshi text-to-speech and speech-to-text system, adapted for benchmarking purposes in the FD-Bench framework.

## Overview

The Moshi module provides a client-server architecture for audio processing:

- **Server**: Processes audio input and generates responses using the Moshi and Mimi models
- **Client**: Handles file-based audio I/O, simulates real-time audio streaming, and logs timestamps and tokens

## Setup

### Requirements

1. Python 3.8+
2. PyTorch
3. Dependencies from the original Moshi repository
4. Additional requirements:
   - silero-vad (for voice activity detection)
   - soundfile
   - aiohttp
   - sphn (for Opus audio encoding/decoding)

### Model Setup

The server requires:
- Moshi model weights (`model.safetensors`)
- Mimi model weights (`tokenizer-e351c8d8-checkpoint125.safetensors`)
- Text tokenizer (`tokenizer_spm_32k_3.model`)

These should be placed in the default path (`../kyutai/moshiko-pytorch-bf16/`) or specified via command-line arguments.

## Usage

### Starting the Server

```bash
python ./moshi/server.py --port 8998 --host localhost
```

Parameters:
- `--port`: Port number (default: 8998)
- `--host`: Hostname (default: localhost)
- `--device`: Computing device (default: cuda)
- `--moshi-weight`: Path to Moshi model weights
- `--mimi-weight`: Path to Mimi model weights
- `--tokenizer`: Path to text tokenizer

### Running the Client

```bash
python ./moshi/client_audiofile.py --port 8998 --audio-folder "audio-folder-name"
```

Parameters:
- `--port`: Port number matching the server
- `--host`: Hostname (default: localhost)
- `--audio-folder`: Space-separated list of audio folders to process

### Batch Processing

The provided script `CS-inference.sh` demonstrates how to process multiple audio folders in parallel:

```bash
./CS-inference.sh
```

This script:
1. Creates necessary output directories
2. Runs multiple client instances in parallel on different ports
3. Processes different sets of audio folders on each client
4. Logs output to separate log files

## Data Structure

The client processes audio files and generates:

1. Input audio with added silence padding
2. Output audio from the Moshi model
3. Text files containing:
   - All tokens generated
   - Text tokens
   - Voice activity detection timestamps for both input and output

## Output Format

The output is saved in a text file with the following format:

```
conversation.wav || [token_ids] || [text_tokens] || [input_timestamps] || [output_timestamps]
```

Where:
- `token_ids`: List of all token IDs generated
- `text_tokens`: List of text tokens
- `input_timestamps`: List of speech segments in the input audio
- `output_timestamps`: List of speech segments in the output audio

## Data Directories

- Input data: `./data/[audio_folder]`
- Output data: `./data/S-C-VAD0.5/moshi_output/[audio_folder]`
- Processed input: `./data/S-C-VAD0.5/moshi_input/[audio_folder]`

## License

The original Moshi code is licensed under the license found in the LICENSE file in the root directory of this source tree.
