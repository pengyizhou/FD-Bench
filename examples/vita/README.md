# VITA Module for FD-Bench

This directory contains the VITA (Voice InTeraction with AI) module, which is used for audio processing and evaluation in the FD-Bench framework.

## Overview

VITA provides a server-client architecture for processing audio data:

- **Server**: Runs the VITA model for audio processing
- **Client**: Processes individual audio files and sends them to the server

## Setup

### Requirements

1. Python 3.x
2. VITA checkpoint model (`demo_VITA_ckpt` directory)
3. Required Python packages

### Flash Attention Installation

**Important**: VITA requires a specific version of `flash_attention` which must be compiled from source. Pre-built packages might not work correctly.

To install the required version:

```bash
# Clone the repository with the specific version
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.6.3  # Use this specific version

# Install
pip install .
pip install ./csrc/layer_norm
pip install ./csrc/rotary
```

Alternatively, you can use:

```bash
pip install flash-attn==1.0.9 --no-build-isolation
```

If you encounter CUDA-related errors, make sure your CUDA version is compatible with this specific version of flash_attention.

### Model Setup

Download the VITA checkpoint and place it in the `demo_VITA_ckpt` directory. The server expects this location by default.

## Usage

### Starting the Server

```bash
./vita/server.sh <port_number>
```

The server script runs the VITA model with the following parameters:
- `--model_path`: Path to the VITA checkpoint (default: demo_VITA_ckpt)
- `--ip`: Server IP address (default: 0.0.0.0)
- `--port`: Port number specified as the first argument

### Running the Client

For a single audio file:

```bash
./vita/web_demo/client-audio.py --port <port_number> --audio-file <path_to_audio_file>
```

### Running Parallel Inference

The `inference_parallel.sh` script demonstrates how to process multiple audio directories in parallel:

```bash
./inference_parallel.sh
```

This script:
1. Starts client processes using SLURM batch jobs for parallel processing
2. Each client processes specific audio directories
3. Output is logged to separate log files

Example from the script:
```bash
# Start client for the first set of audio data on port 8081
sbatch -o ./log/client_8081.log -c 5 ./vita/start_clients_single.sh "data/chattts-single-round-combine-easy data/chattts-single-round-combine-med" 8081

# Additional clients for other audio datasets
port=8082
sbatch -o ./log/client_$port.log -c 5 ./vita/start_clients_single.sh "data/chattts-single-round-combine-hard data/cosyvoice2-single-round-combine-easy" $port
```

## Batch Processing

The `start_clients_single.sh` script processes all WAV files in the specified directories sequentially:

```bash
./vita/start_clients_single.sh "<space_separated_audio_folders>" <port_number>
```

This script:
1. Takes a space-separated list of audio folder paths
2. For each folder, finds all WAV files
3. Processes each file individually by calling the client script

## Data Organization

- Input data: `data/<audio_folder>/*.wav`
- Logs: `./log/client_<port>.log`

## Integration with FD-Bench

VITA is part of the larger FD-Bench framework for evaluating audio generation systems. The results from VITA processing can be used for comparative analysis with other models like Freeze-Omni and Moshi.

## Troubleshooting

### Common Issues

1. **Flash Attention errors**: 
   - Ensure you're using the exact required version (1.0.9)
   - Verify CUDA compatibility with your installed version
   - Try building from source if pre-built packages fail

2. **Model loading issues**:
   - Verify the checkpoint directory structure is correct
   - Check file permissions for the model files

3. **Connection failures**:
   - Ensure the server is running on the specified port
   - Check for firewall restrictions
   - Verify network connectivity between client and server

### Debug Tips

- Check the log files in `./log/` directory for detailed error messages
- Run a single client manually to observe direct output
- Verify the server is operational using a simple test request

## Advanced Configuration

For advanced users, you can modify:
- Server parameters in `server.sh` for different IP configurations
- Client batch size and processing in `start_clients_single.sh`
- Resource allocation in SLURM batch parameters
