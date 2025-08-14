# Freeze-Omni Module for FD-Bench

This directory contains the Freeze-Omni module, which is used for audio processing and language model integration in the FD-Bench framework.

## Overview

Freeze-Omni provides a server-client architecture for processing audio data with the following components:

- **Server**: Runs a language model (Qwen2-7B-Instruct) for audio processing
- **Client**: Processes audio files in specified directories and sends them to the server

## Setup

### Requirements

1. Python 3.x
2. Qwen2-7B-Instruct model
3. Required Python packages (see requirements.txt if available)

### Model Setup

The server requires:
- Freeze-Omni checkpoints (in `./checkpoints` directory)
- Qwen2-7B-Instruct model (in `./Qwen2-7B-Instruct` directory)

Ensure these models are downloaded and placed in the correct directories before running the server.

## Usage

### Starting the Server

```bash
./freeze-omni/server.sh <port_number>
```

The server script sets up the following parameters:
- `--ip`: Server IP address (default: 127.0.0.1)
- `--port`: Port number specified as the first argument
- `--max_users`: Maximum number of concurrent users (default: 50)
- `--llm_exec_nums`: Number of LLM execution instances (default: 1)
- `--timeout`: Request timeout in seconds (default: 80)
- `--model_path`: Path to model checkpoints (default: ./checkpoints)
- `--llm_path`: Path to language model (default: ./Qwen2-7B-Instruct)
- `--top_p`: Top-p sampling parameter (default: 0.8)
- `--top_k`: Top-k sampling parameter (default: 20)
- `--temperature`: Sampling temperature (default: 0.8)

### Running Parallel Inference

The `inference_parallel.sh` script demonstrates how to process multiple audio directories in parallel:

```bash
./inference_parallel.sh
```

This script:
1. Starts client processes using SLURM batch jobs for parallel processing
2. Each client processes a specific audio directory
3. Output is logged to separate log files

Example from the script:
```bash
# Start client for clean audio on port 8086
sbatch -o ./log/noise_client_8086.log -c 5 ./freeze-omni/start_clients_single.sh "data/cosyvoice2-single-round-combine-easy" 8086

# Start client for noisy audio on port 8087
sbatch -o ./log/noise_client_8087.log -c 5 ./freeze-omni/start_clients_single.sh "data/cosyvoice2-single-round-combine-easy-noisy-gap-0dB" 8087
```

### Client Configuration

Clients are started using the `start_clients_single.sh` script with parameters:
1. Path to the audio data directory
2. Port number matching a running server instance

## Data Organization

- Input data: `data/<audio_folder>`
- Logs: `./log/noise_client_<port>.log`

## Batch Processing

For processing multiple audio folders, you can extend the `inference_parallel.sh` script by adding more client instances with different ports and data directories.

## Advanced Usage

For advanced configurations, you can modify the server parameters in `server.sh`:

- Adjust model parameters (`top_p`, `top_k`, `temperature`) to control the language model's output
- Change `max_users` and `llm_exec_nums` to optimize for your hardware
- Modify `timeout` to accommodate longer audio processing needs

## Integration with FD-Bench

This module is part of the larger FD-Bench framework for evaluating audio generation systems. The data processed by Freeze-Omni can be used for further analysis and comparison with other models.

## Troubleshooting

- Ensure all model files are correctly downloaded and placed in the expected directories
- Check log files in the `./log` directory for error messages
- Verify that ports used for the server instances are not already in use
