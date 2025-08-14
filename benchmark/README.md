# FD-Bench: Full-Duplex Conversational AI Benchmarking Pipeline

## Introduction

This comprehensive benchmarking pipeline provides automated evaluation tools for full-duplex conversational AI systems. The pipeline measures both objective and subjective metrics to assess the quality and naturalness of AI-human conversations, with particular focus on real-time interaction capabilities including interruption handling, response timing, and speech quality.

### What is Full-Duplex Conversation?

Full-duplex conversation refers to the ability of AI systems to engage in natural, bidirectional communication where both parties can speak and listen simultaneously, similar to human conversation. This includes:

- **Real-time Response**: Immediate reaction to user input without waiting for complete utterances
- **Interruption Handling**: Graceful management of user interruptions and overlapping speech
- **Context Awareness**: Maintaining conversational context across multiple turns and interruptions
- **Natural Timing**: Appropriate response delays and turn-taking behavior

### Evaluation Metrics Overview

The pipeline calculates a comprehensive set of metrics divided into several categories:

#### 1. Speech Quality Metrics
- **Word Error Rate (WER%)**: Measures transcription accuracy of AI-generated speech
- **Conditional Perplexity (CPPL)**: Evaluates response predictability given conversational context

#### 2. Interaction Success Metrics
- **Success Reply Rate (SRR%)**: Percentage of appropriate responses to user queries
- **Success Interrupt Rate (SIR%)**: Percentage of successful user interruptions acknowledged
- **Success Reply to Interrupt Rate (SRIR%)**: Percentage of appropriate responses after interruptions

#### 3. Timing and Latency Metrics
- **Interrupt Response Delay (IRD%)**: Time between user interruption and AI acknowledgment
- **First Speech Emission Delay (FSED%)**: Initial response latency after user input
- **Early Reply Time (ERT)**: Response time for quick exchanges
- **Early Interrupt Time (EIT)**: Time to recognize and process interruptions

#### 4. Interruption Analysis Metrics
- **Early Interrupt Rate (EIR)**: Frequency of premature AI interruptions
- **Noise Interrupt Rate (NIR)**: False interruptions caused by background noise

### Pipeline Components

1. **Audio Processing**: Automatic speech recognition using WhisperX for transcription
2. **Voice Activity Detection (VAD)**: Identification of speech segments and silence periods
3. **Timing Analysis**: Precise measurement of response delays and interaction patterns
4. **Quality Assessment**: WER calculation and perplexity analysis
5. **Statistical Summarization**: Comprehensive metric calculation and visualization

## Environment Setup

To run the scripts in this directory, you need to set up your environment with the required dependencies:

```bash
pip install -r requirements.txt
```

## Hardware Requirements

This pipeline requires:
- **GPU**: Recommended for WhisperX ASR processing (optional but faster)
- **Memory**: Sufficient RAM for audio processing (8GB+ recommended)
- **Storage**: Space for intermediate audio files and processing results

## Pipeline Usage

Run the benchmarking pipeline on a specific conversation dataset:

```bash
python benchmarking.py path/to/Data_name.txt
```

The script will:
1. Load conversation data and ground truth comparisons
2. Perform automatic speech recognition on audio files
3. Analyze voice activity and timing patterns
4. Calculate all objective metrics
5. Generate visualization plots and summary statistics


## Input Data Format

### Expected Directory Structure
```
data/
├── MODEL-NAME/
│   └── subjective_metrics/...
|   └── objective_metrics/...
|   └── Data_name/
|   └── Data_name.txt

```

## Output Format

The pipeline generates comprehensive results in structured directories:

### Objective Metrics
```
data/MODEL-NAME/objective_metrics/Data_name/
├── metrics.scores        # All calculated metrics
```
### Visualization
```
data/MODEL-NAME/objective_metrics/Data_name/
├── interruption_success_delays.png      # Timing analysis of successful interruption handling
├── lead_times.png                       # Distribution of AI response lead times in conversations
├── lead_times_to_interruption.png       # Lead time analysis specifically for interruption events
├── response_delays.png                  # Overall response delay patterns and distributions
└── response_delays_to_interruption.png  # Response delays measured after user interruptions
```
### Speech Recognition Results (If applicable)
```
data/MODEL-NAME/Data_name_asr/
├── hyp
├── ref
└── wer
```
### Subjective Metrics
```
data/MODEL-NAME/subjective_metrics/
├── Data_name/
│   ├── conversation_round.txt  # processed user-AI conversation text rounds
│   ├── conversation_round.cppl.txt # Conditional Perplexity raw output from Llama3
│   └── conversation_round.cppl.score # CPPL scores for each conversation round
```

## Dependencies and External Tools

- **WhisperX**: For automatic speech recognition
- **PyTorch**: For deep learning model inference
- **matplotlib**: For visualization generation
- **numpy/scipy**: For numerical analysis
- **ASR Models**: Automatic download of required speech recognition models

## Integration with Other Components

This benchmarking pipeline integrates with:
- **CPPL Calculation**: Uses `../Llama3/` directory for perplexity analysis
- **Subjective Scoring**: Works with `../openai/sub_score/` for GPT-based evaluation
- **TTS Generation**: Processes outputs from `../tts-generation/` systems

## Supported Models

The pipeline includes specific configurations for:
- **VITA-1.5**: Vision and language model with conversational capabilities
- **Moshi**: Real-time conversational AI system
- **Freeze-omni**: Multimodal conversational model
- **Custom Models**: Easily adaptable for new conversational AI systems
