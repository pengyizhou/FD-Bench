# Using the Llama-3.3-70B-Instruct model to calculate the perplexity (PPL) of a given conversational context

## Introduction

This toolkit provides a framework for calculating **Conditional Perplexity (CPPL)** of AI responses in conversational contexts using the Llama-3.3-70B-Instruct model. Perplexity is a fundamental metric in natural language processing that measures how well a language model predicts a sequence of words.

### What is Perplexity?

Perplexity (PPL) quantifies the uncertainty of a language model when predicting the next token in a sequence. Mathematically, it's defined as:

```
PPL = exp(H) = exp(-1/N * Σ log P(w_i | w_1, ..., w_{i-1}))
```

Where:
- `H` is the cross-entropy loss
- `N` is the number of tokens being evaluated
- `P(w_i | w_1, ..., w_{i-1})` is the probability of token `w_i` given the previous context

**Lower perplexity indicates better predictability** - the model is more confident about the next tokens.

### Conditional Perplexity (CPPL) in Conversations

In this implementation, we calculate **Conditional Perplexity** specifically for AI responses given the conversational context:

1. **Context Separation**: We separate the user-AI conversation history (context) from the target AI response
2. **Masked Loss Calculation**: The model computes loss only on the AI response tokens, while context tokens are masked (set to -100)
3. **CPPL Computation**: Perplexity is calculated as `exp(loss)` where loss is averaged over the AI response tokens only

### Purpose and Applications

This CPPL calculation serves several important purposes:

- **Response Quality Assessment**: Lower CPPL suggests the AI response is more predictable/coherent given the conversation context
- **Model Evaluation**: Compare how different models perform on the same conversational data
- **Dialogue Coherence**: Measure how well AI responses align with conversational flow and context
- **Benchmarking**: Provide quantitative metrics for conversational AI systems across different scenarios

The toolkit is particularly useful for evaluating conversational AI systems in the FD-Bench framework, where consistent and contextually appropriate responses are crucial for effective human-AI interaction.

## Environment Setup
To run the scripts in this directory, you need to set up your environment with the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Hardware Requirements
This script requires significant computational resources:
- **GPU**: At least one GPU with sufficient VRAM (recommended: 24GB+ VRAM)
- **Storage**: Sufficient space for the Llama-3.3-70B-Instruct model (BF16 format, approximately 40GB)

## Model Setup
The script automatically downloads the Llama-3.3-70B-Instruct model from ModelScope:
- **Model ID**: `LLM-Research/Llama-3.3-70B-Instruct`
- **Quantization**: 8-bit quantization is used to reduce memory requirements
- **Device Mapping**: Automatic device mapping across available GPUs

## Calculate Perplexity
You can calculate perplexity for conversational data using the `CPPL.py` script. This script will:
1. Load the Llama-3.3-70B-Instruct model with 8-bit quantization
2. Process conversational data from input files
3. Calculate conditional perplexity (CPPL) for AI responses given the conversation context
4. Output results to a corresponding `.cppl.txt` file

### Single File Usage
```bash
python CPPL.py --input_file path/to/your/conversation_rounds.txt
```

### Batch Processing
Use the provided shell script to process multiple files across different data folders:
```bash
bash CPPL.sh
```

This script will:
- Search for `conversation_rounds.txt` files in the data directories
- Submit SLURM jobs for each file (requires SLURM cluster environment)
- Use 1 GPU per job (you may adjust CPU numbers per your cluster configuration)

## Input Data Format
The input file should contain a Python dictionary structure where:
- Keys are conversation IDs
- Values contain rounds of conversation with 'Type' and 'Content' fields
- Content follows the format: "User: [message] AI: [response]"

## Output Format
The script generates output files with `.cppl.txt` extension containing:
- **CPPL**: Conditional perplexity score for AI responses
- **Content**: Original conversation content
- **Type**: Conversation type metadata
- **N/A**: Used for conversations ending with "AI: . " (incomplete responses)

