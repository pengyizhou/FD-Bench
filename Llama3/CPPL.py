#!/usr/bin/env python3

import torch
import math
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import ast


# Define the context (prompt) and the target continuation
def eval_ppl(tokenizer=None, model=None, input_file=None, output_file=None):
    with open(input_file, "r") as f:
        data = f.read()
        data_dict = ast.literal_eval(data)
        output_data_dict = dict()
        for key, value in data_dict.items():
            conv_id = key
            score_dict = dict()
            for round, content in value.items():
                type = content.get('Type')
                context = content.get('Content')
                if context.endswith("AI: . "):
                    score_dict[round] = {"CPPL": "N/A", "Content": context, "Type": type}
                else:
                    context = context.replace("User: ", "|").replace("AI: ", "|")
                    content = "".join(context.split("|")[:-1]).replace(". .", "")
                    response = context.split("|")[-1].replace(". .", "")
                    

                    # Concatenate context and continuation
                    full_text = content + response

                    # Tokenize the full input and the context separately
                    inputs = tokenizer(full_text, return_tensors="pt")
                    content_inputs = tokenizer(content, return_tensors="pt")

                    input_ids = inputs.input_ids  # shape: (1, sequence_length)
                    content_length = content_inputs.input_ids.shape[1]

                    # Prepare the labels for loss calculation.
                    # We want to compute loss only on the continuation tokens.
                    # For the context tokens, we set the label to -100 so that they are ignored.
                    labels = input_ids.clone()
                    labels[:, :content_length] = -100  # ignore the context tokens

                    # Optionally, move tensors to GPU if available:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # model.to(device)
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    # Compute the loss with no gradient calculation
                    with torch.no_grad():
                        outputs = model(input_ids, labels=labels)
                    # The model returns the average loss over the unmasked tokens (i.e., continuation tokens)
                        loss = outputs.loss

                    # Calculate perplexity: perplexity = exp(loss)
                    perplexity = math.exp(loss.item())
                    score_dict[round] = {"CPPL": perplexity, "Content": context, "Type": type}
                    print(perplexity)
            output_data_dict[conv_id] = score_dict
        
    with open(output_file, 'w') as f:
        f.write(str(output_data_dict))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    args = parser.parse_args()
    model_id = snapshot_download("LLM-Research/Llama-3.3-70B-Instruct")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
            )
    model.eval()  # set model to evaluation mode
    output_file_path = args.input_file.replace(".txt", ".cppl.txt")
    eval_ppl(tokenizer, model, args.input_file, output_file_path)
    # eval_ppl(None, None, args.input_file, output_file_path)
    
if __name__ == "__main__":
    main()