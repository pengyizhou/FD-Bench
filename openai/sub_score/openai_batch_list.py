#!/usr/bin/env python3

from openai import OpenAI
import sys
import ast
import json
client = OpenAI()

# batches = client.batches.list()

input_file = sys.argv[1]

with open(input_file, 'r') as f:
    lines = f.readlines()
    # print(lines[-1])
    line = json.loads(lines[-1])
    batch_id = line['id']

output_file = input_file.replace(".meta.jsonl", ".result.jsonl")
batch = client.batches.retrieve(batch_id)

if batch.status == "completed":
    content = client.files.content(batch.output_file_id)
    content.write_to_file(output_file)

