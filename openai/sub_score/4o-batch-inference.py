#!/usr/bin/env python3

from openai import OpenAI
import argparse

def upload_file(file_path, client):
    file_client = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    # {
    # "id": "file-abc123",
    # "object": "file",
    # "bytes": 120000,
    # "created_at": 1677610602,
    # "filename": "mydata.jsonl",
    # "purpose": "batch",
    # }
    return file_client

def create_batch(file_id, client):
    batch_client = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch_client

def main():
    parser = argparse.ArgumentParser(description='OpenAI API')
    parser.add_argument('--file-path', type=str, help='File path')
    args = parser.parse_args()
    client = OpenAI()
    meta_data_path = args.file_path.replace('.jsonl', '.meta.jsonl')
    file_client = upload_file(args.file_path, client)
    file_id = file_client.id
    response = create_batch(file_id, client)
    with open(meta_data_path, 'w') as f:
        f.write(file_client.model_dump_json())
        f.write("\n")
        f.write(response.model_dump_json())
        f.write("\n")
        

if __name__ == "__main__":
    main()