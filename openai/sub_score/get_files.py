#!/usr/bin/env python3

from openai import OpenAI
client = OpenAI()

content = client.files.content("your_file_id_here")  # Replace with your actual file ID

content.write_to_file("./log/fail.log")