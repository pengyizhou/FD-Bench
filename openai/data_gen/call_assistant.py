#!/usr/bin/env python

from openai import OpenAI

import json
from concurrent.futures import ThreadPoolExecutor


client = OpenAI()

assistant_id = "your_assistant_id_here"  # Replace with your actual assistant ID

# each thread only has a 10 round conversation
# and we will run for 10 threads

def run_one_thread(index):
    output_folder = "./output_data"
    output_file = f"{output_folder}/output_{index}.json"
    output_original_file = f"{output_folder}/output_original_{index}.json"
    message_thread = client.beta.threads.create()
    print(message_thread)

    thread_message = client.beta.threads.messages.create(
        message_thread.id,
        role="user",
        content="Please give me a simulated conversation",
    )
    # print(thread_message)

    run = client.beta.threads.runs.create_and_poll(
        thread_id=message_thread.id,
        assistant_id=assistant_id
    )

    # To check if the message meets my requirements
    thread_messages_check = client.beta.threads.messages.list(message_thread.id)
    thread_messages_check = thread_messages_check.model_dump().get("data", [])
    if thread_messages_check:
        for message in thread_messages_check:
            if message.get("role") == "assistant":
                # response_message = json.loads(message.get("content").get("text").get('value').split("```")[1].replace("\n", "")[4:])
                response_message = message.get("content")
                try:
                    content = json.loads(response_message[0].get("text").get('value').split("```")[1].replace("\n", "")[4:])
                    print(content)
                    if content == "":
                        print("line 47 Empty response for thread: {}".format(str(message_thread.id)))
                        # exit the current thread
                        with open(output_original_file, "w") as f:
                            f.write(thread_messages.model_dump_json())
                            f.write("\n")
                        return False
                except:
                    with open(output_original_file, "w") as f:
                        f.write(thread_messages.model_dump_json())
                        f.write("\n")
                    print("line 54 Empty response for thread: {}".format(str(message_thread.id)))
                    return False
    else:
        print("Error getting empty response for thread: {}".format(str(message_thread.id)))
        return False            
                    
    # Retrieve the response message from the thread
    # thread = client.beta.threads.retrieve(thread_id=run.thread_id)
    for rd in range(49):
        thread_message = client.beta.threads.messages.create(
            message_thread.id,
            role="user",
            content="Please give me another simulated conversation",
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=message_thread.id,
            assistant_id=assistant_id
        )
        print("run for the {} round".format(rd))
        # print(client.beta.threads.messages.list(message_thread.id))
        thread_messages = client.beta.threads.messages.list(message_thread.id)
        print(thread_messages)
    thread_messages = client.beta.threads.messages.list(message_thread.id, limit=100)
    with open(output_original_file, "w") as f:
        f.write(thread_messages.model_dump_json())
        f.write("\n")

    thread_messages = thread_messages.model_dump().get("data", [])
    with open(output_file, "w") as f:
        if thread_messages:
            for message in thread_messages:
                if message.get("role") == "assistant":
                    # response_message = json.loads(message.get("content").get("text").get('value').split("```")[1].replace("\n", "")[4:])
                    response_message = message.get("content")
                    content = json.loads(response_message[0].get("text").get('value').split("```")[1].replace("\n", "")[4:])
                    f.write(str(content))
                    f.write("\n")

def main():
    # We will run each thread in parallel
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     for i in range(20):
    #        executor.submit(run_one_thread, i)
    run_one_thread(1)

if __name__ == "__main__":
    main()
