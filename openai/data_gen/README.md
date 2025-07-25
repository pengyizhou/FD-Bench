# Using OpenAI's API to Simulate conversational data 

## Environment Setup
To run the scripts in this directory, you need to set up your environment with the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```
## Setting up the OpenAI API Key

Please fill in your OpenAI API key in the `OPENAI_API_KEY` variable in the `path.sh` file.
Then run the following command to set the environment variable:

```bash
source path.sh
```

## Create your assistant
You can create your own assistant by following the instructions in the [OpenAI API documentation](https://platform.openai.com/docs/guides/assistants).

Please refer to the code `create_an_assistant_normal.py` for an example of how to create an assistant using the OpenAI API.

## Generate data
You can generate data using the `call_assistant.py` script. This script will use the OpenAI API to generate conversational data based on the assistant you created.

## Example output data
We have attached the all generated data in `output_all.jsonl` file. 

This file is manually checked and revised to ensure quality.