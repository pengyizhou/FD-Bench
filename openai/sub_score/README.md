# Using OpenAI's API to Score the Conversational Data

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

## Prepare the batch inference data
In this code, we use OpenAI's batch inference to speed up and reduce the cost. 

Use the `generate_all_batches.sh` script to generate the batch inference data.

The generated batch data will be named as `conversation_rounds.batch.jsonl` along with the input data.

We use the model `gpt-4o-2024-11-20` for the batch inference. 

## Batch inference
Using `4o-batch-inference.py` script, you can run the batch inference on the generated batch data. 
```bash
python 4o-batch-inference.py --file-path conversation_rounds.batch.jsonl
```
And it will generate the metadata file `conversation_rounds.batch.meta.jsonl`, which contains the file ID and the session ID.

## Check inference status
You can call `openai_batch_list.py` to list all the batch inference jobs you have submitted. 

## Get inference results
Please check the inference status first.

Once the inference is completed, you can find the results file ID in the returned information. Then, please use the file ID and call `get_files.py` to obtain the inference results.


## Example output data
We have attached the `conversation_rounds.batch.result.jsonl` for each of the example data we provided.