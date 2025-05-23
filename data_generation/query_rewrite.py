from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import datasets
import pandas as pd
import dirtyjson
import json
from tqdm import tqdm
import time

dataset_path = "rabuahmad/climatecheck"
output_file_path = "Data/synthetic/rewritten_query.json"
load_dotenv(dotenv_path='code/.env')
api_key1 = os.getenv("gemini_API_KEY")
api_key2 = os.getenv("gemini_API_KEY2")

climate_dataset = datasets.load_dataset(dataset_path)
climate_dataset = climate_dataset['train']
unique_abstracts = set(climate_dataset['abstract'])
claims_dataset = datasets.load_dataset(dataset_path)['test']

supports_climate_dataset = climate_dataset.filter(lambda example: example["annotation"] == "Supports")
refutes_climate_dataset = climate_dataset.filter(lambda example: example["annotation"] == "Refutes")
third_climate_dataset = climate_dataset.filter(lambda example: example["annotation"] != "Refutes" and example["annotation"] != "Supports")
sample1 = supports_climate_dataset.shuffle(seed=42).select(range(2))
sample2 = refutes_climate_dataset.shuffle(seed=42).select(range(2))
sample3 = third_climate_dataset.shuffle(seed=42).select(range(2))

few_shots = f"""Here is the first example:
The abstract: {sample1[0]['abstract']}
a supported claim: {sample1[0]['claim']}
Here is the second example:
The abstract: {sample2[0]['abstract']}
a refuted claim: {sample2[0]['claim']}"""

system_msg = f"""You are a helpful assisstant. You are given a social media post that contains a claim about climate and climate change.
Your task is to extract the claim in the post and rewrite it so that it can be easier to retrieve relevant paper abstracts that support or refute the claim and address it directly.
Your response should be in the following json schema:
{{'claim':'the generated extracted and enhanced claim'}}
here are some examples for social media posts and thier generated claims so you can understand better:
{few_shots}"""

def generate_claims(claims_dataset):
    parsed_responses = []
    client = genai.Client(api_key=api_key1)
    for item in tqdm(claims_dataset):
        prompt = "\n".join([
            f"Here is a social media post that contains a claim about climate change:",
            f"{item['claim']}",
            "extract the claim and write it in a clear way that can be used to retrieve relevant paper abstracts that support or refute the claim.",
        ])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_msg),
            contents=[prompt]
        )
        try:
            cleaned_response = response.text.strip('` \n')  # remove leading/trailing backticks and whitespace
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[len("json"):].strip()
            parsed_response = dirtyjson.loads(cleaned_response)
            parsed_responses.append({"claim_id": item['claim_id'], "original_claim": item['claim'],"claim": parsed_response['claim']})
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response.text}")
            parsed_responses.append({"claim_id": item['claim_id'], "original_claim": item['claim'],"claim": None})
        time.sleep(5)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(parsed_responses))

generate_claims(claims_dataset)