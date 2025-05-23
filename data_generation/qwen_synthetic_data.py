import os
import datasets
import pandas as pd
import dirtyjson
import json
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dataset_path = "rabuahmad/climatecheck"
abstracts_path = "Data/climatecheck_abstracts.parquet"
output_file_path = "Data/synthetic/gemini.json"

climate_dataset = datasets.load_dataset(dataset_path)
climate_dataset = climate_dataset['train']
unique_abstracts = set(climate_dataset['abstract'])

abstracts = pd.read_parquet(abstracts_path)
abstracts_set = set(abstracts['abstract'])

abstracts_to_generate = abstracts_set - unique_abstracts
abstracts_to_generate = [abstract for abstract in abstracts_to_generate if len(abstract.split()) >= 100]
print(len(abstracts_to_generate))

supports_climate_dataset = climate_dataset.filter(lambda example: example["annotation"] == "Supports")
refutes_climate_dataset = climate_dataset.filter(lambda example: example["annotation"] == "Refutes")
third_climate_dataset = climate_dataset.filter(lambda example: example["annotation"] != "Refutes" and example["annotation"] != "Supports")
sample1 = supports_climate_dataset.shuffle(seed=42).select(range(2))
sample2 = refutes_climate_dataset.shuffle(seed=42).select(range(2))
sample3 = third_climate_dataset.shuffle(seed=42).select(range(2))

few_shots = f"""Here is the first example:
The abstract: {sample1[0]['abstract']}
a supported claim: { {k: v for k, v in sample1[0].items() if k not in ["abstract", "claim_id", "abstract_id"]} }
Here is the second example:
The abstract: {sample2[0]['abstract']}
a refuted claim: { {k: v for k, v in sample2[0].items() if k not in ["abstract", "claim_id", "abstract_id"]} }
Here is the third example:
The abstract: {sample3[1]['abstract']}
a claim that can't be supported or refuted: { {k: v for k, v in sample3[1].items() if k not in ["abstract", "claim_id", "abstract_id"]} }"""

system_msg = f"""You are a helpful assisstant. You are given a scientific paper abstract about climate and climate change.
Your task is to generate three social media-style claims about climate change, the first claim can be supported and verified by the given paper abstract and the second can be refuted and disproved by the given abstract, and the third cannot be supported or refuted by the abstract because there is not enough information.
Try to make it related to the abstract as much as possible. Make sure that the claim imitates social media style not just stating the fact or claim.
Guidelines:
1-The claims should resemble real posts seen on social media — informal, engaging, and possibly opinionated or exaggerated, but still grounded in the abstract's content.
2-Do not copy or quote text directly from the abstract.
3-Do not explicitly mention the abstract or cite it.
4-Try to make the claims as contextually relevant to the abstract as possible.
Your response should be in the following json schema:
[{{'claim':'the first generated claim', 'annotation':'supports'}}, {{'claim':'the second generated claim', 'annotation':'refutes'}}, {{'claim':'the third generated claim', 'annotation':'no information'}}]
here are some examples for the task so you can understand better:
{few_shots}"""

def parse_response(response):
    parsed_response = response.split('<|im_start|>assistant')[-1]
    return parsed_response

def generate_claims(abstracts_to_generate, tokenizer, model):
    parsed_responses = []
    for abstract in tqdm(abstracts_to_generate):
        prompt = f"""Here is the abstract:
        {abstract}
        Generate three claims for the given abstract. Your response should be in the following json schema:
        [{{'claim':'the first generated claim', 'annotation':'supports'}}, {{'claim':'the second generated claim', 'annotation':'refutes'}}], {{'claim':'the third generated claim', 'annotation':'no information'}}]
        follow the schema exactly and do not add any other text or explanation."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        # todo: call model
        chat_template_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_prompt = tokenizer(chat_template_prompt, return_tensors='pt').to('cuda')
        start = time.time()
        response = model.generate(**tokenized_prompt, max_new_tokens=30000)
        decoded_response = tokenizer.batch_decode(response)[0]
        response = parse_response(decoded_response)
        try:
            cleaned_response = response.text.strip('` \n')  # remove leading/trailing backticks and whitespace
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[len("json"):].strip()
            parsed_response = dirtyjson.loads(cleaned_response)
            parsed_response[0]['abstract'] = abstract
            parsed_response[1]['abstract'] = abstract
            parsed_responses.extend(parsed_response)
        except Exception as e:
            print("parsing failed for sample!")
            print(response.text)
            # print(e)
        if len(parsed_responses) % 100 == 0:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(parsed_responses))
        time.sleep(5)
        

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(parsed_responses))


generate_claims(abstracts_to_generate)