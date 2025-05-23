from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets
import time
import json
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

valid_dataset_path = "Data/split/valid"
dataset_path = "rabuahmad/climatecheck"
model_path = "Qwen/Qwen2.5-32B-Instruct"
test_output_path = "Data/synthetic/test_query_augmentation.json"
valid_output_path = "Data/synthetic/valid_query_augmentation.json"

load_dotenv(dotenv_path='code/.env')
api_key = os.getenv("gemini_API_KEY")
client = genai.Client(api_key=api_key)

# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, max_length=30000)
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

climate_dataset = datasets.load_dataset(dataset_path)
climate_dataset_train = climate_dataset['train']
climate_dataset_test = climate_dataset['test']
climate_dataset_valid = datasets.load_from_disk(valid_dataset_path)

supports_climate_dataset = climate_dataset_train.filter(lambda example: example["annotation"] == "Supports")
refutes_climate_dataset = climate_dataset_train.filter(lambda example: example["annotation"] == "Refutes")
sample1 = supports_climate_dataset.shuffle(seed=42).select(range(2))
sample2 = refutes_climate_dataset.shuffle(seed=42).select(range(2))

def infer_with_llm(tokenizer, model, prompt):
    fewshot_example = "\n".join(["First example:",
    f"claim: {sample1[0]['claim']}",
    f"abstract: {sample1[0]['abstract']}",
    "Second example:",
    f"claim: {sample1[1]['claim']}",
    f"abstract: {sample1[1]['abstract']}",
    "Third example:",
    f"claim: {sample2[0]['claim']}",
    f"abstract: {sample2[0]['abstract']}",
    "Fourth example:",
    f"claim: {sample2[1]['claim']}",
    f"abstract: {sample2[1]['abstract']}"])
    messages = [
        {
            'role': 'system',
            'content': "\n".join([
                "You are a helpful assisstant. You are given a social media claim which is used to retrive the most relevant abstract to the claim from a corpus of paper abstracts related to climate sciences, you should generate some keywords to help in the search for the related abstract.",
                "Here are some examples of a claim and a related abstract:",
                f"{fewshot_example}"
            ])
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]
    chat_template_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized_prompt = tokenizer(chat_template_prompt, return_tensors='pt').to('cuda')
    start = time.time()
    response = model.generate(**tokenized_prompt, max_new_tokens=30000)
    # print(response)
    decoded_response = tokenizer.batch_decode(response)[0]
    end = time.time()
    # print(f"The time taken for generation is {end - start} seconds")
    return decoded_response



def infer_gemini(prompt):
    fewshot_example = "\n".join(["First example:",
    f"claim: {sample1[0]['claim']}",
    f"abstract: {sample1[0]['abstract']}",
    "Second example:",
    f"claim: {sample1[1]['claim']}",
    f"abstract: {sample1[1]['abstract']}",
    "Third example:",
    f"claim: {sample2[0]['claim']}",
    f"abstract: {sample2[0]['abstract']}",
    "Fourth example:",
    f"claim: {sample2[1]['claim']}",
    f"abstract: {sample2[1]['abstract']}"])
    messages = [
        {
            'role': 'system',
            'content': "\n".join([
                "You are a helpful assisstant. You are given a social media claim which is used to retrive the most relevant abstract to the claim from a corpus of paper abstracts related to climate sciences, you should generate some keywords to help in the search for the related abstract.",
                "Here are some examples of a claim and a related abstract:",
                f"{fewshot_example}"
            ])
        }
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=messages[0]['content']),
        contents=[prompt]
    )
    time.sleep(5)
    return response.text

def parse_response(response):
    parsed_response = response.split('<|im_start|>assistant')[-1]
    return parsed_response

claims_dicts = []
for element in tqdm(climate_dataset_test):
    prompt = f"""Here is a claim from social media about climate and climatechange:
{element['claim']}
Generate some key words that can help retrieve relevant abstracts. generate only the keywords without formatting, DO NOT generate anything else. Make sure to generate in English only.
"""
    response = infer_gemini(prompt)
    # parsed_response = parse_response(response)
    claim_dict = {
        "claim_id": element['claim_id'],
        "claim": element['claim'],
        "keywords": response
    }
    claims_dicts.append(claim_dict)
    # print(parsed_response)
# Save the claims_dicts to a file
with open(test_output_path, 'w') as f:
    f.write(json.dumps(claims_dicts, indent=4))

claims_dicts = []   
visited = set()
for element in tqdm(climate_dataset_valid):
    if element['claim_id'] in visited:
        continue
    visited.add(element['claim_id'])
    prompt = f"""Here is a claim from social media about climate and climatechange:
{element['claim']}
Generate some key words that can help retrieve relevant abstracts. generate only the keywords without formatting, DO NOT generate anything else. Make sure to generate in English only.
"""
    response = infer_gemini(prompt)
    # parsed_response = parse_response(response)
    claim_dict = {
        "claim_id": element['claim_id'],
        "claim": element['claim'],
        "keywords": response
    }
    claims_dicts.append(claim_dict)
    # print(parsed_response)
# Save the claims_dicts to a file
with open(valid_output_path, 'w') as f:
    f.write(json.dumps(claims_dicts, indent=4))
