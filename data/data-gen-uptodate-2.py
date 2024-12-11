# %%
import os
import json
import csv
from openai import OpenAI
from multiprocessing import Lock, Manager, Pool
# import pandas as pd

# %%
# open_ai_key = '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'
open_ai_key = os.environ['OPENAI_API_KEY']

# %%
def init_files(output_file):
    """Initialize output and log files with headers if they don't exist."""
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Evidence', 'Supports', 'Contradicts', 'Ambiguous'])
            writer.writeheader()

# %%
uptodate_texts = os.path.join('.', 'uptodate')
output_file = os.path.join('.', 'csv', 'generated_claim_triplets_with_topics_uptodate.csv')
input_file = os.path.join('.', 'csv', 'categorized_content_links_unique.csv')

init_files(output_file)

# %%
# model schema
from pydantic import BaseModel, Field
from typing import Literal, List, Any

# %%
class PubHealthSchema(BaseModel):
    supporting_claim: str
    contradictory_claim: str
    ambiguous_claim: str # 0: +ve, 1: -ve, 2: ambiguous 
    
    def to_json(self):
        return {"supporting_claim": self.supporting_claim, "contradictory_claim": self.contradictory_claim, "ambiguous_claim": self.ambiguous_claim}
    
class Schema(BaseModel):
    claims: List[PubHealthSchema]
    
    def to_json(self):
        return [data.to_json() for data in self.claims]



# %%
def write_row(file_path, row_dict, file_lock):
    """Write a single row to the CSV file in a thread-safe manner."""
    with file_lock:
        file_exists = os.path.exists(file_path)
        mode = 'a' if file_exists else 'w'
        
        with open(file_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

# %%
def get_prompt(num_samples):
    prompt = f"""You are a medical expert. You will be given a medical document, generate three types of claims:\n\n1) A supporting claim that paraphrases a key assertion.\n2) A contradictory claim that directly contradicts a key evidence provided in the summary.\n3) An ambiguous claim that either partially supports or contradicts, or presents elements that are neither clearly supported nor contradicted.\n\nEach claim should be one or two sentences long. Ideally, the claims should be generated from different key assertions or sections of the summary.\n\nReturn ONLY the claims in this exact JSON format below. DO NOT include any extra text or explanations. DO NOT add ```json``` formatting. Just output the exact JSON as a string.\n[{{\n  \"supporting_claim\": '...',\n  \"contradictory_claim\": '...',\n  \"ambiguous_claim\": '...'\n}}].\n\n Output {num_samples} triplets of supporting, contradictory, and ambiguous claims from the provided summary. Give all {num_samples} triplets in the same JSON array."""
    return prompt

# %%
client = OpenAI(api_key=open_ai_key)

# %%
def parse_file(args):
    file_path, file_lock = args
    text = open(file_path, "r").read()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_prompt(20)},
            {"role": "user", "content": f"Document:\n\n{text}"}
        ],
        response_format=Schema,
    )
    responses = completion.choices[0].message.parsed.to_json()
    print(responses)
    for response in responses:
        supports = response.get("supporting_claim", "N/A")
        contradicts = response.get("contradictory_claim", "N/A")
        ambiguous = response.get("ambiguous_claim", "N/A")
        result = {
                'Evidence': text,
                'Supports': supports,
                'Contradicts': contradicts,
                'Ambiguous': ambiguous
            }
        write_row(output_file, result, file_lock)

# %%
files = []
for filename in os.listdir(uptodate_texts):
    file_path = os.path.join(uptodate_texts, filename)

    # Check if it's a file (to avoid directories)
    if os.path.isfile(file_path):
        files.append(file_path)

if __name__ == "__main__":
    manager = Manager()
    file_lock = manager.Lock()
    for file in files:
        parse_file((file, file_lock))
    
