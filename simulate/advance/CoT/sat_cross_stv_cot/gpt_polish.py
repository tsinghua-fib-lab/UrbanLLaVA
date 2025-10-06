from openai import OpenAI
import httpx
import os
import argparse
import pandas as pd
import json
import base64
import tqdm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# API Key and Proxy settings
PROXY = "http://127.0.0.1:10190"
API_KEY_MAPPING = { 
    "siliconflow": "SiliconFlow_API_KEY",
    "DeepInfra": "DeepInfra_API_KEY",
    "OpenAI": "OpenAI_API_KEY"
}
API_URL_MAPPING = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "DeepInfra": "https://api.deepinfra.com/v1/openai",
    "OpenAI": "https://api.openai.com/v1"
}
API_TYPE = "OpenAI"
API_KEY = os.environ[API_KEY_MAPPING[API_TYPE]]
API_URL = API_URL_MAPPING[API_TYPE]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def polish_text(client, model_name, og_text):
    prompt = f'''
    Please polish the following paragraph to make it more fluent and natural.
    Please shorten the text to 2048 tokens or less, keeping the most important information like the reference answer. 
    Remove redundant or less helpful information. Only keep the most important parts that can help with the task.    
    You can make necessary changes to the text, like removing the square brackets, adding punctuation, or rephrasing the text.
    Don't change the meaning of the text.
    Only output the polished text, without any additional information or appending text.
    Here is the original text:
    {og_text}
    '''
    
    dialogs = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]

    try:
      completion = client.chat.completions.create(
          model=model_name,
          messages=dialogs,
          max_tokens=2048,
          temperature=0.1,
      )
      return completion.choices[0].message.content
    except Exception as e:
      print(e)
      return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--task', type=str, default='sat-cross-stv-cot', choices=['sat-cross-stv-cot', 'sat-address-cot'])
    args = parser.parse_args()
    
    city = args.city
    work_dir = args.work_dir
    task = args.task
    CoT_dir = work_dir + f'dev-{city}/CoT/{task}/'

    model_name = "gpt-4o-2024-08-06"
    client = OpenAI(base_url=API_URL, api_key=API_KEY, http_client=httpx.Client(proxies=PROXY))

    for dataset in ["SAT_STV_location_CoT", "SAT_STV_mapping_CoT"]:
        og_path = f"{dataset}_{city}.json"
        og_path = os.path.join(CoT_dir, og_path)

        with open(og_path, 'r') as f:
            og_data = json.load(f)

        output = []
        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = {executor.submit(polish_text, client, model_name, item['CoT']): item for item in og_data}

            for future in tqdm(futures):
                item = futures[future]
                polished_CoT = future.result()
                output.append({
                    "image": item["image"],
                    "prompt": item["prompt"],
                    "polished_CoT": polished_CoT,
                    "og_CoT": item["CoT"],
                    "reference": item["reference"]
                })

        output_path = os.path.join(CoT_dir, f'polished_{dataset}_{city}.json')

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(f"Polished {dataset} for {city} saved to {output_path}")
        print("Length of output:", len(output))

    # for zl in ['zl15', 'zl17']:
    #     og_path = f"{task}_{city}_{zl}.json"
    #     og_path = os.path.join(CoT_dir, og_path)

    #     with open(og_path, 'r') as f:
    #         og_data = json.load(f)

    #     output = []

    #     with ThreadPoolExecutor(max_workers=128) as executor:
    #         futures = {executor.submit(polish_text, client, model_name, item['CoT']): item for item in og_data}

    #         for future in tqdm(futures):
    #             item = futures[future]
    #             polished_CoT = future.result()
    #             output.append({
    #                 "img_name": item["img_name"],
    #                 "polished_CoT": polished_CoT,
    #                 "og_CoT": item["CoT"],
    #                 "description": item["description"],
    #                 "address": item["address"]
    #             })

    #     output_path = os.path.join(CoT_dir, f'polished_{task}_{city}_{zl}.json')

    #     with open(output_path, 'w') as f:
    #         json.dump(output, f, indent=4, ensure_ascii=False)
