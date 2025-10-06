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
    You can make any necessary changes to the text, like removing the square brackets, adding punctuation, or rephrasing the text.
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
          max_tokens=1024,
          temperature=0
      )
      return completion.choices[0].message.content
    except Exception as e:
      print(e)
      return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    
    city = args.city
    work_dir = args.work_dir

    model_name = "gpt-4o-mini-2024-07-18"
    client = OpenAI(base_url=API_URL, api_key=API_KEY, http_client=httpx.Client(proxies=PROXY))

    for zl in ["zl15", "zl17"]:
        og_path = f"stv_landmark_{city}_{zl}.json"
        with open(og_path, 'r') as f:
            og_data = json.load(f)

        output = []

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(polish_text, client, model_name, item['CoT']): item for item in og_data}

            for future in tqdm(futures):
                item = futures[future]
                polished_CoT = future.result()
                output.append({
                    "image_name": item['image_name'],
                    "near_pois": item['near_pois'],
                    "landmark": item['landmark'],
                    "CoT": item['CoT'],
                    "polished_CoT": polished_CoT
                })

        with open(f"polished_stv_landmark_{city}_{zl}.json", 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
