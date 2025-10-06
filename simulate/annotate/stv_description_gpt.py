from openai import OpenAI  # >=1.0, test version 1.16.0
import httpx
import os
import argparse
import pandas as pd
import json
import base64
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def generate_description(client, img_name, image_dir, model_name):
    img_url = os.path.join(image_dir, img_name)
    if not os.path.exists(img_url):
        return None
    
    base64_image = encode_image(img_url)

    prompt = '''
    Please describe in detail the given image following the principles: 
    (1) Describing object attributes, including object quantity, color, material, shape, size; 
    (2) Describing the spatial relationship between objects, including the relative position of objects, the distance between objects, and the direction of objects;
    (3) Only describe the content that has high confidently. 
    (4) Do not describe the contents by itemizing them in list form. 
    (5) Make sure the description is coherent and fluent.
    '''

    dialogs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }}
        ]
    }]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=dialogs,
            max_tokens=1024,
            temperature=0
        )
        return {
            "img_name": img_url,
            "text": completion.choices[0].message.content.strip(),
        }
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini-2024-07-18')
    args = parser.parse_args()
    
    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f"dev-{city}"
    # TODO: Change the following path to the actual path
    image_dir = f"....../ThreeCityImage/{city}/StreetView"
    model_name = args.model_name

    if API_TYPE == "OpenAI":
        client = OpenAI(base_url=API_URL, api_key=API_KEY, http_client=httpx.Client(proxies=PROXY))
    elif API_TYPE == "siliconflow":
        client = OpenAI(base_url=API_URL, api_key=API_KEY)
    elif API_TYPE == "DeepInfra":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        client = OpenAI(base_url=API_URL, api_key=API_KEY, http_client=httpx.Client(proxies=PROXY))

    img_set = []
    for zl in ['zl15', 'zl17']:
        stv_in_sat_path = f"{working_dir}/stv_in_sat_{city}_{zl}.csv"
        assert os.path.exists(stv_in_sat_path), f"{stv_in_sat_path} does not exist."
        df = pd.read_csv(stv_in_sat_path)
        img_set.extend(df['image_name'].tolist())
    
    img_set = list(set(img_set))
    assert os.path.exists(image_dir), f"{image_dir} does not exist."
    print(f"Start generating descriptions for {len(img_set)} images.")

    with ThreadPoolExecutor(max_workers=128) as executor:  
        futures = {executor.submit(generate_description, client, img_name, image_dir, model_name): img_name for img_name in img_set}
        with open(os.path.join(working_dir, "stv_description.jsonl"), "a") as fout:
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

