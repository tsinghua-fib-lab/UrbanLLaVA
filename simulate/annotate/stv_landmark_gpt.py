import os
import pandas as pd
import json
from tqdm import tqdm, trange
import argparse
import httpx
import base64
from openai import OpenAI  # >=1.0, test version 1.16.0
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

def process_row(client, df, cnt, image_dir, model_name, working_dir):
    image_name = df.at[cnt, 'image_name']
    image_url = os.path.join(image_dir, image_name)
    if not os.path.exists(image_url):
        return None

    base64_image = encode_image(image_url)
    near_pois = df.at[cnt, 'feature_names']

    prompt = '''
    You are given one street view image and the nearest 10 pois as background information. 
    The nearest pois are ''' + str(near_pois) + '''. 
    Based on the given pois and the image, please use LESS THAN FIVE WORDS to describe what the landmark in the image is.
    A landmark is a recognizable natural or artificial feature used for navigation, for example, a building, a statue, a bridge, etc.
    Please give the name of the landmark and illustrate the landmark if possible. For example, "Eiffel Tower" and "a tall iron tower".
    Keep your response short and concise, USE LESS THAN FIVE WORDS to describe the landmark.
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
            max_tokens=2048,
            temperature=0
        )

        return {
            "img_name": image_url,
            "text": completion.choices[0].message.content.strip()
        }
    except Exception as e:
        print(f"Error processing row {cnt}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini-2024-07-18')
    args = parser.parse_args()

    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f'dev-{city}/'
    model_name = args.model_name

    if API_TYPE == "OpenAI":
        client = OpenAI(base_url=API_URL, api_key=API_KEY, http_client=httpx.Client(proxies=PROXY))
    elif API_TYPE == "siliconflow":
        client = OpenAI(base_url=API_URL, api_key=API_KEY)
    elif API_TYPE == "DeepInfra":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        client = OpenAI(base_url=API_URL, api_key=API_KEY, http_client=httpx.Client(proxies=PROXY))

    # TODO: Change the following path to the actual path
    image_dir = f"....../ThreeCityImage/{city}/StreetView"

    if os .path.exists(working_dir + "stv_poi_landmark_update.jsonl"):
        os.remove(working_dir + "stv_poi_landmark_update.jsonl")
    print(f"{working_dir}stv_poi_landmark_update.jsonl File Removed!")

    for zl in ['zl15', 'zl17']:
        df = pd.read_csv(working_dir + f'stv_in_sat_nearest_features_update_{city}_{zl}.csv')
        output_file = working_dir + "stv_poi_landmark_update.jsonl"

        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = {executor.submit(process_row, client, df, cnt, image_dir, model_name, working_dir): cnt for cnt in range(len(df))}
            
            with open(output_file, "a") as fout:
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

