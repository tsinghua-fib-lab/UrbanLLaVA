from openai import OpenAI  # >=1.0, test version 1.16.0
import httpx
import os
import tqdm
from tqdm import trange
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_RETRIES = 3  


PROXY = "http://127.0.0.1:10190"

API_KEY_MAPPING = {
    "siliconflow": "SiliconFlow_API_KEY",  # See：https://siliconflow.cn/models
    "DeepInfra": "DeepInfra_API_KEY",  # See：https://deepinfra.com/models
    "OpenAI": "OpenAI_API_KEY"  # See：https://openai.com/api/pricing/
}
API_URL_MAPPING = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "DeepInfra": "https://api.deepinfra.com/v1/openai",
    "OpenAI": "https://api.openai.com/v1"
}
API_TYPE = "OpenAI"
API_KEY = os.environ[API_KEY_MAPPING[API_TYPE]]
API_URL = API_URL_MAPPING[API_TYPE]

def read_txt_into_string(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data

def process_file(cnt, df, work_dir, city, zl, client, model_name):
    img_name = df.at[cnt, 'img_name'].split('.')[0]
    poi_path = work_dir + f"dev-{city}/short_clipped_results_{zl}/driving_{img_name}.txt"
    road_path = work_dir + f"dev-{city}/short_clipped_results_{zl}/pois_{img_name}.txt"

    if os.path.exists(poi_path) and os.path.exists(road_path):
        poi_text = read_txt_into_string(poi_path)
        road_text = read_txt_into_string(road_path)
        dialogs = [{
            "role": "user",
            "content": "You are provided a detailed coordinates of POIs and roads within a 256x256 pixel area. "
                       "Please merge those information including "
                       "(1) stating which road going which direction, "
                       "(2) describing which POIs are located alongside which roads, "
                       "(3) telling how the roads distributed in the region, "
                       "(4) stating the concentration of specific POIs at certain areas, etc. Below are the rules of the coordinates, "
                       "where the first coordinate represents the x-axis and the second the y-axis, with the top-left corner being [0,0]. "
                       "In the response, (1) Please do not itemize any object. (2) Attach the road coordinates with each road. "
                       "(3) The generated answer is within one single paragraph. (4) Make it natural and fluent. "
                       "The descriptions are:" + road_text + ' ' + poi_text
        }]

        for attempt in range(MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=dialogs,
                    max_tokens=1024,
                    temperature=0.3
                )
                return img_name + '.png', str(completion.choices[0].message.content)
            except Exception as e:
                print(f"Error in API request (attempt {attempt + 1}/{MAX_RETRIES}) for image {img_name}: {e}")
                if attempt + 1 == MAX_RETRIES:
                    raise

        return img_name + '.png', str(completion.choices[0].message.content)
    else:
        raise Exception(f"In function process_file, file {poi_path} or {road_path} does not exist.")
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini-2024-07-18')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    model_name = args.model_name

    if API_TYPE == "OpenAI":
        client = OpenAI(
            base_url=API_URL,
            api_key=API_KEY,
            http_client=httpx.Client(proxies=PROXY)
        )
    elif API_TYPE == "siliconflow":
        client = OpenAI(
            base_url=API_URL,
            api_key=API_KEY
        )
    elif API_TYPE == "DeepInfra":
        client = OpenAI(
            base_url=API_URL,
            api_key=API_KEY,
            http_client=httpx.Client(proxies=PROXY),
        )

    for zl in ['zl15', 'zl17']:

        input_file_path = work_dir + f"dev-{city}/SAT_interpolate_address_{city}_{zl}.csv"
        output_file_path = work_dir + f"dev-{city}/rs_osm_description_{city}_{zl}.csv"

        df = pd.read_csv(input_file_path)
        region_list = []
        text_list = []

        with ThreadPoolExecutor(max_workers=128) as executor:
            future_to_result = {executor.submit(process_file, cnt, df, work_dir, city, zl, client, model_name): cnt for cnt in range(len(df))}
            for future in tqdm.tqdm(as_completed(future_to_result), total=len(future_to_result)):
                try:
                    img_name, text = future.result()
                    if img_name and text:
                        region_list.append(img_name)
                        text_list.append(text)
                except Exception as e:
                    print(f"Error processing file: {e}")

        pd_dict = pd.DataFrame({'img_name': region_list, 'text': text_list})
        pd_dict.to_csv(output_file_path, index=False)
        print(f"rs_osm_description_{city}_{zl}.csv done!")

