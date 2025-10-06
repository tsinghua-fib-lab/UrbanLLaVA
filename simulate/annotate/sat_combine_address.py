from openai import OpenAI  # >=1.0, test version 1.16.0
import httpx
import os
import pandas as pd
import argparse
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

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

def process_chunk(client, df, i, interpolate_num, model_name):
    string_adr = ''
    for j in range(i, i + interpolate_num):
        string_adr += str(df.at[j, 'adr']) + ', '

    prompts = f'''
    I give you a detailed address description of a square area. 
    The square area is evenly divided into a {int(interpolate_num**0.5)}*{int(interpolate_num**0.5)} grid. 
    Starting from the upper left corner and going down is the first column, 
    and then the second column continues from top to bottom, from grid 0 to grid {interpolate_num-1}. 
    The detailed addresses of grid 0 to grid {interpolate_num-1} are: {string_adr}
    Please form a general description of this area. Please include where are the east, west, north and south of this area, 
    where are the relative locations of the pois and roads in the area, 
    which POIs are adjacent, which road connects which pois in this area, etc. 
    Avoid including words like grid or poi in your answer and only generate one paragraph. Please make it natural and fluent.
    '''

    dialogs = [{
        "role": "user",
        "content": prompts
    }]

    completion = client.chat.completions.create(
        model=model_name,
        messages=dialogs,
        max_tokens=1024,
        temperature=0.3,
    )

    return df.at[i, 'img_name'], completion.choices[0].message.content.strip()

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
        if zl == 'zl15':
            interpolate_num = 5 * 5
        else:
            interpolate_num = 3 * 3

        input_file_path = work_dir + f"dev-{city}/SAT_interpolate_address_{city}_{zl}.csv"
        output_file_path = work_dir + f"dev-{city}/sat_address_combined_{city}_{zl}.csv"

        df = pd.read_csv(input_file_path)
        region_list = []
        combined_adr_list = []

        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = [
                executor.submit(process_chunk, client, df, i, interpolate_num, model_name)
                for i in range(0, len(df), interpolate_num)
            ]

            for future in tqdm(futures, total=len(futures)):
                region, combined_adr = future.result()
                region_list.append(region)
                combined_adr_list.append(combined_adr)

        pd_dict = pd.DataFrame({'img_name': region_list, 'combined_adr': combined_adr_list})
        pd_dict.to_csv(output_file_path, index=False)
