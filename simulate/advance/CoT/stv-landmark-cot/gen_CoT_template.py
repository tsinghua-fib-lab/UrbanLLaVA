import json
import argparse
import pandas as pd
from tqdm import trange
import os
import httpx
import base64
from openai import OpenAI # >=1.0, test version 1.16.0


PROXY = "http://127.0.0.1:10190"

API_KEY_MAPPING = { 
  "siliconflow": "SiliconFlow_API_KEY", # See：https://siliconflow.cn/models
  "DeepInfra": "DeepInfra_API_KEY", # See https://deepinfra.com/models
  "OpenAI": "OpenAI_API_KEY" # See：https://openai.com/api/pricing/
}
API_URL_MAPPING = {
  "siliconflow": "https://api.siliconflow.cn/v1",
  "DeepInfra": "https://api.deepinfra.com/v1/openai",
  "OpenAI": "https://api.openai.com/v1"
}
API_TYPE = "OpenAI"
API_KEY = os.environ[API_KEY_MAPPING[API_TYPE]]
API_URL = API_URL_MAPPING[API_TYPE]


# Generate CoT ground truth
# Three reasoning steps:
# 1. Tell the city name
# 2. Extract the location's pois around
# 3. Tell the location's address


def stv_prompt_template(city_name:str, near_feature:str, address:str, landmark:str)->str:
    """
    Generate the prompt for the street view task
    """
    prompt = f"""
    According to the street view image, this is a street view image in {city_name}.
    More specifically, this is a probable street view image in {address}.
    Around this region, there are some features: {near_feature}.
    Based on my observation and knowledge about this region, the landmark shown in the image is {landmark}.
    """
    prompt = str(prompt).replace('\n', ' ').strip()

    return prompt

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def query_landmark(image_url:str, client:OpenAI, near_pois:str)->str:
    """
    Query the landmark from the image
    """
    # encode the image
    base64_image = encode_image(image_url)
    # prompt
    prompt ='''
    You are given one street view image and the nearest 10 pois. The nearest pois are ''' + str(near_pois) + '''. \
        Based on the given pois, could you see any landmark in the given image? The landmark is an place or object that is easily recognizable. \
        If yes, please generate the name of the landmark. And illustrate how the near pois related to this landmark. \
        If you do not recognize any landmark, just produce one word "None". \
            Please generate only one paragraph and generate no more than 512 words. \
            Please do not itemize any object.
    '''         
    dialogs = [{
        "role": "user",
        "content": [
        {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
            }
        ]
    }]

    # API call
    response = client.chat.completions.create(
        model=model_name,
        messages=dialogs,
        max_tokens=1024,
        temperature=0
    )
    return response.choices[0].message.content.strip()
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir      


    model_name = "gpt-4o-mini-2024-07-18"  
    if API_TYPE == "OpenAI":
      model_name = "gpt-4o-mini-2024-07-18"
      client = OpenAI(
      base_url=API_URL,
      api_key=API_KEY,
      http_client=httpx.Client(proxies=PROXY)
    )
    elif API_TYPE == "siliconflow":
      # Model List：https://docs.siliconflow.cn/docs/model-names Docs：https://docs.siliconflow.cn/docs/4-api%E8%B0%83%E7%94%A8
      client = OpenAI(
      base_url=API_URL,
      api_key=API_KEY
    )
    elif API_TYPE=="DeepInfra":
      # Docs：https://deepinfra.com/docs  Models：https://deepinfra.com/models
      model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
      client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY,
        http_client=httpx.Client(proxies=PROXY),
    )  

    work_dir = work_dir + f'dev-{city}/'

    CITY_NAME = city

    # Satellite view
    for zl in ["zl15", "zl17"]:
        df = pd.read_csv(work_dir + f'stv_in_sat_nearest_features_update_{city}_{zl}.csv')
        adderss_df = pd.read_csv(work_dir + f'stv_in_sat_address_deploy_{zl}.csv')
        output_file = f"stv_landmark_cot_{city}_{zl}.json"
        output = []
        if os.path.exists(output_file):
            os.remove(output_file)
            print("File Removed!")

        df = df[:100]

        for cnt in trange(len(df)):
            # TODO: Change the image_dir to your own image directory
            image_dir = f"....../ThreeCityImage/{city}/StreetView"
            image_name = df.at[cnt,'image_name']
            image_url = os.path.join(image_dir, image_name)

            near_pois = df.at[cnt,'feature_names']
            landmark = query_landmark(image_url, client, near_pois)
            address = adderss_df[adderss_df['image_name'] == image_name]['adr']

            prompt = stv_prompt_template(CITY_NAME, near_pois, address, landmark)
            # API call
            # print(VQA_GPT(prompt, base64_image))

            output.append({
                "image_name": image_name,
                "CoT": prompt,
                "near_pois": near_pois,
                "landmark": landmark    
            })

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
            print(f"Saved to {output_file}")

