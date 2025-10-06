# This script is used to convert a address QA into a multi-choice question for evaluation.

import os
import argparse
import pandas as pd

from tqdm import tqdm
import json

from config import MULTI_IMAGE_FOLDER
from serving.vlm_serving import VLMWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--task_name', type=str, default='STV_compare', help='task name')
    parser.add_argument('--data_name', type=str, default='all', help='data name', choices=["all", "mini"])
    args = parser.parse_args() 

    model_name = args.model_name
    city_name = args.city_name
    task_name = args.task_name   

    print("Load the model")
    model_wrapper = VLMWrapper(args.model_name)
    model = model_wrapper.get_vlm_model()
    

    print("Load the image list")
    # path = os.path.join(f"./{task_name}/{city_name}", f"{task_name}_{city_name}_{zl}.json")
    path = os.path.join(MULTI_IMAGE_FOLDER, task_name, city_name, f"{city_name}_{task_name}_test.json")
    with open(path, "r") as f:
        data = json.load(f)

    if args.data_name == "mini":
        data = data[:10]

    response = []
    for d in tqdm(data):
        prompt = d["prompt"]
        reference = d["reference"]
        img_path = d["image"]

        ret = model.generate(img_path + [prompt])
        response.append({
            "image": img_path,
            "prompt": prompt,
            "reference": reference,
            "response": ret
        })

    print("Save the response")
    output_path = os.path.join(MULTI_IMAGE_FOLDER, task_name, city_name, model_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{city_name}_{task_name}_response.json"), "w") as f:
        json.dump(response, f, indent=4, ensure_ascii=False)

    model_wrapper.clean_proxy()