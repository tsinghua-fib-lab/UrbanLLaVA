# This script is used to convert a address QA into a multi-choice question for evaluation.

import os
import argparse
import pandas as pd
from setproctitle import setproctitle

from tqdm import tqdm
import json

from config import UNI_IMAGE_FOLDER, RESULTS_PATH
from serving.vlm_serving import VLMWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--task_name', type=str, default='sat_landuse_mc', help='task name')
    parser.add_argument('--data_name', type=str, default='all', help='data name', choices=["all", "mini"])

    args = parser.parse_args() 

    model_name = args.model_name
    city_name = args.city_name
    task_name = args.task_name   

    print("Load the model")
    model_wrapper = VLMWrapper(args.model_name)
    model = model_wrapper.get_vlm_model()
    
    for zl in ["zl15", "zl17"]:

        print("Load the image list")
        # path = os.path.join(f"./{task_name}/{city_name}", f"{task_name}_{city_name}_{zl}.json")
        path = os.path.join(UNI_IMAGE_FOLDER, task_name, city_name, f"{city_name}_{task_name}_{zl}.json")

        output_path = os.path.join(RESULTS_PATH, task_name, city_name, f"{city_name}_{task_name}_{zl}_{args.model_name}_response.json")
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(path, "r") as f:
            data = json.load(f)

        if args.data_name == "mini":
            data = data[:10]

        response = []
        for d in tqdm(data):
            prompt = d["prompt"]
            choices = d["choices"]
            reference = d["reference"]
            img_path = d["image"]
            img_name = img_path.split("/")[-1]

            assert os.path.exists(img_path), f"Image {img_path} not found"
            ret = model.generate([img_path, prompt])
            response.append({
                "image": img_name,
                "prompt": prompt,
                "choices": choices,
                "reference": reference,
                "response": ret
            })

        output_path = os.path.join(UNI_IMAGE_FOLDER, task_name, city_name, model_name)
        os.makedirs(output_path, exist_ok=True)
        print("Save the response in" + output_path)
        with open(os.path.join(output_path, f"{city_name}_{task_name}_{zl}_response.json"), "w") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)

    model_wrapper.clean_proxy()
