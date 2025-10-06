# This script is used to convert a address QA into a multi-choice question for evaluation.

import os
import argparse
import pandas as pd
from tqdm import tqdm
import json

from config import MULTI_IMAGE_FOLDER
from serving.llm_api import extract_choice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--task_name', type=str, default='SAT_count_pois', help='task name')
    args = parser.parse_args()    

    city = args.city_name
    model_name = args.model_name
    task_name = args.task_name

    for zl in ["zl15", "zl17"]:
        path = os.path.join(MULTI_IMAGE_FOLDER, task_name, city, model_name, f"{city}_{task_name}_{zl}_response.json")

        with open(path, "r") as f:
            data = json.load(f)

        correct = 0
        num_A = 0
        num_B = 0
        num_C = 0
        num_D = 0
        
        for d in data:
            prompt = d["prompt"]
            reference = d["reference"]
            response = d["response"]
            img_name = d["image"]

            model_choice = extract_choice(response, ["A", "B", "C", "D"])

            if model_choice == reference:
                correct += 1

            if model_choice == "A":
                num_A += 1
            elif model_choice == "B":
                num_B += 1
            elif model_choice == "C":
                num_C += 1
            elif model_choice == "D":
                num_D += 1


        print("For Response file:", path)
        print("Accuracy:", correct / len(data))
        print("Num A:", num_A)
        print("Num B:", num_B)
        print("Num C:", num_C)
        print("Num D:", num_D)
        print()

        # save the stats
        stats_folder = os.path.join(MULTI_IMAGE_FOLDER, task_name, city, "stats")

        os.makedirs(stats_folder, exist_ok=True)

        with open(os.path.join(stats_folder, f"{task_name}_{city}_{model_name}_{zl}.json"), "w") as f:
            json.dump({
                "Length of Data": len(data),
                "Accuracy": correct / len(data),
                "Num A": num_A,
                "Num B": num_B,
                "Num C": num_C,
                "Num D": num_D
            }, f, indent=4)