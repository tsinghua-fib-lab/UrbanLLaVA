# This script is used to convert a address QA into a multi-choice question for evaluation.

import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
import csv

from config import UNI_IMAGE_FOLDER, RESULTS_PATH
from serving.llm_api import extract_choice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--task_name', type=str, default='stv_address_mc', help='task name')
    args = parser.parse_args()    

    city = args.city_name
    model_name = args.model_name
    task_name = args.task_name

    for zl in ["zl17"]:
        path = os.path.join(UNI_IMAGE_FOLDER, task_name, city, model_name, f"{city}_{task_name}_{zl}_response.json")


    headers_needed_zl17 = not os.path.exists(csv_path_zl17)

    with open(csv_path_zl17, mode='a', newline='') as file_zl17:
        writer_zl17 = csv.writer(file_zl17)
        if headers_needed_zl17:
            writer_zl17.writerow(["model_name", "city", "Accuracy"])

        for zl in ["zl17"]:
            
            path = os.path.join(RESULTS_PATH, task_name, city, f"{city}_{task_name}_{zl}_{args.model_name}_response.json")

            with open(path, "r") as f:
                data = json.load(f)

            correct = 0
            num_A = 0
            num_B = 0
            num_C = 0
            num_D = 0
            
            for d in data:
                prompt = d["prompt"]
                choices = d["choices"]
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
                
            accuracy = correct / len(data)
            writer_zl17.writerow([model_name, city, accuracy])
            print("For Response file:", path)
            print("Accuracy:", correct / len(data))
