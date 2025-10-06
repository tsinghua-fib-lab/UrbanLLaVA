# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
random.seed(0)
import sys
from config import UNI_IMAGE_FOLDER, BEIJING_STV_IMAGE_FOLDER, LONDON_STV_IMAGE_FOLDER
import tqdm


def prompt_template(choice1, choice2, choice3, choice4):
    s =  f"""
    The following is a multiple-choice question about selecting the most possible nearby POIs(Place of Interests) or landmarks description in the region of a street view image.
    A. {choice1}
    B. {choice2}
    C. {choice3}
    D. {choice4}
    Please choose the most suitable one among A, B, C and D as the answer to this question. 
    Please output the option directly. No need for explaination.\n
    """

    return s.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--task_name', type=str, default='stv_landmark_mc', help='task name')
    args = parser.parse_args() 

    city_name = args.city_name
    task_name = args.task_name   

    work_dir = UNI_IMAGE_FOLDER

    work_dir = os.path.join(work_dir, f"{city_name}/")

    all_train_data = []
    all_test_data = []

    for zl in ['zl15', 'zl17']:
        if city_name == "Beijing":
            stv_img_dir = BEIJING_STV_IMAGE_FOLDER
        elif city_name == "London":
            stv_img_dir = LONDON_STV_IMAGE_FOLDER
        elif city_name == "NewYork":
            pass

        output = []

        input_path = os.path.join(work_dir, "stv_poi_landmark_update.jsonl")
        with open(input_path, "r") as f:
            data = [json.loads(line) for line in f]

        output = []

        all_choices = [d["text"] for d in data]
        all_choices = list(set(all_choices))

        for d in tqdm.tqdm(data):
            # is absolute path
            img_name = d["img_name"].split('/')[-1]
            text = d["text"]

            other_choices = random.sample([text for text in all_choices if text != d["text"]], 3)
            choices = [text] + other_choices
            random.shuffle(choices, random=random.seed(data.index(d)))

            reference = chr(ord('A') + choices.index(text))

            prompt = prompt_template(choices[0], choices[1], choices[2], choices[3])

            output.append({
                "prompt": prompt,
                "choices": choices,
                "reference": reference,
                "image": os.path.join(stv_img_dir, img_name)
            })



        output_dir = os.path.join(UNI_IMAGE_FOLDER, task_name, city_name)
        os.makedirs(output_dir, exist_ok=True)

        
        test = random.sample(output, min(200, len(output)))
        train = [d for d in output if d not in test]

        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_test.json"), "w") as f:
            json.dump(test, f, indent=4, ensure_ascii=False)

        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_test.json')}")
        print("Test size:", len(test))
              
        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_train.json"), "w") as f:
            json.dump(train, f, indent=4, ensure_ascii=False)
        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_train.json')}")
        print("Train size:", len(train))

        all_train_data.extend(train)
        all_test_data.extend(test)

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train.json')}")
    print("Total train size:", len(all_train_data))

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_test.json"), "w") as f:
        json.dump(all_test_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_test.json')}")
    print("Total test size:", len(all_test_data))
