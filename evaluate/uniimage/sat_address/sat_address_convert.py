# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
random.seed(0)

from config import UNI_IMAGE_FOLDER

def prompt_template(choice1, choice2, choice3, choice4):
    s =  f"""
    The following is a multiple-choice question about selecting the most appropriate address for a satellite image.
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
    parser.add_argument('--task_name', type=str, default='sat_address_mc', help='task name')
    args = parser.parse_args() 

    city_name = args.city_name
    task_name = args.task_name   

    work_dir = UNI_IMAGE_FOLDER

    cur_dir = os.path.join(work_dir, f"{city_name}/")

    all_train_data = []

    for zl in ['zl15', 'zl17']:
        sat_img_dir = cur_dir + f"sample_sat_image_{zl}/"
        sat_address_file = cur_dir + f"sat_address_combined_{city_name}_{zl}.csv"
        df = pd.read_csv(sat_address_file)
        # remove the rows with empty address
        df = df.dropna(subset=["combined_adr"])


        print("Input file:", sat_address_file)
        print("Valid records:", len(df))

        output = []

        for i in range(len(df)):
            row = df.iloc[i]
            img_name = row["img_name"]
            combined_adr = row["combined_adr"]

            # print(combined_adr)

            assert os.path.exists(sat_img_dir + img_name), f"Image {img_name} not found"

            # Randomly select 3 other addresses
            other_choices = df[df["img_name"] != img_name].sample(3)["combined_adr"].tolist()
            choices = [combined_adr] + other_choices
            random.shuffle(choices, random=random.seed(i))

            # print(choices)

            reference = chr(ord('A') + choices.index(combined_adr))

            # print(reference)

            prompt = prompt_template(choice1=choices[0], choice2=choices[1], choice3=choices[2], choice4=choices[3])

            # print(prompt)

            output.append({
                "prompt": prompt,
                "choices": choices,
                "reference": reference,
                "image": sat_img_dir + img_name
            })

            # print(output[-1])

            # exit()

        output_dir = os.path.join(work_dir, task_name, city_name)

        os.makedirs(output_dir, exist_ok=True)

        test = random.sample(output, min(200, len(output)))
        train = [x for x in output if x not in test]

        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_test.json"), "w") as f:
            json.dump(test, f, indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_train.json"), "w") as f:
            json.dump(train, f, indent=4, ensure_ascii=False)

        all_train_data.extend(train)

        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_test.json')}")
        print("Test size:", len(test))
        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_train.json')}")
        print("Train size:", len(train))

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)