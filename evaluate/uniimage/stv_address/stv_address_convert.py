# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
random.seed(0)
from tqdm import trange
from config import UNI_IMAGE_FOLDER, BEIJING_STV_IMAGE_FOLDER, LONDON_STV_IMAGE_FOLDER

def prompt_template(choice1, choice2, choice3, choice4):
    s =  f"""
    The following is a multiple-choice question about selecting the most appropriate address for a street view image.
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
    parser.add_argument('--task_name', type=str, default='stv_address_mc', help='task name')
    args = parser.parse_args() 

    city_name = args.city_name
    task_name = args.task_name   

    work_dir = UNI_IMAGE_FOLDER

    cur_dir = os.path.join(work_dir, f"{city_name}/")


    all_train_data = []

    for zl in ['zl15', 'zl17']:
        sat_address_file = cur_dir + f"stv_in_sat_address_deploy_{zl}.csv"
        df = pd.read_csv(sat_address_file)
        # remove the rows with empty address
        df = df.dropna(subset=["adr"])

        # randomly shuffle df, to avoid the same address in the same order
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        for i in range(len(df)):
            df.at[i, "img_name"] = df.iloc[i]["image_name"]        

        if city_name == "Beijing":
            stv_img_dir = BEIJING_STV_IMAGE_FOLDER
        elif city_name == "London":
            stv_img_dir = LONDON_STV_IMAGE_FOLDER
        elif city_name == "NewYork":
            pass


        print("Input file:", sat_address_file)
        print("Valid records:", len(df))

        output = []

        for i in range(len(df)):
            row = df.iloc[i]
            img_name = row["img_name"]
            adr = row["adr"]

            # print(combined_adr)

            assert os.path.exists(os.path.join(stv_img_dir, img_name)), f"Image {os.path.join(stv_img_dir, img_name)} not found"

            # Randomly select 3 other addresses
            valid_choices = df[(df["img_name"] != img_name) & (df["adr"] != adr)]["adr"].unique()
            if len(valid_choices) < 3:
                raise ValueError(f"Not enough valid choices to sample for image {img_name}")
            
            # other_choices = df[(df["img_name"] != img_name) & (df["adr"] != adr)].sample(3)["adr"].tolist()
            other_choices = random.sample(list(valid_choices), 3)
            choices = [adr] + other_choices

            assert len(list(set(choices))) == 4

            random.shuffle(choices, random=random.seed(i))

            # print(choices)

            reference = chr(ord('A') + choices.index(adr))

            # print(reference)

            prompt = prompt_template(choice1=choices[0], choice2=choices[1], choice3=choices[2], choice4=choices[3])

            # print(prompt)

            output.append({
                "prompt": prompt,
                "choices": choices,
                "reference": reference,
                "image": os.path.join(stv_img_dir, img_name)
            })

            # print(output[-1])

            # exit()

        # os.makedirs(f"./{task_name}/{city_name}", exist_ok=True)
        output_dir = os.path.join(work_dir, task_name, city_name)
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

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train.json')}")
    print("Total train size:", len(all_train_data))

