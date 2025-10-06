# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
from sklearn.model_selection import train_test_split
random.seed(0)



def prompt_template(choice1, choice2, choice3, choice4):
    s =  f"""
    The following is a multiple-choice question about selecting the most possible landuse type in the region of a satellite image.
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
    parser.add_argument('--task_name', type=str, default='sat_landuse_mc', help='task name')
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args() 

    city_name = args.city_name
    task_name = args.task_name   
    work_dir = args.work_dir

    cur_dir = os.path.join(work_dir, f"dev-{city_name}/")

    all_train_data = []
    for zl in ['zl15', 'zl17']:
        sat_img_dir = cur_dir + f"sample_sat_image_{zl}/"

        output = []

        if os.path.exists(f"rs_landuse_description_{zl}.jsonl"):
            os.remove(f"rs_landuse_description_{zl}.jsonl")
            print(f"Removed rs_landuse_description_{zl}.jsonl")
        df = pd.read_csv(cur_dir + f"SAT_{city_name}_{zl}.csv")

        for cnt in range(len(df)):
            img_name = df.at[cnt,'img_name'].split('.')[0]  

            if not os.path.exists(cur_dir + f'short_clipped_results_{zl}/landuse_'+img_name +'.txt'):
                continue

            with open(cur_dir + f'short_clipped_results_{zl}/landuse_'+img_name +'.txt', 'r') as file:
            # with open('short_clipped_results_wudaokou_zl17/landuse_'+img_name +'.txt', 'r') as file:
                lines = file.readlines()

            landuse_types_list = ["Retail", "Recreation_ground", "Commercial", "Residential", "Grass", "Forest", "Construction", "Meadow", "Garages", "Railway", "Brownfield", "Farmland", "Religious", "Industrial", "Recreation"]

            for line in lines:
                parts = line.split('location:')
                landuse_type = line.split('region')[0].strip().split()[-1].capitalize()  
                if not landuse_type in landuse_types_list:
                    landuse_types_list.append(landuse_type)            
            i = 0
            for line in lines:
                parts = line.split('location:')
                landuse_type = line.split('region')[0].strip().split()[-1].capitalize()  
                assert landuse_type in landuse_types_list, landuse_type

                other_choices = [d for d in landuse_types_list if d != landuse_type]
                other_choices = random.sample(other_choices, 3)
                choices = [landuse_type] + other_choices
                random.shuffle(choices)
                i += 1
                reference = chr(ord('A') + choices.index(landuse_type))
                prompt = prompt_template(choice1=choices[0], choice2=choices[1], choice3=choices[2], choice4=choices[3])
            

                output.append({
                    "prompt": prompt,
                    "choices": choices,
                    "reference": reference,
                    "image": os.path.join(sat_img_dir, img_name + '.png')
                })

        test = random.sample(output, min(200, len(output)))
        train = [d for d in output if d not in test]
        all_train_data.extend(train)
        output_dir = os.path.join(cur_dir, "uni_image_data", task_name, city_name)

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_test.json"), "w") as f:
            json.dump(test, f, indent=4, ensure_ascii=False)

        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_test.json')}")
        print("Test size:", len(test))

        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_train.json"), "w") as f:
            json.dump(train, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_train.json')}")
    print("Total train size:", len(train))
