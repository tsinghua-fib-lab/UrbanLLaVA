# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
random.seed(0)
from tqdm import trange

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
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args() 

    city_name = args.city_name
    task_name = args.task_name   
    work_dir = args.work_dir

    cur_dir = os.path.join(work_dir, f"dev-{city_name}/")

    other_city_names = ['Beijing', 'London', 'NewYork']
    other_city_names.remove(city_name)
    other_city = random.choice(other_city_names)
    other_city_dir = os.path.join(work_dir, f"dev-{other_city}/")


    all_data = []
    easy_all_data = []

    # For street view, we donnot need to distinguish between zl15 and zl17 finally
    for zl in ['zl15', 'zl17']:
        sat_address_file = cur_dir + f"stv_in_sat_address_deploy_{zl}.csv"
        df = pd.read_csv(sat_address_file)
        # remove the rows with empty address
        df = df.dropna(subset=["adr"])

        other_city_address_file = other_city_dir + f"stv_in_sat_address_deploy_{zl}.csv"
        other_city_df = pd.read_csv(other_city_address_file)
        other_city_df = other_city_df.dropna(subset=["adr"])

        # randomly shuffle df, to avoid the same address in the same order
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        for i in range(len(df)):
            df.at[i, "img_name"] = df.iloc[i]["image_name"]        

        stv_img_dir = cur_dir + f"sampled_stv_images/"


        print("Input file:", sat_address_file)
        print("Valid records:", len(df))

        output = []
        easy_output = []

        for i in trange(len(df)):
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

            random.shuffle(choices)

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

            other_city_choices = other_city_df.sample(3)["adr"].tolist()
            easy_choices = [adr] + other_city_choices
            random.shuffle(easy_choices)
            easy_reference = chr(ord('A') + easy_choices.index(adr))
            easy_prompt = prompt_template(choice1=easy_choices[0], choice2=easy_choices[1], choice3=easy_choices[2], choice4=easy_choices[3])
            easy_output.append({
                "prompt": easy_prompt,
                "choices": easy_choices,
                "reference": easy_reference,
                "image": os.path.join(stv_img_dir, img_name)
            })

            all_data.append({
                "prompt": prompt,
                "choices": choices,
                "reference": reference,
                "image": os.path.join(stv_img_dir, img_name)
            })

            easy_all_data.append({
                "prompt": easy_prompt,
                "choices": easy_choices,
                "reference": easy_reference,
                "image": os.path.join(stv_img_dir, img_name)
            })


    all_test_data = random.sample(all_data, min(200, len(all_data)))
    all_train_data = [d for d in all_data if d not in all_test_data]

    easy_all_test_data = random.sample(easy_all_data, min(200, len(easy_all_data)))
    easy_all_train_data = [d for d in easy_all_data if d not in easy_all_test_data]


    output_dir = os.path.join(cur_dir, "uni_image_data", task_name, city_name)
    os.makedirs(output_dir, exist_ok=True)        

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train.json')}")
    print("Total train size:", len(all_train_data))

    all_test_data = random.sample(all_test_data, 200)
    with open(os.path.join(output_dir, f"{city_name}_{task_name}_test.json"), "w") as f:
        json.dump(all_test_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_test.json')}")
    print("Total test size:", len(all_test_data))    

    easy_all_test_data = random.sample(easy_all_test_data, 200)
    with open(os.path.join(output_dir, f"{city_name}_{task_name}_test_easy.json"), "w") as f:
        json.dump(easy_all_test_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_test_easy.json')}")
    print("Total test size:", len(easy_all_test_data) )

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train_easy.json"), "w") as f:
        json.dump(easy_all_train_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train_easy.json')}")
    print("Total train size:", len(easy_all_train_data))