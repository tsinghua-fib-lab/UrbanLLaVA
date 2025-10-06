# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
random.seed(0)

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
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--task_name', type=str, default='sat_address_mc')

    args = parser.parse_args() 

    city_name = args.city_name
    work_dir = args.work_dir
    task_name = args.task_name

    cur_dir = os.path.join(work_dir, f"dev-{city_name}/")

    other_city_names = ['Beijing', 'London', 'NewYork']
    other_city_names.remove(city_name)
    other_city = random.choice(other_city_names)
    other_city_dir = os.path.join(work_dir, f"dev-{other_city}/")

    all_train_data = []
    easy_all_train_data = []

    for zl in ['zl15', 'zl17']:
        sat_img_dir = cur_dir + f"sample_sat_image_{zl}/"
        sat_address_file = cur_dir + f"sat_address_combined_{city_name}_{zl}.csv"
        other_city_address_file = other_city_dir + f"sat_address_combined_{other_city}_{zl}.csv"

        df = pd.read_csv(sat_address_file)
        # remove the rows with empty address
        df = df.dropna(subset=["combined_adr"])

        other_city_df = pd.read_csv(other_city_address_file)
        other_city_df = other_city_df.dropna(subset=["combined_adr"])

        print("Input file:", sat_address_file)
        print("Valid records:", len(df))

        # We want two versions of the data: one with easy choices and one with hard choices
        # Here, easy choices are the correct address and 3 randomly selected incorrect addresses from another city
        # Hard choices are the correct address and 3 randomly selected incorrect addresses from the same city
        # output:list is for hard choices, easy_output:list is for easy choices
        output = []
        easy_output = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            img_name = row["img_name"]
            combined_adr = row["combined_adr"]

            # print(combined_adr)

            assert os.path.exists(sat_img_dir + img_name), f"Image {img_name} not found"

            # Randomly select 3 other addresses
            other_choices = df[df["img_name"] != img_name].sample(3)["combined_adr"].tolist()
            choices = [combined_adr] + other_choices
            random.shuffle(choices)

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

            # Easy level: correct address and 3 random addresses from another city
            other_city_choices = other_city_df.sample(3)["combined_adr"].tolist()
            easy_choices = [combined_adr] + other_city_choices
            random.shuffle(easy_choices)
            easy_reference = chr(ord('A') + easy_choices.index(combined_adr))
            easy_prompt = prompt_template(choice1=easy_choices[0], choice2=easy_choices[1], choice3=easy_choices[2], choice4=easy_choices[3])
            easy_output.append({
                "prompt": easy_prompt,
                "choices": easy_choices,
                "reference": easy_reference,
                "image": sat_img_dir + img_name
            })

            # print(output[-1])

            # exit()

        output_dir = os.path.join(cur_dir, "uni_image_data",
                                   task_name, city_name)

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

        easy_test = random.sample(easy_output, min(200, len(easy_output)))
        easy_train = [x for x in easy_output if x not in easy_test]
        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_test_easy.json"), "w") as f:
            json.dump(easy_test, f, indent=4, ensure_ascii=False)
        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_test_easy.json')}")
        print("Test size:", len(easy_test))
        with open(os.path.join(output_dir, f"{city_name}_{task_name}_{zl}_train_easy.json"), "w") as f:
            json.dump(easy_train, f, indent=4, ensure_ascii=False)
        print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_{zl}_train_easy.json')}")
        print("Train size:", len(easy_train))

        easy_all_train_data.extend(easy_train)


    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train.json')}")
    print("Train size:", len(all_train_data))

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train_easy.json"), "w") as f:
        json.dump(easy_all_train_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train_easy.json')}")
    print("Train size:", len(easy_all_train_data))