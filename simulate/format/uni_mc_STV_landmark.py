# This script is used to convert a address QA into a multi-choice question for evaluation.

import json
import os
import argparse
import pandas as pd
import random
random.seed(0)
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

    stv_img_dir = cur_dir + f"sampled_stv_images/"

    output = []

    input_path = os.path.join(cur_dir, "stv_poi_landmark_update.jsonl")
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]

    data = random.sample(data, min(3000, len(data)))

    other_city_path = os.path.join(other_city_dir, "stv_poi_landmark_update.jsonl")
    with open(other_city_path, "r") as f:
        other_city_data = [json.loads(line) for line in f]

    output = []
    easy_output = []

    for d in tqdm.tqdm(data):
        # is absolute path
        img_name = d["img_name"].split('/')[-1]
        text = d["text"]
        if "Unknown" in text or "Unidentified" in text or "No landmark" in text or "No recognizable" in text:
            continue
        all_choices = [d["text"] for d in data]
        other_choices = random.sample([text for text in all_choices 
                                       if (text != d["text"] and "Unknown" not in text and "Unidentified" not in text and "No landmark" not in text and "No recognizable" not in text)
                                        ], 3)
        choices = [text] + other_choices
        random.shuffle(choices)
        reference = chr(ord('A') + choices.index(text))
        prompt = prompt_template(choices[0], choices[1], choices[2], choices[3])
        output.append({
            "prompt": prompt,
            "choices": choices,
            "reference": reference,
            "image": os.path.join(stv_img_dir, img_name)
        })

        easy_all_choices = [d["text"] for d in other_city_data]
        easy_other_choices = random.sample([text for text in all_choices 
                                       if (text != d["text"] and "Unknown" not in text and "Unidentified" not in text and "No landmark" not in text and "No recognizable" not in text)
                                        ], 3)        
        easy_choices = [text] + easy_other_choices
        random.shuffle(easy_choices)
        easy_reference = chr(ord('A') + easy_choices.index(text))
        easy_prompt = prompt_template(easy_choices[0], easy_choices[1], easy_choices[2], easy_choices[3])
        easy_output.append({
            "prompt": easy_prompt,
            "choices": easy_choices,
            "reference": easy_reference,
            "image": os.path.join(stv_img_dir, img_name)
        })

    all_data.extend(output)
    easy_all_data.extend(easy_output)

    # output_dir = os.path.join(UNI_IMAGE_FOLDER, task_name, city_name)
    output_dir = os.path.join(cur_dir, "uni_image_data", task_name, city_name)
    os.makedirs(output_dir, exist_ok=True)

    all_test_data = random.sample(all_data, min(200, len(all_data)))
    all_train_data = [d for d in all_data if d not in all_test_data]

    easy_all_test_data = random.sample(easy_all_data, min(200, len(easy_all_data)))
    easy_all_train_data = [d for d in easy_all_data if d not in easy_all_test_data]
        

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train.json"), "w") as f:
        json.dump(all_train_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train.json')}")
    print("Total train size:", len(all_train_data))

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_test.json"), "w") as f:
        json.dump(all_test_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_test.json')}")
    print("Total test size:", len(all_test_data))

    with open(os.path.join(output_dir, f"{city_name}_{task_name}_train_easy.json"), "w") as f:
        json.dump(easy_all_train_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_train_easy.json')}")
    print("Train size:", len(easy_all_train_data))
          
    with open(os.path.join(output_dir, f"{city_name}_{task_name}_test_easy.json"), "w") as f:
        json.dump(easy_all_test_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {os.path.join(output_dir, f'{city_name}_{task_name}_test_easy.json')}")
    print("Test size:", len(easy_all_test_data))

