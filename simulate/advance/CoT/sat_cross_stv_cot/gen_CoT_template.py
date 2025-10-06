import json
import pandas as pd
from tqdm import trange
import argparse
import os
import random
random.seed(0)

# Generate CoT ground truth for satelite view and street view image crossing tasks
# The First is sat-stv mapping, the second is sat-stv location

# SAT-STV mapping, find the corresponding satelite view image for a street view image
# 1. First, describe the street view image
# 2. Second, reason the address of the street view image
# 3. Third, find the corresponding satelite view image

def get_STV_SAT_mapping_CoT_template(stv_description:str, stv_address:str, sat_address_list:list, reference:str):
    assert len(sat_address_list) == 4
    prompt = f"""
    Step 1: Describe the street view image:
    This street view image is described as: {stv_description}.
    Step 2: Recall the address of the street view image:
    According to the street view image, this street view image is taken in a region with the address: {stv_address}.
    Step 3: Recall the addresses of each possible satelite view image:
    The first image is a satelite view image with the address: {sat_address_list[0]}.
    The second image is a satelite view image with the address: {sat_address_list[1]}.
    The third image is a satelite view image with the address: {sat_address_list[2]}.
    The fourth image is a satelite view image with the address: {sat_address_list[3]}.
    Step 4: Reason the corresponding satelite view image, A for the first, B for the second, C for the third, D for the fourth:
    So, the final answer to the satelite view image that corresponds to the street view image is: {reference}.

    """
    prompt = str(prompt).replace('\n', ' ').strip()
    return prompt


# SAT-STV location, find the location of a street view image in satelite view image
# 1. First, describe the satelite view image
# 2. Second, describe the street view image
# 3. Third, find the location of the street view image in satelite view image

def get_STV_SAT_location_CoT_template(sat_address:str, stv_address:str, reference:str):
    prompt = f"""
    Step 1: Identify the satelite view image's address:
    According to the satelite view image, this satelite view image is taken in a region with the address: {sat_address}.
    Step 2: Identify the street view image's address:
    The street view image is taken in a region with the address: {stv_address}.
    Step 3: Reason the quadrant where the street view image lies in the satelite view image:
    So, the final answer to the quadrant where the street view image lies in the satelite view image is: {reference}.

    """
    prompt = str(prompt).replace('\n', ' ').strip()
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--task', type=str, default='sat-cross-stv-cot', choices=['sat-cross-stv-cot', 'sat-loc-stv-cot'])

    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    task = args.task

    cur_dir = work_dir + f'dev-{city}/'
    output_dir = os.path.join(cur_dir, 'CoT', task)
    os.makedirs(output_dir, exist_ok=True)

    STV_SAT_location_json_path = os.path.join(cur_dir, 'multi_image_data', 'STV_SAT_location_train.json')

    with open(STV_SAT_location_json_path, 'r') as f:
        STV_SAT_location = json.load(f)

    STV_SAT_location = random.sample(STV_SAT_location, min(3000, len(STV_SAT_location)))

    def get_STV_SAT_location_prompt():
        options = [
            "You are given a satellite image <image> and a street view image <image>, and please predict which quadrant the street view image lies in the satellite image.\n            A. Top left\n            B. Top right\n            C. Bottom left\n            D. Bottom right\n            Only provide one letter as the answer and please select your answer from A, B, C, or D.",
            "You are given a satellite image <image> and a street view image <image>, which quadrant does the street view image lie in the satellite image?\n            A. Top left\n            B. Top right\n            C. Bottom left\n            D. Bottom right\n            Please provide one letter as the answer and select your answer from A, B, C, or D.",
            "You are given a satellite image <image> and a street view image <image>, please point out the quadrant where the street view image is located in the satellite image.\n            A. Top left\n            B. Top right\n            C. Bottom left\n            D. Bottom right\n            Please provide one letter as the answer and select your answer from A, B, C, or D."
        ]
        prompt = random.choice(options)
        return prompt
    
    output = []

    for i in trange(len(STV_SAT_location)):
        prompt = get_STV_SAT_location_prompt()
        reference = STV_SAT_location[i]["conversations"][1]["value"]
        # The First image is satelite view image, the second is street view image
        image_list = STV_SAT_location[i]["image"]
        sat_image = image_list[0].split("/")[-1]
        stv_image = image_list[1].split("/")[-1]

        sat_address = None
        # query for the satelite view image's combined address
        for zl in ["zl15", "zl17"]:
            sat_address_combined_csv = os.path.join(cur_dir, f'sat_address_combined_{city}_{zl}.csv')
            df_combined_address = pd.read_csv(sat_address_combined_csv)

            for j in range(len(df_combined_address)):
                if df_combined_address.loc[j, 'img_name'] == sat_image:
                    sat_address = df_combined_address.loc[j, 'combined_adr']
                    break

        if sat_address is None:
            continue

        

        # query for the street view image's description
        stv_address = None
        for zl in ["zl15", "zl17"]:
            stv_in_sat_address_csv = os.path.join(cur_dir, f'stv_in_sat_address_deploy_{zl}.csv')

            df_stv_in_sat_address = pd.read_csv(stv_in_sat_address_csv)

            for j in range(len(df_stv_in_sat_address)):
                if df_stv_in_sat_address.loc[j, 'image_name'] == stv_image:
                    stv_address = df_stv_in_sat_address.loc[j, 'adr']
                    break

        if stv_address is None:
            continue

        output.append({
            "image": [sat_image, stv_image],
            "prompt": prompt,
            "CoT": get_STV_SAT_location_CoT_template(sat_address, stv_address, reference),
            "sat_address": sat_address,
            "stv_address": stv_address,
            "reference": reference
        })

    output_path = os.path.join(output_dir, f'SAT_STV_location_CoT_{city}.json')
    print(f"Saving CoT to {output_path}")
    print("Total number of CoT:", len(output))

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


    ##############################################

    STV_SAT_mapping_json_path = os.path.join(cur_dir, 'multi_image_data', 'STV_SAT_mapping_train.json')
    with open(STV_SAT_mapping_json_path, 'r') as f:
        STV_SAT_mapping = json.load(f)

    STV_SAT_mapping = random.sample(STV_SAT_mapping, min(3000, len(STV_SAT_mapping)))
    # STV_SAT_mapping = STV_SAT_mapping[:30]

    STV_SAT_mapping_output = []

    def get_STV_SAT_mapping_prompt():
        options = [
            "You are given one street view image <image> and four satellite images. Which satellite image shows the surroundings of the street view image?\n            A. The first image <image>\n            B. The second image <image>\n            C. The third image <image>\n            D. The fourth image <image>\n            Only provide one letter as the answer and please select your answer from A, B, C, or D.",
            "You are given one street view image <image> and four satellite images. Which satellite image contains the street view image?\n            A. The first image <image>\n            B. The second image <image>\n            C. The third image <image>\n            D. The fourth image <image>\n            Only provide one letter as the answer and please select your answer from A, B, C, or D.",
        ]
        prompt = random.choice(options)
        return prompt

                   
    for i in trange(len(STV_SAT_mapping)):
        prompt = get_STV_SAT_mapping_prompt()
        reference = STV_SAT_mapping[i]["conversations"][1]["value"]

        # The First image is street view image, the second is satelite view image
        image_list = STV_SAT_mapping[i]["image"]
        stv_image = image_list[0].split("/")[-1]
        sat_image_list = [image.split("/")[-1] for image in image_list[1:]]

        # print(stv_image, sat_image_list)

        sat_address_list = []
        for sat_image in sat_image_list:
            for zl in ["zl15", "zl17"]:
                sat_address_combined_csv = os.path.join(cur_dir, f'sat_address_combined_{city}_{zl}.csv')
                df_combined_address = pd.read_csv(sat_address_combined_csv)
                for j in range(len(df_combined_address)):
                    if df_combined_address.loc[j, 'img_name'] == sat_image:
                        sat_address = df_combined_address.loc[j, 'combined_adr']
                        sat_address_list.append(sat_address)
                        break

        # print(len(sat_address_list))
        if len(sat_address_list) != 4:
            continue

        # query for the street view image's description
        stv_description = None
        for zl in ["zl15", "zl17"]:
            stv_description_jsonl = os.path.join(cur_dir, f'stv_description.jsonl')
            with open(stv_description_jsonl, 'r') as f:
                data_description = [json.loads(line) for line in f]

            for j in range(len(data_description)):
                if data_description[j]['img_name'].split("/")[-1] == stv_image:
                    stv_description = data_description[j]['text']
                    break

        if stv_description is None:
            continue

        # query for the street view image's address
        stv_address = None
        for zl in ["zl15", "zl17"]:
            stv_in_sat_address_csv = os.path.join(cur_dir, f'stv_in_sat_address_deploy_{zl}.csv')

            df_stv_in_sat_address = pd.read_csv(stv_in_sat_address_csv)

            for j in range(len(df_stv_in_sat_address)):
                if df_stv_in_sat_address.loc[j, 'image_name'] == stv_image:
                    stv_address = df_stv_in_sat_address.loc[j, 'adr']
                    break

        if stv_address is None:
            continue

        STV_SAT_mapping_output.append({
            "image": [stv_image] + sat_image_list,
            "prompt": prompt,
            "CoT": get_STV_SAT_mapping_CoT_template(stv_description, stv_address, sat_address_list, reference),
            "stv_description": stv_description,
            "stv_address": stv_address,
            "sat_address_list": sat_address_list,
            "reference": reference
        })

    STV_SAT_mapping_output_path = os.path.join(output_dir, f'SAT_STV_mapping_CoT_{city}.json')
    with open(STV_SAT_mapping_output_path, 'w') as f:
        json.dump(STV_SAT_mapping_output, f, indent=4, ensure_ascii=False)

    print(f"Saving CoT to {STV_SAT_mapping_output_path}")
    print("Total number of CoT:", len(STV_SAT_mapping_output))