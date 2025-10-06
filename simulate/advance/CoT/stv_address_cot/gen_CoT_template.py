import json
import argparse
import pandas as pd
from tqdm import trange
import os
import random
random.seed(0)

# Generate CoT ground truth
# Three reasoning steps:
# 1. Tell the city name
# 2. Extract the location's pois around
# 3. Tell the location's address


def stv_prompt_template(city_name:str, near_feature:str, description:str, address:str):
    """
    Generate the prompt for the street view task
    """
    prompt = f"""
    Step 1: Describe the street view image:
    This is a street view image, in thie image, {description}.
    Step 2: Tell the city name:
    According to the street view image, this is probably in {city_name}.
    Step 3: Extract the location's features around:
    The street view image is taken in a region with the following features: {near_feature}.
    Step 4: Tell the location's address:
    Based on my observation and knowledge about this region, the address of this region is {address}.
    """
    prompt = str(prompt).replace('\n', ' ').strip()

    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--task', type=str, default='stv-address-cot', choices=['stv-address-cot', 'sat-address-cot'])
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir      
    task = args.task
    work_dir = work_dir + f'dev-{city}'
    output_dir = os.path.join(work_dir, 'CoT', task)
    import os
    os.makedirs(output_dir, exist_ok=True)

    CITY_NAME = city

    stv_output = []


    for zl in ["zl15", "zl17"]:
        # Streetview    
        address_csv = os.path.join(work_dir, f'stv_in_sat_address_deploy_{zl}.csv')
        near_feature_csv = os.path.join(work_dir, f'stv_in_sat_nearest_features_update_{city}_{zl}.csv')
        stv_description_jsonl = os.path.join(work_dir, f'stv_description.jsonl')

        addr_df = pd.read_csv(address_csv)
        near_feature_df = pd.read_csv(near_feature_csv)
        with open(stv_description_jsonl, 'r') as f:
            stv_description = f.readlines()
        stv_description = [json.loads(x) for x in stv_description]

        addr_df = addr_df[:1000]

        for i in trange(len(addr_df)):
            image_name = addr_df.loc[i, 'image_name']
            address = addr_df.loc[i, 'adr']
            for j in range(len(stv_description)):
                image_name2 = stv_description[j]['img_name'].split('/')[-1]
                if image_name == image_name2:
                    description = stv_description[j]['text']
                    for k in range(len(near_feature_df)):
                        if near_feature_df.loc[k, 'image_name'] == image_name:
                            near_feature = near_feature_df.loc[k, 'feature_names']
                            
                            prompt = stv_prompt_template(CITY_NAME, near_feature, description, address)
                            stv_output.append({
                                "img_name": image_name,
                                "CoT": prompt,
                                "address": address,
                                "description": description,
                                "near_feature": near_feature
                            })



    # for i in trange(len(addr_df)):
    #     region_name = addr_df.loc[i, 'region_nam']
    #     sid = addr_df.loc[i, 'sid']
    #     adr = addr_df.loc[i, 'adr']
    #     near_feature = None
    #     for j in range(len(near_feature_df)):
    #         if near_feature_df.loc[j, 'sid'] == sid:
    #             near_feature = near_feature_df.loc[j, 'feature_names']
    #             break

    #     if near_feature is not None:
    #         near_feature_lst = str(near_feature).split(',')
    #         near_feature_lst = [x for x in near_feature_lst if x != '' and not x.isdigit()]
    #         prompt = stv_prompt_template(CITY_NAME, near_feature_lst, adr)
    #         stv_output.append({
    #             "region_name": region_name,
    #             "sid": sid,
    #             "adr": adr,
    #             "near_feature": near_feature_lst,
    #             "CoT": prompt
    #         })


    with open(os.path.join(output_dir, f'{task}_{CITY_NAME}.json'), 'w') as f:
        json.dump(stv_output, f, indent=4, ensure_ascii=False)

    print(f"Total number of CoT: {len(stv_output)}")
    print(f"Saving CoT to {os.path.join(output_dir, f'{task}_{CITY_NAME}.json')}")
        
        
