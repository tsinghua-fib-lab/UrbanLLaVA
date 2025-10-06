import glob
import os
import pandas as pd
import json
import random
random.seed(0)
# random.randint(0, 1)

import json
import random
import argparse
import tqdm
from tqdm import tqdm, trange

mapping_list = ["A", "B", "C", "D"]


def train_test_split(all_data, all_data_file, min_test_num=200, test_size=0.2):
    test = random.sample(all_data, min(min_test_num, len(all_data)))
    train = [x for x in all_data if x not in test]
    with open(all_data_file.replace(".json", "_test.json"), "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    print("Test data saved!", "Length of test: ", len(test))
    print("File saved at: ", all_data_file.replace(".json", "_test.json"))
    with open(all_data_file.replace(".json", "_train.json"), "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    print("Train data saved!", "Length of train: ", len(train))
    print("File saved at: ", all_data_file.replace(".json", "_train.json"))
    print("=====================================")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city_name = args.city
    work_dir = args.work_dir
    working_dir = os.path.join(work_dir, f"dev-{city_name}")

    # Task 1: Choose the satellite image with the most buildings out of four options

    task_1_data_zl15 = []
    task_1_data_zl17 = []
    
    def sat_most_buildings_prompt()->str:
        rtn = """
        In the provided four satellite images in urban area, which image shows most buildings? 
        A. The first image <image>\n
        B. The second image <image>\n
        C. The third image <image>\n
        D. The fourth image <image>\n
        Only provide one letter as the answer and please select your answer from A, B, C, or D.
        """
        return rtn.strip()
    
    for zl in ["zl15", "zl17"]:
        src_csv = os.path.join(working_dir, f"building_num_sat_{zl}_{city_name}.csv")
        df = pd.read_csv(src_csv)
        for idx in trange(len(df)):
            img_name = df.at[idx,'img_name']
            building_num = df.at[idx,'building_num']

            if building_num > 20:
                # find images with less than 10 buildings
                less_building_num = df[df['building_num'] < 10]
                if len(less_building_num) < 3:
                    continue
            elif building_num > 10:
                # find images with less than 5 buildings
                less_building_num = df[df['building_num'] < 5]
                if len(less_building_num) < 3:
                    continue
            else:
                # find entries with less buildings than current entry
                less_building_num = df[df['building_num'] < building_num]
                if len(less_building_num) < 3:
                    continue
     
            # randomly select 3 entries with less buildings
            other_options = random.sample(less_building_num['img_name'].tolist(), 3)
            # add current entry to the list
            options = [img_name] + other_options
            random.shuffle(options)
            reference = mapping_list[options.index(img_name)]

            one_data = {}
            one_data["id"] = f"SC_buildings_{zl}_{city_name}_{idx}"
            
            # TODO: Fill in the image directory
            sat_image_dir = os.path.join("....../resources/ThreeCityImage", city_name, f"Sat_{zl}")

            one_data["conversations"] = []
            one_conversation = {}
            one_conversation["from"] = "human"
            one_conversation["value"] = sat_most_buildings_prompt()
            one_data["conversations"].append(one_conversation)
            one_conversation = {}
            one_conversation["from"] = "gpt"
            one_conversation["value"] = reference
            one_data["conversations"].append(one_conversation)
            one_data["image"] = [os.path.join(sat_image_dir, f"{img_name}.png") for img_name in options]

            if zl == "zl15":
                task_1_data_zl15.append(one_data)
            elif zl == "zl17":
                task_1_data_zl17.append(one_data)
            else:            
                raise ValueError("Invalid zl value")
            
            # TODO: copy images to target directory


    # Task 2: Choose the satellite image with the most POIs out of four options

    task_2_data_zl15 = []
    task_2_data_zl17 = []

    key_groups = {
        'group_1': ['kindergartens', 'schools', 'colleges', 'research_institutes', 'universitys'],
        'group_2': ['conveniences',  'malls', 'supermarkets'],
        'group_3': ['restaurants', 'bakerys','foods', 'fast_foods', 'beveragess', 'food_courts', 'bars', 'cafes', 'coffees', 'vending_machines', 'nightclubs'],
        'group_4': ['apartments', 'hostels', 'hotels'],
        'group_5': ['attractions']
    }

    def sat_most_pois_prompt(group_key:str)->str:
        
        representive_pois_list = key_groups[group_key]

        rtn = f"""
        In the provided four satellite images in urban area, which image probably shows most POIs (For example, {str(representive_pois_list)})?
        A. The first image <image>\n
        B. The second image <image>\n
        C. The third image <image>\n
        D. The fourth image <image>\n
        Only provide one letter as the answer and please select your answer from A, B, C, or D.
        """
        return rtn.strip()
    
    for zl in ["zl15", "zl17"]:
        src_csv = os.path.join(working_dir, f"POI_key_group_sums_{zl}_{city_name}.csv")
        df = pd.read_csv(src_csv)
        df['img_name'] = df['img_name'].apply(lambda x: x.replace("pois_", "").replace("_update", ""))
        for idx in trange(len(df)):
            img_name = df.at[idx,'img_name']

            for group_key in key_groups.keys():
                poi_sum = df.at[idx, group_key]

                if poi_sum > 20:
                    # find images with less than 10 buildings
                    less_poi_sum = df[df[group_key] < 10]
                    if len(less_poi_sum) < 3:
                        continue

                elif poi_sum > 10:
                    # find images with less than 5 buildings
                    less_poi_sum = df[df[group_key] < 5]
                    if len(less_poi_sum) < 3:
                        continue

                else:
                    # find entries with less buildings than current entry
                    less_poi_sum = df[df[group_key] < poi_sum]
                    if len(less_poi_sum) < 3:
                        continue
                # randomly select 3 entries with less buildings
                other_options = random.sample(less_poi_sum['img_name'].tolist(), 3)
                # add current entry to the list
                options = [img_name] + other_options
                random.shuffle(options)
                reference = mapping_list[options.index(img_name)]

                one_data = {}
                one_data["id"] = f"SC_pois_{zl}_{city_name}_{group_key}_{idx}"

                # TODO: Fill in the image directory
                sat_image_dir = os.path.join("....../resources/ThreeCityImage", city_name, f"Sat_{zl}")

                one_data["conversations"] = []
                one_conversation = {}
                one_conversation["from"] = "human"
                one_conversation["value"] = sat_most_pois_prompt(group_key)
                one_data["conversations"].append(one_conversation)
                one_conversation = {}
                one_conversation["from"] = "gpt"
                one_conversation["value"] = reference
                one_data["conversations"].append(one_conversation)
                one_data["image"] = [os.path.join(sat_image_dir, f"{img_name}.png") for img_name in options]

                if zl == "zl15":
                    task_2_data_zl15.append(one_data)
                elif zl == "zl17":
                    task_2_data_zl17.append(one_data)
                else:            
                    raise ValueError("Invalid zl value")
                
                # TODO: copy images to target directory

    # Save the data
    all_data = task_1_data_zl15 + task_1_data_zl17 + task_2_data_zl15 + task_2_data_zl17
    all_zl15_data = task_1_data_zl15 + task_2_data_zl15
    all_zl17_data = task_1_data_zl17 + task_2_data_zl17
    
    output_dir = os.path.join(work_dir, f"dev-{city_name}", "multi_image_data")
    os.makedirs(output_dir, exist_ok=True)

    # Save all data
    with open(os.path.join(output_dir, "SAT_count_all_data.json"), "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print("All data saved!", "Length of all_data: ", len(all_data))
    print("File saved at: ", os.path.join(output_dir, "SAT_count_all_data.json"))
    train_test_split(all_data, os.path.join(output_dir, "SAT_count_all_data.json"))

    with open(os.path.join(output_dir, "SAT_count_zl15_data.json"), "w") as f:
        json.dump(all_zl15_data, f, indent=4, ensure_ascii=False)
    print("ZL15 data saved!", "Length of all_zl15_data: ", len(all_zl15_data))
    train_test_split(all_zl15_data, os.path.join(output_dir, "SAT_count_zl15_data.json"))


    with open(os.path.join(output_dir, "SAT_count_zl17_data.json"), "w") as f:
        json.dump(all_zl17_data, f, indent=4, ensure_ascii=False)
    print("ZL17 data saved!", "Length of all_zl17_data: ", len(all_zl17_data))
    train_test_split(all_zl17_data, os.path.join(output_dir, "SAT_count_zl17_data.json"))


    with open(os.path.join(output_dir, "SAT_count_buildings_zl15_data.json"), "w") as f:
        json.dump(task_1_data_zl15, f, indent=4, ensure_ascii=False)
    print("ZL15 buildings data saved!", "Length of task_1_data_zl15: ", len(task_1_data_zl15))
    train_test_split(task_1_data_zl15, os.path.join(output_dir, "SAT_count_buildings_zl15_data.json"))


    with open(os.path.join(output_dir, "SAT_count_buildings_zl17_data.json"), "w") as f:
        json.dump(task_1_data_zl17, f, indent=4, ensure_ascii=False)
    print("ZL17 buildings data saved!", "Length of task_1_data_zl17: ", len(task_1_data_zl17))
    train_test_split(task_1_data_zl17, os.path.join(output_dir, "SAT_count_buildings_zl17_data.json"))


    with open(os.path.join(output_dir, "SAT_count_pois_zl15_data.json"), "w") as f:
        json.dump(task_2_data_zl15, f, indent=4, ensure_ascii=False)
    print("ZL15 pois data saved!", "Length of task_2_data_zl15: ", len(task_2_data_zl15))
    train_test_split(task_2_data_zl15, os.path.join(output_dir, "SAT_count_pois_zl15_data.json"))


    with open(os.path.join(output_dir, "SAT_count_pois_zl17_data.json"), "w") as f:
        json.dump(task_2_data_zl17, f, indent=4, ensure_ascii=False)
    print("ZL17 pois data saved!", "Length of task_2_data_zl17: ", len(task_2_data_zl17))
    train_test_split(task_2_data_zl17, os.path.join(output_dir, "SAT_count_pois_zl17_data.json"))


