import os
import pandas as pd
import json
import random
random.seed(0)
import argparse
from tqdm import tqdm, trange

mapping_list = ["A", "B", "C", "D"]

def STV_compare_prompt() -> str:
    choices = [
        "You are given one street view image <image>. Which of the following images is taken in the most close location?",
        "You are given one street view image <image>. Please choose the most close image from the following options.",
    ]
    choice = random.choice(choices)
    rtn = f"""{choice}
    A. Image A <image>
    B. Image B <image>
    C. Image C <image>
    D. Image D <image>
    Only provide one letter as the answer and please select your answer from A, B, C, or D.
    """

    return rtn.strip()

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
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--task_name', type=str, default='STV_compare', help='task name')
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

    # Task 1: Given one stv image, choose from 4 other stv images that are closest to the given one
    # easy task: 3 of the 4 images are from another city, and 1 is from the same city
    # hard task: 4 images are from the same city

    # maintain a dictionary to store the mapping from one address to the corresponding stv images
    address2stv = {}

    task1_same_city = []

    for zl in ["zl15", "zl17"]:
        stv_address_file = os.path.join(cur_dir, f"stv_in_sat_address_deploy_{zl}.csv")
        stv_address_df = pd.read_csv(stv_address_file)
        for index, row in stv_address_df.iterrows():
            address = row["adr"]
            image_name = row["image_name"]
            if address not in address2stv:
                address2stv[address] = []
            if image_name not in address2stv[address]:
                address2stv[address].append(image_name)

    print("Number of addresses: ", len(address2stv))
    # save the address2stv mapping to a json file
    with open(os.path.join(cur_dir, "address2stv.json"), "w") as f:
        json.dump(address2stv, f, indent=4, ensure_ascii=False)
    print("Address2stv mapping saved in ", os.path.join("address2stv.json"))

    for key, value in tqdm(address2stv.items()):
        if len(value) < 2:
            continue
        else:
            # make sure image names are not paths
            assert "/" not in value[0]
            given_image = random.choice(value)
            copy_value = value.copy()
            copy_value.remove(given_image)
            gt_image = random.choice(copy_value)
            assert gt_image != given_image

            # other 3 options are from different cities
            other_3_address = random.sample([addr for addr in address2stv.keys() if addr != key], 3)
            other_3_images = []
            for address in other_3_address:
                other_3_images.append(random.choice(address2stv[address]))

            options = [gt_image] + other_3_images
            random.shuffle(options)
            
            reference = mapping_list[options.index(gt_image)]

            one_data = {}
            one_data["id"] = f"STV_compare_{city_name}_{len(task1_same_city)}"
            one_data["conversations"] = [
                {
                    "from": "human",
                    "value": STV_compare_prompt()
                },
                {
                    "from": "gpt",
                    "value": reference
                }
            ]
            img_dir = ""
            one_data["image"] = [os.path.join(img_dir, given_image)]
            one_data["image"].extend([os.path.join(img_dir, img) for img in options])
            # one_data["options"] = options
            assert len(one_data["image"]) == 5
            task1_same_city.append(one_data)

    output_dir = os.path.join(work_dir, f"dev-{city_name}", "multi_image_data")
    os.makedirs(output_dir, exist_ok=True)      
    with open(os.path.join(output_dir, f"{task_name}_all_data.json"), "w") as f:
        json.dump(task1_same_city, f, indent=4, ensure_ascii=False)      
    print("Task 1 data saved!", "Length of task 1: ", len(task1_same_city))
    print("File saved at: ", os.path.join(output_dir, f"{task_name}_all_data.json"))
    train_test_split(task1_same_city, os.path.join(output_dir, f"{task_name}_all_data.json"), min_test_num=200, test_size=0.2)