import os
import pandas as pd
import json
import random
random.seed(0)
import argparse
from tqdm import tqdm, trange
import shutil


'''
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      },
    ]
  },
  ...
]
'''

quad_mapping = {
    'Top_left': 'A',
    'Top_right': 'B',
    'Bottom_left': 'C',
    'Bottom_right': 'D'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city_name = args.city
    work_dir = args.work_dir

    task_1_data = []
    task_2_data = []                            
    for zl in ['zl15','zl17']:
        
        # Task 1: Given a SAT image and a STV image, determine which quadrant of the SAT image shows the location of the STV image
        def sat_partition_ques()->str:
            sat_partition_ques = ["You are given a satellite image <image> and a street view image <image>. Which quadrant of the satellite image shows the location of the street view image.",
                          "You are given a satellite image <image> and a street view image <image>, and please predict which quadrant the street view image lies in the satellite image."]
                    
            prompt = sat_partition_ques[random.randint(0,1)]

            rtn = f"""
            {prompt}
            A. Top left
            B. Top right
            C. Bottom left
            D. Bottom right
            Only provide one letter as the answer and please select your answer from A, B, C, or D.
            """

            return rtn.strip()


        relative_target_dir = 'multi_image_data_' + city_name + '/'

        sat_list = []  
        df = pd.read_csv(os.path.join(work_dir, f"dev-{city_name}", f"sat_stv_corr_{city_name}_{zl}_partition.csv"))
        print(f"Processing {city_name} {zl} data for task 1: Given a SAT image and a STV image, determine which quadrant of the SAT image shows the location of the STV image")
        for idx in trange(len(df)):
            quadrant = df.at[idx,'partition']
            assert quadrant in quad_mapping, f"Invalid quadrant: {quadrant}"
            reference = quad_mapping[quadrant]

            sat_img_name = df.at[idx,'sat_img_name']
            r = df.at[idx,'image_name']
            sat_list.append(sat_img_name)
            one_data = {}
            one_data["id"] = 'CL_'+zl+'_'+str(idx)   

            one_data["conversations"] = []
            one_conversation = {}
            one_conversation["from"] = "human"
            one_conversation["value"] = sat_partition_ques()
            one_data["conversations"].append(one_conversation)
            one_conversation = {}
            one_conversation["from"] = "gpt"
            one_conversation["value"] = str(reference)
            one_data["conversations"].append(one_conversation)
            one_data["image"] = []
            # make sure the satellite image is the first one
            one_data["image"].append(relative_target_dir+sat_img_name)
            one_data["image"].append(relative_target_dir+r)
            task_1_data.append(one_data.copy())
            # TODO: copy images to target_dir         

        # Task 2: Given a STV image and four SAT images, determine which SAT image shows the surroundings of the STV image
        def sat_partition_ques2()->str:
            sat_partition_ques2 = ["You are given one street view image <image> and four satellite images. Which satellite image shows the surroundings of the street view image?",
                            "You are given one street view image <image> and four satellite images. Which satellite image contains the street view image?"]
            prompt = sat_partition_ques2[random.randint(0,1)]

            rtn = f"""
            {prompt}
            A. The first image <image>\n
            B. The second image <image>\n
            C. The third image <image>\n
            D. The fourth image <image>\n
            Only provide one letter as the answer and please select your answer from A, B, C, or D.
            """

            return rtn.strip()


        sat_list = []  
        print(f"Processing {city_name} {zl} data for task 2: Given a STV image and four SAT images, determine which SAT image shows the surroundings of the STV image")
        df = pd.read_csv(os.path.join(work_dir, f"dev-{city_name}", f"sat_stv_corr_{city_name}_{zl}_partition.csv"))
        for idx in trange(len(df)):
            quadrant = df.at[idx,'partition']
            sat_img_name = df.at[idx,'sat_img_name']
            r = df.at[idx,'image_name']
            sat_list.append(sat_img_name)

            one_data = {}
            one_data["id"] ='IR_'+zl+'_'+str(idx) #sat_zl15
            

            unmatched_list = list(set(df['sat_img_name']))
            unmatched_list.remove(sat_img_name)

            other_options = random.sample(unmatched_list, 3)
            if len(other_options) < 3:
                raise ValueError("Not enough unmatched images")
            
            options = [sat_img_name] + other_options

            random.shuffle(options)
            num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

            matched_idx = options.index(sat_img_name)
            reference = num_to_letter[matched_idx]

            one_data["conversations"] = []
            one_conversation = {}
            one_conversation["from"] = "human"
            one_conversation["value"] = sat_partition_ques2()
            one_data["conversations"].append(one_conversation)
            one_conversation = {}
            one_conversation["from"] = "gpt"
            one_conversation["value"] = num_to_letter[int(matched_idx)]
            one_data["conversations"].append(one_conversation)
            one_data["image"] = []
            # make sure the street view image is the first one
            one_data["image"].append(relative_target_dir+r)            
            one_data["image"].append(relative_target_dir+options[0])
            one_data["image"].append(relative_target_dir+options[1])
            one_data["image"].append(relative_target_dir+options[2])
            one_data["image"].append(relative_target_dir+options[3])
            task_2_data.append(one_data.copy())

    # Save to file
    all_data = task_1_data + task_2_data
    output_dir = os.path.join(work_dir, f"dev-{city_name}", "multi_image_data")
    os.makedirs(output_dir, exist_ok=True)

    # Save all data
    with open(os.path.join(output_dir, "SAT_cross_STV_all_data.json"), "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print("Multi-image data generation done!")
    print("Length of all_data: ", len(all_data))
    print("File saved at: ", os.path.join(output_dir, "SAT_cross_STV_all_data.json"))

    test_all = random.sample(all_data, min(200, len(all_data)))
    train_all = [x for x in all_data if x not in test_all]

    with open(os.path.join(output_dir, "SAT_cross_STV_all_data_test.json"), "w") as f:
        json.dump(test_all, f, indent=4, ensure_ascii=False)
    print("Test data saved!")
    print("Length of test_all: ", len(test_all))
    print("File saved at: ", os.path.join(output_dir, "SAT_cross_STV_all_data_test.json"))

    with open(os.path.join(output_dir, "SAT_cross_STV_all_data_train.json"), "w") as f:
        json.dump(train_all, f, indent=4, ensure_ascii=False)
    print("Train data saved!")
    print("Length of train_all: ", len(train_all))
    print("File saved at: ", os.path.join(output_dir, "SAT_cross_STV_all_data_train.json"))
    print("=====================================")

    # STV_SAT_location
    with open(os.path.join(output_dir, "STV_SAT_location.json"), 'w', encoding='utf-8') as f:
        json.dump(task_1_data, f, ensure_ascii=False, indent=4)
    print("STV_SAT_location data generation done!, Length of STV_SAT_location data: ", len(task_1_data))
    print("File saved at: ", os.path.join(output_dir, "STV_SAT_location.json"))
    
    test_location = random.sample(task_1_data, min(200, len(task_1_data)))
    train_location = [x for x in task_1_data if x not in test_location]

    with open(os.path.join(output_dir, "STV_SAT_location_test.json"), "w") as f:
        json.dump(test_location, f, indent=4, ensure_ascii=False)
    print("Test data saved!")
    print("Length of test_location: ", len(test_location))
    print("File saved at: ", os.path.join(output_dir, "STV_SAT_location_test.json"))

    with open(os.path.join(output_dir, "STV_SAT_location_train.json"), "w") as f:
        json.dump(train_location, f, indent=4, ensure_ascii=False)
    print("Train data saved!")
    print("Length of train_location: ", len(train_location))
    print("File saved at: ", os.path.join(output_dir, "STV_SAT_location_train.json"))
    print("=====================================")

    # STV_SAT_mapping
    with open(os.path.join(output_dir, "STV_SAT_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(task_2_data, f, ensure_ascii=False, indent=4)
    print("STV_SAT_mapping data generation done!, Length of STV_SAT_mapping data: ", len(task_2_data))
    print("File saved at: ", os.path.join(output_dir, "STV_SAT_mapping.json"))

    test_mapping = random.sample(task_2_data, min(200, len(task_2_data)))
    train_mapping = [x for x in task_2_data if x not in test_mapping]

    with open(os.path.join(output_dir, "STV_SAT_mapping_test.json"), "w") as f:
        json.dump(test_mapping, f, indent=4, ensure_ascii=False)
    print("Test data saved!")
    print("Length of test_mapping: ", len(test_mapping))
    print("File saved at: ", os.path.join(output_dir, "STV_SAT_mapping_test.json"))

    with open(os.path.join(output_dir, "STV_SAT_mapping_train.json"), "w") as f:
        json.dump(train_mapping, f, indent=4, ensure_ascii=False)
    print("Train data saved!")
    print("Length of train_mapping: ", len(train_mapping))
    print("File saved at: ", os.path.join(output_dir, "STV_SAT_mapping_train.json"))

    print("Multi-image data generation done!")