import os
import pandas as pd
import pandas as pd
import random
from tqdm import tqdm, trange
# random.randint(0, 1)
import json
import jsonlines

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
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
  parser.add_argument('--work_dir', type=str, default='../../data/')
  args = parser.parse_args()
  city_name = args.city
  work_dir = args.work_dir

  # Task 1: Satellite image description

  sat_description_ques = ["<image>\nPlease describe the given arial image in detail.",
                      "<image>\nCould you provide a description of the content shown in the arial image."]
                              
  # make data like new_data
  sat_description_data = []

  for zl in ["zl15", "zl17"]:
     df = pd.read_csv(os.path.join(work_dir, f"dev-{city_name}", f"rs_osm_description_{city_name}_{zl}.csv"))
     for idx in range(len(df)):
        img_name = df.at[idx,'img_name']
        text = df.at[idx,'text']
        one_data = {}
        one_data["id"] = img_name
        # TODO: Change the following path to the actual path
        one_data["image"] =  os.path.join("....../ThreeCityImage/", city_name, f"Sat_{zl}", img_name)
        assert os.path.exists(one_data["image"]), f"{one_data['image']} does not exist."

        one_data["conversations"] = []
        one_conversation = {}
        one_conversation["from"] = "human"
        one_conversation["value"] = sat_description_ques[random.randint(0,1)]
        one_data["conversations"].append(one_conversation)
        one_conversation = {}
        one_conversation["from"] = "gpt"
        one_conversation["value"] = text 
        one_data["conversations"].append(one_conversation)
        sat_description_data.append(one_data.copy())
  print("Satellite image description done!")

  # Task 2: Satellite image combined address

  rs_address_ques = ["<image>\nPlease tell me the address and layout of the amenities in the arial image.",
                          "<image>\nWhat amenities can you see in the arial image? And what are the possible addresses?"]

  sat_address_data = []

  for zl in ["zl15", "zl17"]:
    df = pd.read_csv(os.path.join(work_dir, f"dev-{city_name}",f"sat_address_combined_{city_name}_{zl}.csv"))
    for idx in range(len(df)):
        img_name = df.at[idx,'img_name']
        text = df.at[idx,'combined_adr']
        one_data = {}
        one_data["id"] = img_name
        # TODO: Change the following path to the actual path
        one_data["image"] =  os.path.join("....../ThreeCityImage/", city_name, f"Sat_{zl}", img_name)
        assert os.path.exists(one_data["image"]), f"{one_data['image']} does not exist."

        one_data["conversations"] = []
        one_conversation = {}
        one_conversation["from"] = "human"
        one_conversation["value"] = rs_address_ques[random.randint(0,1)]
        one_data["conversations"].append(one_conversation)
        one_conversation = {}
        one_conversation["from"] = "gpt"
        one_conversation["value"] = text 
        one_data["conversations"].append(one_conversation)
        sat_address_data.append(one_data.copy())
  print("Satellite image address done!")

  # Task 3: Satellite image grounding

  sat_grounding_data = []

  for zl in ["zl15", "zl17"]:
    df = pd.read_csv(os.path.join(work_dir, f"dev-{city_name}",f"rs_grounding_selfmade_{zl}.csv"))
    for idx in range(len(df)):
        img_name = df.at[idx,'img_name']
        Q = df.at[idx,'Q']
        A = df.at[idx,'A']
        one_data = {}
        one_data["id"] = img_name
        # TODO: Change the following path to the actual path
        one_data["image"] =  os.path.join("....../ThreeCityImage/", city_name, f"Sat_{zl}", f"{img_name}.png")
        assert os.path.exists(one_data["image"]), f"{one_data['image']} does not exist."

        one_data["conversations"] = []
        one_conversation = {}
        one_conversation["from"] = "human"
        one_conversation["value"] = "<image>\n"+Q
        one_data["conversations"].append(one_conversation)
        one_conversation = {}
        one_conversation["from"] = "gpt"
        one_conversation["value"] = A
        one_data["conversations"].append(one_conversation)
        sat_grounding_data.append(one_data.copy())

  print("Satellite image POI grounding done!")

  # Task 4: Satellite image landuse

  sat_landuse_data = []

  for zl in ["zl15", "zl17"]:
    with jsonlines.open(os.path.join(work_dir, f"dev-{city_name}",f"rs_landuse_description_{zl}.jsonl")) as reader:
      for obj in reader:
        img_name = obj['img_name']
        Q = obj['Q']
        A = obj['A']
        one_data = {}
        one_data["id"] = img_name
        # TODO: Change the following path to the actual path
        one_data["image"] =  os.path.join("....../ThreeCityImage/", city_name, f"Sat_{zl}", f"{img_name}.png")
        assert os.path.exists(one_data["image"]), f"{one_data['image']} does not exist."

        one_data["conversations"] = []
        one_conversation = {}
        one_conversation["from"] = "human"
        one_conversation["value"] = "<image>\n"+Q
        one_data["conversations"].append(one_conversation)
        one_conversation = {}
        one_conversation["from"] = "gpt"
        one_conversation["value"] = A
        one_data["conversations"].append(one_conversation)
        sat_landuse_data.append(one_data.copy())

  print("Satellite image landuse done!")

  # Task 5: Street view image address

  stv_geoloc_ques = ["<image>\nPlease tell me the detailed address of this street view image.",
                          "<image>\nWhat is the detailed address of the provided street view image?"]
  
  stv_address_data = []

  for zl in ["zl15", "zl17"]:
    df = pd.read_csv(os.path.join(work_dir, f"dev-{city_name}",f"stv_in_sat_address_deploy_{zl}.csv"))
    for idx in range(len(df)):
      image_name = df.at[idx,'image_name']
      addr = df.at[idx,'adr']
      one_data
      one_data["id"] = image_name
      # TODO: Change the following path to the actual path
      one_data["image"] =  os.path.join("....../ThreeCityImage/", city_name, f"StreetView", image_name)
      assert os.path.exists(one_data["image"]), f"{one_data['image']} does not exist."

      one_data["conversations"] = []
      one_conversation = {}
      one_conversation["from"] = "human"
      one_conversation["value"] = stv_geoloc_ques[random.randint(0,1)]
      one_data["conversations"].append(one_conversation)
      one_conversation = {}
      one_conversation["from"] = "gpt"
      one_conversation["value"] = addr
      one_data["conversations"].append(one_conversation)
      stv_address_data.append(one_data.copy())

  print("Street view image address done!")      

  # Task 6: Street view image understanding

  stv_des_ques = ["<image>\nPlease describe the street view image and possible surrounding POIs.",
                          "<image>\nWhat is shown in this street view image? And what related pois could be estimated nearby."]
  
  stv_description_data = []

  with jsonlines.open(os.path.join(work_dir, f"dev-{city_name}","stv_description.jsonl")) as reader:
    for obj in reader:
      img_name = obj['img_name']
      text = obj['text']
      # print(text)

  # for idx in range(len(all_csv_df)):
      one_data = {}
      one_data["id"] = img_name.split('/')[-1]
      one_data["image"] =  img_name
      # pop
      one_data["conversations"] = []
      one_conversation = {}
      one_conversation["from"] = "human"
      one_conversation["value"] = stv_des_ques[random.randint(0,1)]
      one_data["conversations"].append(one_conversation)
      one_conversation = {}
      one_conversation["from"] = "gpt"
      one_conversation["value"] = text 
      one_data["conversations"].append(one_conversation)
      stv_description_data.append(one_data.copy())

  print("Street view image understanding done!")

  # Task 7: Street view image landmark

  stv_landmark_ques = ["<image>\nPlease find the landmark in this street view image and possible surrounding POIs. If there is no landmark, please say 'None'.",
                          "<image>\nWhat is the landmark in this image? If there is no landmark, please say 'None'."]
  
  stv_landmark_data = []

  with jsonlines.open(os.path.join(work_dir, f"dev-{city_name}","stv_poi_landmark_update.jsonl")) as reader:
    for obj in reader:
      img_name = obj['img_name']
      text = obj['text']
      # print(text)

  # for idx in range(len(all_csv_df)):
      one_data = {}
      one_data["id"] = img_name.split('/')[-1]
      one_data["image"] =  img_name
      # pop
      one_data["conversations"] = []
      one_conversation = {}
      one_conversation["from"] = "human"
      one_conversation["value"] = stv_landmark_ques[random.randint(0,1)]
      one_data["conversations"].append(one_conversation)
      one_conversation = {}
      one_conversation["from"] = "gpt"
      one_conversation["value"] = text 
      one_data["conversations"].append(one_conversation)
      stv_landmark_data.append(one_data.copy())

  print("Street view image landmark done!")

  # Save to file
  all_data = sat_description_data + sat_address_data + sat_grounding_data + sat_landuse_data + stv_address_data + stv_description_data + stv_landmark_data
  output_dir = os.path.join(work_dir, "dev-"+city_name, "llava_format")
  os.makedirs(output_dir, exist_ok=True)

  with open(os.path.join(output_dir, "sat_description.json"), "w") as f:
      json.dump(sat_description_data, f, indent=4, ensure_ascii=False)
  print("Satellite image description done!")
  print("Length of sat_description_data: ", len(sat_description_data))

  with open(os.path.join(output_dir, "sat_address.json"), "w") as f:
      json.dump(sat_address_data, f, indent=4, ensure_ascii=False)
  print("Satellite image address done!")
  print("Length of sat_address_data: ", len(sat_address_data))

  with open(os.path.join(output_dir, "sat_grounding.json"), "w") as f:
      json.dump(sat_grounding_data, f, indent=4, ensure_ascii=False)
  print("Satellite image POI grounding done!")
  print("Length of sat_grounding_data: ", len(sat_grounding_data))

  with open(os.path.join(output_dir, "sat_landuse.json"), "w") as f:
      json.dump(sat_landuse_data, f, indent=4, ensure_ascii=False)
  print("Satellite image landuse done!")
  print("Length of sat_landuse_data: ", len(sat_landuse_data))

  with open(os.path.join(output_dir, "stv_address.json"), "w") as f:
      json.dump(stv_address_data, f, indent=4, ensure_ascii=False)
  print("Street view image address done!")
  print("Length of stv_address_data: ", len(stv_address_data))

  with open(os.path.join(output_dir, "stv_description.json"), "w") as f:
      json.dump(stv_description_data, f, indent=4, ensure_ascii=False)
  print("Street view image understanding done!")
  print("Length of stv_description_data: ", len(stv_description_data))

  with open(os.path.join(output_dir, "stv_landmark.json"), "w") as f:
      json.dump(stv_landmark_data, f, indent=4, ensure_ascii=False)
  print("Street view image landmark done!")
  print("Length of stv_landmark_data: ", len(stv_landmark_data))

  with open(os.path.join(output_dir, f"{city_name}_basic_all_data_llava.json"), "w") as f:
      json.dump(all_data, f, indent=4, ensure_ascii=False)
  print("All data done!")
  print("Length of all_data: ", len(all_data))

  with open(os.path.join(output_dir, "summary.json"), "w") as f:
      content = {
          "sat_description": len(sat_description_data),
          "sat_address": len(sat_address_data),
          "sat_grounding": len(sat_grounding_data),
          "sat_landuse": len(sat_landuse_data),
          "stv_address": len(stv_address_data),
          "stv_description": len(stv_description_data),
          "stv_landmark": len(stv_landmark_data),
          "all_data": len(all_data)
      }
      json.dump(content, f, indent=4, ensure_ascii=False)

  print("All done!")
