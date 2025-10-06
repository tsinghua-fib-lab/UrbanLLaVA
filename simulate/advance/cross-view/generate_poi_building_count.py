import json
import pandas as pd
import re
import os
import glob
import argparse
import random
import tqdm
from tqdm import tqdm, trange

def extract_keys_from_json_files(filenames):


    all_keys = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            data = json.load(f)
            all_keys.update(data.keys())
    return list(all_keys)





def create_location_dict(filename):


    location_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            match = re.match(r"(\w+)\s+are\s+at\s+locations:\s+\[(.*)\]", line)
            if match:
                location_type, coordinates = match.groups()
                num_locations = len(coordinates.split(',')) // 2
                location_dict[location_type] = num_locations
    return location_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city_name = args.city
    work_dir = args.work_dir
    working_dir = os.path.join(work_dir, f"dev-{city_name}")
    # Convert the POI txt files to JSON files
    for zl in ['zl15','zl17']:
        target_dir = os.path.join(working_dir, f"poi_json_{city_name}")
        os.makedirs(target_dir, exist_ok=True)
        sat_path = pd.read_csv(os.path.join(working_dir, f"SAT_{city_name}_{zl}.csv"))
        for i in range(len(sat_path)):
            sat_name = sat_path.at[i,'img_name'].split('.')[0]
            filename = os.path.join(working_dir, f"short_clipped_results_{zl}/pois_{sat_name}.txt")
            if not os.path.exists(filename):
                continue
            result_dict = create_location_dict(filename)
            with open(os.path.join(target_dir, f"pois_{sat_name}_update.json"), 'w') as f:
                json.dump(result_dict, f, indent=4)


    key_groups = {
        'group_1': ['kindergartens', 'schools', 'colleges', 'research_institutes', 'universitys'],
        'group_2': ['conveniences',  'malls', 'supermarkets'],
        'group_3': ['restaurants', 'bakerys','foods', 'fast_foods', 'beveragess', 'food_courts', 'bars', 'cafes', 'coffees', 'vending_machines', 'nightclubs'],
        'group_4': ['apartments', 'hostels', 'hotels'],
        'group_5': ['attractions']
    }

    # Count the number of POIs in each satellite image
    for zl in ['zl15','zl17']:
        sat_df = pd.read_csv(os.path.join(working_dir, f"SAT_{city_name}_{zl}.csv"))
        img_name_list = list(sat_df['img_name'])
        json_file_list = [os.path.join(working_dir, f"poi_json_{city_name}/pois_{x.split('.')[0]}_update.json") for x in img_name_list]
        result_data = []

        for json_file in json_file_list:
            file_path = json_file
            img_name = file_path.split('/')[-1].split('.')[0]
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            group_sums = {group: 0 for group in key_groups}

            for group, keys in key_groups.items():
                for key in keys:
                    group_sums[group] += json_data.get(key, 0)

            result_data.append([img_name] + list(group_sums.values()))

        columns = ['img_name'] + list(key_groups.keys())
        df = pd.DataFrame(result_data, columns=columns)
        df.to_csv(os.path.join(working_dir, f"POI_key_group_sums_{zl}_{city_name}.csv"), index=False)
        print(df)

    ################-----------------------------------------------------------------------------------------------------

        # count the number of buildings in each satellite image
        for zl in ['zl15','zl17']:

            sat_df = pd.read_csv(os.path.join(working_dir, f"SAT_{city_name}_{zl}.csv"))
            img_name_list = list(sat_df['img_name'])
            sat_name_list = []
            sat_building_num = []
            for i in img_name_list:
                img_name = i.split('.')[0]
                if not os.path.exists(os.path.join(working_dir, f"clipped_results_{zl}/clipped_buildings_{img_name}.geojson")):
                    continue
                with open(os.path.join(working_dir, f"clipped_results_{zl}/clipped_buildings_{img_name}.geojson"), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                num_features = len(json_data['features'])
                sat_name_list.append(img_name)
                sat_building_num.append(num_features)

            pd_dict = pd.DataFrame({'img_name':sat_name_list,'building_num':sat_building_num})
            pd_dict.to_csv(os.path.join(working_dir, f"building_num_sat_{zl}_{city_name}.csv"),index=False)
            print(pd_dict)

