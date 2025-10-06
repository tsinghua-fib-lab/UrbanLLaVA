import json
import argparse
import pandas as pd
from tqdm import trange

# Generate CoT ground truth
# Three reasoning steps:
# 1. Tell the city name
# 2. Extract the location's pois around
# 3. Tell the location's address


def sat_adr_prompt_template(city_name:str, description:str, address:str):
    """
    Generate the prompt for the satelite view image addressing task
    """
    prompt = f"""
    Step 1: Tell the city name
    According to the satelite view image, this image is taken in {city_name}.
    Step 2: Extract the location's features around
    From the image, I can see the following features: {description}.
    Step 3: Tell the location's address
    Based on my observation and knowledge about this region, the address of this region is {address}.
    """
    prompt = str(prompt).replace('\n', ' ').strip()

    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--task', type=str, default='sat-address-cot', choices=['sat-address-cot', 'street-view-address'])
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir      
    task = args.task
    work_dir = work_dir + f'dev-{city}/'

    output_dir = work_dir + f'CoT/{task}/'
    import os
    os.makedirs(output_dir, exist_ok=True)

    CITY_NAME = city

    # Satellite view
    for zl in ["zl15", "zl17"]:
        description_csv = work_dir + f'rs_osm_description_{CITY_NAME}_{zl}.csv'

        address_csv = work_dir + f'sat_address_combined_{CITY_NAME}_{zl}.csv'

        df_description = pd.read_csv(description_csv)
        df_address = pd.read_csv(address_csv)

        df_address = df_address.dropna(subset=["combined_adr"]).reset_index(drop=True)
        df_description = df_description.dropna(subset=["text"]).reset_index(drop=True)

        # assert len(df_description) == len(df_address)
        print("After dropping NaN values:")
        print(f"Number of description: {len(df_description)}")
        print(f"Number of address: {len(df_address)}")
        output = []
        print("Generating CoT...")

        for i in trange(len(df_description)):
            img_name_1 = df_description.loc[i, 'img_name']

            for j in range(len(df_address)):
                img_name_2 = df_address.loc[j, 'img_name']
                if img_name_1 == img_name_2:

                    description = df_description.loc[i, 'text']

                    address = df_address.loc[j, 'combined_adr']

                    prompt = sat_adr_prompt_template(CITY_NAME, description, address)

                    df_description.loc[i, 'CoT'] = prompt

                    output.append({
                        "img_name": img_name_1,
                        "CoT": prompt,
                        "description": description,
                        "address": address
                    })
            
        # output_path = f'sat_address_cot_{city}_{zl}.json'
        output_path = output_dir + f'{task}_{CITY_NAME}_{zl}.json'
        print(f"Saving CoT to {output_path}")
        print("Total number of CoT:", len(output))

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
