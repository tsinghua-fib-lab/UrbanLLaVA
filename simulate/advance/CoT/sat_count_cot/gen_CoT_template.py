import json
import argparse
import pandas as pd
from tqdm import trange

# Generate CoT ground truth for satelite view image counting building or POIs task
# Three reasoning steps:
# 1. Describe the satelite image
# 2. Recall the location's pois around and address
# 3. Tell how many buildings or POIs in the image


def sat_adr_prompt_template(city_name:str, description:str, address:str, building_num:int):
    """
    Generate the prompt for the satelite view image addressing task
    """
    prompt = f"""
    Step 1: Tell the city name:
    According to the satelite view image, this image is probably taken in {city_name}.
    Step 2: Extract the location's features around:
    From the image, I can see the following features: {description}.
    Step 3: Tell the location's address:
    Based on my observation and knowledge about this region, the address of this region is {address}.
    Step 4: Count the number of buildings or POIs:
    There are {str(building_num)} buildings in this image.
    """
    prompt = str(prompt).replace('\n', ' ').strip()

    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    parser.add_argument('--task', type=str, default='sat-count-cot', choices=['sat-count-cot', 'sat-address-cot'])
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir      

    work_dir = work_dir + f'dev-{city}/'
    task = args.task
    output_dir = work_dir + 'CoT/' + task + '/'
    import os
    os.makedirs(output_dir, exist_ok=True)

    CITY_NAME = city

    # Satellite view
    for zl in ["zl15", "zl17"]:

        building_num_df = pd.read_csv(work_dir + f'building_num_sat_{zl}_{CITY_NAME}.csv')
        # add .png to img_name
        assert 'img_name' in building_num_df.columns and '.png' not in building_num_df['img_name'].values[0]
        building_num_df['img_name'] = building_num_df['img_name'].apply(lambda x: x + '.png')

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

        for i in trange(len(building_num_df)):
            img_name = building_num_df.loc[i, 'img_name']
            building_num = building_num_df.loc[i, 'building_num']
            for j in range(len(df_description)):
                img_name_1 = df_description.loc[j, 'img_name']
                if img_name == img_name_1:
                    description = df_description.loc[j, 'text']
                    for k in range(len(df_address)):
                        img_name_2 = df_address.loc[k, 'img_name']
                        if img_name == img_name_2:
                            address = df_address.loc[k, 'combined_adr']
                            prompt = sat_adr_prompt_template(CITY_NAME, description, address, building_num)
                            output.append({
                                "img_name": img_name,
                                "CoT": prompt,
                                "description": description,
                                "address": address,
                                "building_num": str(building_num)
                            })

        # for i in trange(len(df_description)):
        #     img_name_1 = df_description.loc[i, 'img_name']

        #     for j in range(len(df_address)):
        #         img_name_2 = df_address.loc[j, 'img_name']
        #         if img_name_1 == img_name_2:

        #             description = df_description.loc[i, 'text']

        #             address = df_address.loc[j, 'combined_adr']

        #             prompt = sat_adr_prompt_template(CITY_NAME, description, address)

        #             df_description.loc[i, 'CoT'] = prompt

        #             output.append({
        #                 "img_name": img_name_1,
        #                 "CoT": prompt,
        #                 "description": description,
        #                 "address": address
        #             })
            
        # output_path = f'sat_address_cot_{city}_{zl}.json'
        output_path = output_dir + f'{task}_{CITY_NAME}_{zl}.json'
        print(f"Saving CoT to {output_path}")
        print("Total number of CoT:", len(output))

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(f"CoT saved to {output_path}")
