import pandas as pd
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir            

    # work_dir = "../../data/dev-Beijing/"
    work_dir = work_dir + f"dev-{city}/"
    for zl in ["zl15", "zl17"]:
        if os.path.exists(f"rs_landuse_description_{zl}.jsonl"):
            os.remove(f"rs_landuse_description_{zl}.jsonl")
            print(f"Removed rs_landuse_description_{zl}.jsonl")
        df = pd.read_csv(work_dir + f"SAT_{city}_{zl}.csv")
        for cnt in range(len(df)):
            img_name = df.at[cnt,'img_name'].split('.')[0]

            if not os.path.exists(work_dir + f'short_clipped_results_{zl}/landuse_'+img_name +'.txt'):
                continue

            with open(work_dir + f'short_clipped_results_{zl}/landuse_'+img_name +'.txt', 'r') as file:
            # with open('short_clipped_results_wudaokou_zl17/landuse_'+img_name +'.txt', 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.split('location:')
                landuse_type = line.split('region')[0].strip().split()[-1].capitalize()  
                coordinates = parts[1].strip()  
                
                question = f"You are provided a 256*256 satellite image. What is the landuse type in region {coordinates}?"
                answer = f"{landuse_type}"
                
                print(f"Q: {question}")
                print(f"A: {answer}")
                with open(work_dir + f"rs_landuse_description_{zl}.jsonl", "a") as fout:
                    value = {
                    "img_name": img_name,
                    "Q": f"You are provided a 256*256 satellite image. What is the landuse type in region {coordinates}?",
                    "A":f"{landuse_type}"
                }
                    fout.write(json.dumps(value, ensure_ascii=False) + "\n")
            print(f"Finished generating rs_landuse_description_{zl}.jsonl")