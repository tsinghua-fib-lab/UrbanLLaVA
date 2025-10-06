import re
from tqdm import trange
import argparse

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def merge_segments(road_segments):
    merged_segments = []
    for segment in road_segments:
        if not merged_segments:
            merged_segments.append(segment)
        else:
            if merged_segments[-1][-1] == segment[0]:
                merged_segments[-1].extend(segment[1:])
            else:
                merged_segments.append(segment)

    simplified_segments = []
    for seg in merged_segments:
        if len(seg) > 1:  
            simplified_segments.append([seg[0], seg[-1]])
    
    return simplified_segments

def process_road_data(road_data):
    compressed_data = {}
    
    for road_name, segments in road_data.items():
        merged_segments = merge_segments(segments)
        
        if merged_segments:
            compressed_data[road_name] = merged_segments

    return compressed_data

def parse_input_txt(input_txt):
    road_data = {}
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if "is at location:" in line:
                road_name = line.split("is at location:")[0].strip()
                coordinates = eval(line.split("is at location:")[1].strip())
                road_data[road_name] = coordinates
    return road_data

def write_output_txt(output_txt, compressed_data):
    with open(output_txt, 'w', encoding='utf-8') as f:
        for road_name, segments in compressed_data.items():
            for seg in segments:
                f.write(f"{road_name} from {seg[0]} to {seg[1]}\n")

import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f"dev-{city}/"
    for zl in ['zl17','zl15']:
        # df = pd.read_csv('../SAT_BJ_wudaokou_zl15.csv')
        df = pd.read_csv(working_dir + f'SAT_{city}_'+zl+'.csv')
        for cnt in trange(len(df)):
            img_name = df.at[cnt,'img_name'].split('.')[0]

            output_dir = working_dir + "short_clipped_results_"+zl
            os.makedirs(output_dir, exist_ok=True)
            input_txt = working_dir + "clipped_results_pixel_non_null_"+zl+"/driving_"+img_name+".txt"
            if not os.path.exists(input_txt):
                continue

            output_txt = working_dir + "short_clipped_results_"+zl+"/driving_"+img_name+".txt"

            road_data = parse_input_txt(input_txt)

            compressed_data = process_road_data(road_data)

            write_output_txt(output_txt, compressed_data)

        # print(f"Data has been processed and saved to {output_txt}")
