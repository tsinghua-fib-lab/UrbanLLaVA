import ast

def parse_input_txt(input_txt):
    region_data = []
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if "is at location:" in line:
                region_type = line.split("is at location:")[0].strip()
                try:
                    coordinates = ast.literal_eval(line.split("is at location:")[1].strip())
                    region_data.append((region_type, coordinates))
                except (SyntaxError, ValueError):
                    print(f"Error parsing coordinates in line: {line}")
    return region_data

def is_valid_polygon(coordinates):
    if len(coordinates) < 4:  
        return False
    if coordinates[0] == coordinates[-1]:  
        if len(coordinates) == 3:  
            return False
    return True

def filter_invalid_regions(region_data):
    valid_regions = []
    for region_type, coordinates in region_data:
        for polygon in coordinates:
            if is_valid_polygon(polygon):
                valid_regions.append((region_type, polygon))
    return valid_regions

def write_output_txt(output_txt, region_data):
    with open(output_txt, 'w', encoding='utf-8') as f:
        for region_type, coordinates in region_data:
            f.write(f"{region_type} is at location: {coordinates}\n")


import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f"dev-{city}/"    
    for zl in ['zl17','zl15']:
        df = pd.read_csv( working_dir + f'SAT_{city}_'+zl+'.csv')
        for cnt in range(len(df)):
            img_name = df.at[cnt,'img_name'].split('.')[0]
            output_dir = working_dir +"short_clipped_results_"+zl
            os.makedirs(output_dir, exist_ok=True)
            # input_txt = "clipped_results_wudaokou_pixel_non_null/landuse_"+img_name+".txt"
            input_txt = working_dir + "clipped_results_pixel_non_null_"+zl+"/landuse_"+img_name+".txt"


            if not os.path.exists(input_txt):
                continue
            # output_txt = "short_clipped_results_wudaokou/landuse_"+img_name+".txt"
            output_txt = working_dir + "short_clipped_results_"+zl+"/landuse_"+img_name+".txt"

            region_data = parse_input_txt(input_txt)

            try:
                valid_region_data = filter_invalid_regions(region_data)
            except:
                continue

            write_output_txt(output_txt, valid_region_data)

        # print(f"Data has been processed and saved to {output_txt}")
