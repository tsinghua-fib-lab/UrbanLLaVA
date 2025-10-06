import pandas as pd

def remove_duplicates(points):

    seen = set()
    result = []
    for point in points:
        if tuple(point) not in seen:
            seen.add(tuple(point))
            result.append(point)
    return result

def calculate_centroid(polygon):

    print(polygon)
    polygon = remove_duplicates(polygon)
    
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]

    cx = sum(x_coords) / len(polygon)
    cy = sum(y_coords) / len(polygon)
    return [int(cx), int(cy)]

def is_near(p1, p2, threshold=3):
    return abs(p1[0] - p2[0]) <= threshold and abs(p1[1] - p2[1]) <= threshold

def merge_nearby_points(locations):
    merged = calculate_centroid(locations)

    return merged

def merge_poi_by_category(poi_data):
    merged_poi = {}

    for poi in poi_data:
        name, location = poi.split(" is at location: ")
        category = name.split(" ")[-1]
        location = eval(location.strip())

        processed_locations = []
        # print(location)
        coord = location
        if isinstance(coord[0], list) and len(coord[0])!=2:# and len(coord[0][0]) > 2 and all(isinstance(i, list) and len(i) == 2 for i in coord):
            loc = location[0]
            if isinstance(loc, list) and len(loc) > 1:  
                processed_locations.append(loc)

                if processed_locations:
                    # print(processed_locations)
                    merged_points = merge_nearby_points(processed_locations[0])
                    # print(merged_points)

                    if category not in merged_poi:
                        merged_poi[category] = []
                    merged_poi[category].extend([merged_points])
                else:
                    if category not in merged_poi:
                        merged_poi[category] = []
                    merged_poi[category].append(location)  
        else:
            if category not in merged_poi:
                merged_poi[category] = []
            merged_poi[category].append(location)  


    return merged_poi

def format_merged_poi(merged_poi):
    output = []
    for category, locations in merged_poi.items():
        output.append(f"{category}s are at locations: {', '.join(str(loc) for loc in locations)}")
    return output

def read_poi_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_poi_to_file(filename, formatted_output):
    with open(filename, 'w', encoding='utf-8') as file:
        for line in formatted_output:
            file.write(line + '\n')

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
        df = pd.read_csv(working_dir + f'SAT_{city}_'+zl+'.csv')
        for cnt in range(len(df)):
            img_name = df.at[cnt,'img_name'].split('.')[0]
            output_dir = working_dir + "short_clipped_results_"+zl
            os.makedirs(output_dir, exist_ok=True)
            input_filename = working_dir + 'clipped_results_pixel_non_null_'+zl+'/pois_'+img_name+'.txt' 

            if not os.path.exists(input_filename):
                continue
            poi_data = read_poi_from_file(input_filename)

            merged_poi = merge_poi_by_category(poi_data)
            formatted_output = format_merged_poi(merged_poi)

            output_filename = working_dir + 'short_clipped_results_'+zl+'/pois_'+img_name+'.txt'  
            write_poi_to_file(output_filename, formatted_output)

