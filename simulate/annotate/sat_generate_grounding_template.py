import random
import re
import math
import argparse
import pandas as pd

def read_locations(filename):
    locations = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r'(.*?)\sis\sat\slocation:\s(\[\[\[.*?\]\]\])', line)
            if match:
                name = match.group(1)
                try:
                    coordinates = eval(match.group(2))
                    locations.append((name, coordinates))
                except:
                    return -1
    return locations

def get_two_random_locations(locations):
    loc1, loc2 = random.sample(locations, 2)
    return loc1, loc2

def calculate_centroid(coordinates):
    print(coordinates)
    x_sum = sum(point[0] for point in coordinates[0])
    y_sum = sum(point[1] for point in coordinates[0])
    num_points = len(coordinates[0])
    return (x_sum / num_points, y_sum / num_points)

def calculate_direction(coord1,coord2):
    dx = coord2[0] - coord1[0]  
    dy = coord2[1] - coord1[1] 
    
    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
        return "Same location"
    
    if dy > 0:
        if dx > 0:
            return "Southeast (SE)"
        elif dx < 0:
            return "Southwest (SW)"
        else:
            return "South (S)"
    elif dy < 0:
        if dx > 0:
            return "Northeast (NE)"
        elif dx < 0:
            return "Northwest (NW)"
        else:
            return "North (N)"
    else:  # dy == 0
        if dx > 0:
            return "East (E)"
        elif dx < 0:
            return "West (W)"
        else:
            return "Same location"


def calculate_distance(coord1, coord2, x_dist, y_dist):
    print(coord1,coord2)
    print((coord1[0] - coord2[0])/256,(coord1[1] - coord2[1])/256)
    print(((coord1[0] - coord2[0])/256*x_dist)**2 , ((coord1[1] - coord2[1])/256*y_dist)**2)
    return math.sqrt(((coord1[0] - coord2[0])/256*x_dist)**2 + ((coord1[1] - coord2[1])/256*y_dist)**2)

from haversine import haversine


import os
import pandas as pd
import json
def main(working_dir:str, zl:str, city:str):
    df = pd.read_csv(working_dir + f'SAT_{city}_{zl}.csv')
    output = []
    for cnt in range(len(df)):
        top_left = (df.at[cnt,'tl_lat'],df.at[cnt,'tl_lng'])
        bottom_right = (df.at[cnt,'bt_lat'],df.at[cnt,'bt_lng'])
        img_name = df.at[cnt,'img_name'].split('.')[0]

        if not os.path.exists(working_dir + f'clipped_results_pixel_non_null_{zl}/pois_'+img_name+'.txt'):
            continue
        locations = read_locations(working_dir + f'clipped_results_pixel_non_null_{zl}/pois_'+img_name+'.txt')#pois_12408_26978.txt')
        if locations == -1:
            continue
        x_dist = haversine((df.at[cnt,'tl_lat'],df.at[cnt,'tl_lng']),(df.at[cnt,'tl_lat'],df.at[cnt,'bt_lng']), unit='m')
        y_dist = haversine((df.at[cnt,'tl_lat'],df.at[cnt,'tl_lng']),(df.at[cnt,'bt_lat'],df.at[cnt,'tl_lng']), unit='m')
        # print(x_dist, y_dist)
        
        if locations==None or len(locations)<2:
            continue
        loc1, loc2 = get_two_random_locations(locations)
        

        
        centroid1 = calculate_centroid(loc1[1])
        centroid2 = calculate_centroid(loc2[1])
        
        distance = calculate_distance(centroid1, centroid2,x_dist,y_dist)
        direction = calculate_direction(centroid1, centroid2)
        print(f"Distance between {loc1[0]} and {loc2[0]}: {direction} {distance:.2f} units")
        print('Find a(n)', str(loc2[0]).split(' ')[1]+ ' located '+ str(distance)+ ' meters in ' +str(direction) +' from '+str(loc1[0])+' '+str(loc1[1][0]))
        print(str(loc2[0])+', and the location is at '+str(loc2[1][0]))

        output.append({
            "img_name": img_name,
            "Q": 'Find a(n) '+ str(loc2[0]).split(' ')[1]+ ' located '+ str(distance)+ ' meters in ' +str(direction) +' from '+str(loc1[0])+' '+str(loc1[1][0]),
            "A": str(loc2[0])+', and the location is at '+str(loc2[1][0])
        })

    # save to csv
    df = pd.DataFrame(output)
    df.to_csv(working_dir + f'rs_grounding_selfmade_{zl}.csv', index=False)
    print(f"Finished generating rs_grounding_selfmade_{zl}.csv")



# if os.path.exists("rs_grounding_selfmade.jsonl"):
#     os.remove("rs_grounding_selfmade.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir            
    
    # working_dir = '../../data/dev-Beijing/'
    working_dir = work_dir + f'dev-{city}/'

    for zl in ['zl15', 'zl17']:
        if os.path.exists(working_dir + f"rs_grounding_selfmade_{zl}.csv"):
            os.remove(working_dir + f"rs_grounding_selfmade_{zl}.csv") 
            print("File removed")
        main(working_dir, zl, city)
