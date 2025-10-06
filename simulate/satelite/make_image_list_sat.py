# Function: Copy the satellite images in the given area to the sample_sat_image_zl15 and sample_sat_image_zl17 folders, and generate the corresponding csv file about their names and boundary information.

import glob
import pandas as pd
import math
import os
from tqdm import tqdm, trange
import argparse
import shutil

def deg2num(lat_deg, lon_deg, zoom):
    """
    Converts latitude/longitude to tile numbers at a given zoom level.

    Args:
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.
        zoom (int): Zoom level.

    Returns:
        tuple: A tuple (xtile, ytile) representing the tile numbers.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """
    Converts tile coordinates (xtile, ytile) and zoom level to latitude and longitude.
    
    y_x.png image, num2deg(x+1,y+1,zoom) is the bottom right corner of the image
    
    y_x.png image, num2deg(x,y,zoom) is the top left corner of the image

    Args:
        xtile (int): The x coordinate of the tile.
        ytile (int): The y coordinate of the tile.
        zoom (int): The zoom level.

    Returns:
        tuple: A tuple containing:
            - lat_deg (float): The latitude in degrees.
            - lon_deg (float): The longitude in degrees.
    """
    # y_x.png image, num2deg(x+1,y+1,zoom) is the bottom right corner of the image
    # y_x.png image, num2deg(x,y,zoom) is the top left corner of the image

    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    task = args.city
    work_dir = args.work_dir


# task = "Beijing" # "London" "New York" "Beijing"

# TODO: Change the following paths to the actual paths

    if task == "NewYork":
        satelite_zl15_source_img_folder = '......ThreeCityImage/NewYork/Sat_zl15/'
        satelite_zl17_source_img_folder = '......ThreeCityImage/NewYork/Sat_zl17/'
        satelite_zl15_target_img_folder = work_dir + 'dev-NewYork/sample_sat_image_zl15/'
        satelite_zl17_target_img_folder = work_dir + 'dev-NewYork/sample_sat_image_zl17/'
        satelite_zl15_csv_file = work_dir + 'dev-NewYork/SAT_NewYork_zl15.csv'
        satelite_zl17_csv_file = work_dir + 'dev-NewYork/SAT_NewYork_zl17.csv'
        Lat_tl = 40.85
        Lng_tl = -74.02
        Lat_bt = 40.70
        Lng_bt = -73.90

    elif task == "London":
        satelite_zl15_source_img_folder = '......ThreeCityImage/London/Sat_zl15/'
        satelite_zl17_source_img_folder = '......ThreeCityImage/London/Sat_zl17/'
        satelite_zl15_target_img_folder = work_dir + 'dev-London/sample_sat_image_zl15/'
        satelite_zl17_target_img_folder = work_dir + 'dev-London/sample_sat_image_zl17/'
        satelite_zl15_csv_file = work_dir + 'dev-London/SAT_London_zl15.csv'
        satelite_zl17_csv_file = work_dir + 'dev-London/SAT_London_zl17.csv'
        Lat_tl = 51.53
        Lng_tl = -0.13
        Lat_bt = 51.49
        Lng_bt = -0.07

    elif task == "Beijing":
        satelite_zl15_source_img_folder = '......ThreeCityImage/Beijing/Sat_zl15/'
        satelite_zl17_source_img_folder = '......ThreeCityImage/Beijing/Sat_zl17/'
        satelite_zl15_target_img_folder = work_dir + 'dev-Beijing/sample_sat_image_zl15/'
        satelite_zl17_target_img_folder = work_dir + 'dev-Beijing/sample_sat_image_zl17/'
        satelite_zl15_csv_file = work_dir + 'dev-Beijing/SAT_Beijing_zl15.csv'
        satelite_zl17_csv_file = work_dir + 'dev-Beijing/SAT_Beijing_zl17.csv'
        Lat_tl = 40.03
        Lng_tl = 116.26
        Lat_bt = 39.96
        Lng_bt = 116.40



    if not os.path.exists(satelite_zl15_target_img_folder):
        os.makedirs(satelite_zl15_target_img_folder)
    else:
        shutil.rmtree(satelite_zl15_target_img_folder)
        os.makedirs(satelite_zl15_target_img_folder)

    if not os.path.exists(satelite_zl17_target_img_folder):
        os.makedirs(satelite_zl17_target_img_folder)
    else:
        shutil.rmtree(satelite_zl17_target_img_folder)
        os.makedirs(satelite_zl17_target_img_folder)

    # Find the boundary of the area in form of tile number
    # zoom level 15, whic is 5m resolution
    XT1,YT1 = deg2num(Lat_tl+0.01,Lng_tl+0.01,15)  
    XT2,YT2 = deg2num(Lat_bt+0.01,Lng_bt+0.01,15)
    print(deg2num(Lat_tl+0.01,Lng_tl+0.01,15))
    print(deg2num(Lat_bt+0.01,Lng_bt+0.01,15))

    # boundary of zl15
    for x in trange(XT1,XT2):
        for y in range(YT1,YT2):
            image_name = str(y)+'_'+str(x)
            if not os.path.exists(os.path.join(satelite_zl15_source_img_folder,image_name+'.png')):
                continue
            shutil.copy(os.path.join(satelite_zl15_source_img_folder,image_name+'.png'),os.path.join(satelite_zl15_target_img_folder,image_name+'.png'))

    # zoom level 17, which is 1m resolution
    XT1,YT1 = deg2num(Lat_tl+0.01,Lng_tl+0.01,17)
    XT2,YT2 = deg2num(Lat_bt+0.01,Lng_bt+0.01,17)
    print(deg2num(Lat_tl+0.01,Lng_tl+0.01,17))
    print(deg2num(Lat_bt+0.01,Lng_bt+0.01,17))
    # boundary of zl17
    for x in trange(XT1,XT2):
        for y in range(YT1,YT2):
            image_name = str(y)+'_'+str(x)
            if not os.path.exists(os.path.join(satelite_zl17_source_img_folder,image_name+'.png')):
                continue
            shutil.copy(os.path.join(satelite_zl17_source_img_folder,image_name+'.png'),os.path.join(satelite_zl17_target_img_folder,image_name+'.png'))


    img_list = glob.glob(satelite_zl15_target_img_folder+'*.png')

    img_name = [x.split('/')[-1] for x in img_list]

    pd_dict = pd.DataFrame({'img_name':img_name})
    pd_dict.to_csv(satelite_zl15_csv_file, index=False)

    data = pd.read_csv(satelite_zl15_csv_file)
    data['tl_lat'] = 0.0 
    # top left lat
    data['tl_lng'] = 0.0
    data['bt_lat'] = 0.0 
    data['bt_lng'] = 0.0 
    # bottom right lng

    for i in trange(len(data)):
        sat_name = data.at[i,'img_name'].split('.')[0]
        y_tile = sat_name.split('_')[0]
        x_tile = sat_name.split('_')[1]
        tl_lat,tl_lng = num2deg(int(x_tile),int(y_tile),15)
        bt_lat,bt_lng = num2deg(int(x_tile)+1,int(y_tile)+1,15)

        data.at[i,'tl_lat'] = tl_lat
        data.at[i,'tl_lng'] = tl_lng
        data.at[i,'bt_lat'] = bt_lat
        data.at[i,'bt_lng'] = bt_lng

    data.to_csv(satelite_zl15_csv_file, index=False)
    # print("For SAT_BJ_wudaokou_zl15.csv, the number of images is ",len(data))
    print("For ",satelite_zl15_csv_file," the number of images is ",len(data))

    ######################################################################################
    img_list = glob.glob(satelite_zl17_target_img_folder+'*.png')

    img_name = [x.split('/')[-1] for x in img_list]

    pd_dict = pd.DataFrame({'img_name':img_name})
    pd_dict.to_csv(satelite_zl17_csv_file, index=False)

    data = pd.read_csv(satelite_zl17_csv_file)
    data['tl_lat'] = 0.0
    data['tl_lng'] = 0.0
    data['bt_lat'] = 0.0 
    data['bt_lng'] = 0.0

    for i in trange(len(data)):
        sat_name = data.at[i,'img_name'].split('.')[0]
        y_tile = sat_name.split('_')[0]
        x_tile = sat_name.split('_')[1]
        tl_lat,tl_lng = num2deg(int(x_tile),int(y_tile),17)
        bt_lat,bt_lng = num2deg(int(x_tile)+1,int(y_tile)+1,17)

        data.at[i,'tl_lat'] = tl_lat
        data.at[i,'tl_lng'] = tl_lng
        data.at[i,'bt_lat'] = bt_lat
        data.at[i,'bt_lng'] = bt_lng

    data.to_csv(satelite_zl17_csv_file, index=False)

    print("For ",satelite_zl17_csv_file," the number of images is ",len(data))