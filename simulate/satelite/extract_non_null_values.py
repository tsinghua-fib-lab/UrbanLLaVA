import json
import os
import glob
import pandas as pd
import os
import argparse

def extract_non_null_keys(data):
    non_null_keys = []
    for key, value in data.items():
        if value is not None:
            non_null_keys.append(key)
    return non_null_keys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f"dev-{city}/"

    f_list = glob.glob(working_dir + 'clipped_results_pixel_non_null/*.txt')
    # f_list = glob.glob('clipped_results_wudaokou_pixel_non_null_zl17/*.txt')
    for f in f_list:
        os.remove(f)

    for zl in ['zl15','zl17']:

        df = pd.read_csv(working_dir + f'SAT_{city}_'+zl+'.csv')
        # os.makedirs("clipped_results_wudaokou_pixel_non_null_zl17", exist_ok=True)
        os.makedirs(working_dir + 'clipped_results_pixel_non_null_'+zl, exist_ok=True)

        for cnt in range(len(df)):
            top_left = (df.at[cnt,'tl_lat'],df.at[cnt,'tl_lng'])
            bottom_right = (df.at[cnt,'bt_lat'],df.at[cnt,'bt_lng'])
            img_name = df.at[cnt,'img_name'].split('.')[0]
            for typ in ['buildings','driving','pois','landuse']:

        # # Load the JSON data
        # with open("clipped_buildings_12409_26975.png_pixel.geojson", "r") as f:
        #     data = json.load(f)
                # if not os.path.exists("clipped_results_wudaokou_pixel_zl17/clipped_"+typ+"_"+img_name+"_pixel.geojson"):
                if not os.path.exists(working_dir + "clipped_results_pixel_"+zl+"/clipped_"+typ+"_"+img_name+"_pixel.geojson"):
                    continue
                # with open("clipped_results_wudaokou_pixel_zl15/clipped_"+typ+"_"+img_name+"_pixel.geojson", "r", encoding="utf-8") as f:
                # with open("clipped_results_wudaokou_pixel_zl17/clipped_"+typ+"_"+img_name+"_pixel.geojson", "r", encoding="utf-8") as f:
                with open(working_dir + "clipped_results_pixel_"+zl+"/clipped_"+typ+"_"+img_name+"_pixel.geojson", "r", encoding="utf-8") as f:
                    data = json.load(f)
                for i in range(len(data['features'])):
                # print(len(data))
                    # print(data['features'][i]['properties'])
                    # print(data['features'][i]['geometry']['coordinates'][0])
                    # Extract non-null keys
                    result = extract_non_null_keys(data['features'][i]['properties'])
                    # print(result)
                    
                    if typ == 'driving':
                        road_name = ''
                        road_type = ''
                        road_length = ''
                        for r in result:
                            if r == 'name':# not in ['building','id','timestamp', 'version', 'osm_type','name:zh-Hans', 'name:zh-Hant']:
                                road_name  = data['features'][i]['properties'][r]
                            elif r == 'highway':
                                road_type =  data['features'][i]['properties'][r]
                            elif r == 'length':
                                road_length =  data['features'][i]['properties'][r]                
                        # with open('clipped_results_wudaokou_pixel_non_null_zl15/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                        # with open('clipped_results_wudaokou_pixel_non_null_zl17/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                        with open(working_dir + 'clipped_results_pixel_non_null_'+zl+'/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                            file.write(road_type+' road '+road_name+' is'+' at location: '+str(data['features'][i]['geometry']['coordinates'])+'\n')

                    elif typ == 'buildings':
                        for r in result:
                            if r == 'name':
                                # with open('clipped_results_wudaokou_pixel_non_null_zl15/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                                # with open('clipped_results_wudaokou_pixel_non_null_zl17/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                                with open(working_dir + 'clipped_results_pixel_non_null_'+zl+'/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                                    file.write(data['features'][i]['properties'][r] +' is at location: '+str(data['features'][i]['geometry']['coordinates'])+'\n')
                    elif typ == 'landuse':
                        land_name = ''
                        land_type = ''
                        for r in result:
                            if r == 'name':
                                land_name = data['features'][i]['properties'][r]
                            if r == 'landuse':
                                land_type = data['features'][i]['properties'][r]
                        # with open('clipped_results_wudaokou_pixel_non_null_zl15/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                        # with open('clipped_results_wudaokou_pixel_non_null_zl17/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                        with open(working_dir + 'clipped_results_pixel_non_null_'+zl+'/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                            file.write(land_type+' region '+land_name+' is at location: '+str(data['features'][i]['geometry']['coordinates'])+'\n')
                        # print(data['features'][i]['properties'][r]+': '+str(data['features'][i]['geometry']['coordinates'][0]))

                    elif typ == 'pois':
                        poi_name = ''
                        poi_type = ''
                        for r in result:
                            if r == 'name':
                                poi_name = data['features'][i]['properties'][r]
                            # elif r not in ["opening_hours","osm_type", "operator", "phone" ,"ref","url", "website","name:en", "name:zh","brand:wikidata", "brand:wikipedia","brand:en"] \
                            #     and data['features'][i]['properties'][r] not in ['no','yes'] and not str(data['features'][i]['properties'][r]).isdigit():
                            #data['features'][i]['properties'][r] not in ['node','way']:
                            elif r in ["amenity", "atm", "bicycle_parking", "bar", "drinking_water", "fast_food", "internet_access", "landuse", "office", "parking", "social_facility", "bicycle", "clothes", "organic", "religion", "shop", "trade", "water", "information", "museum", "tourism",  "fountain", "school", "theatre", "attraction", "zoo"]:
                                poi_type = data['features'][i]['properties'][r]
                                #  print(poi_type)
                                #  print(r)
                                break
                        if poi_name != '':
                            # with open('clipped_results_wudaokou_pixel_non_null_zl15/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                            # with open('clipped_results_wudaokou_pixel_non_null_zl17/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                            with open(working_dir + 'clipped_results_pixel_non_null_'+zl+'/'+typ+"_"+img_name+'.txt', "a", encoding="utf-8") as file:
                                file.write(poi_name+' '+poi_type+' is at location: '+str(data['features'][i]['geometry']['coordinates'])+'\n')
