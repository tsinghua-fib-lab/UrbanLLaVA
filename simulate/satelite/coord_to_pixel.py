# Function: Convert the coordinates in the GeoJSON file to pixel coordinates in the image.
import geopandas as gpd
import json
import pandas as pd
import os
from tqdm import trange
import argparse
from shapely.geometry import shape, Point


# top_left = (lat1, lon1)  
# bottom_right = (lat2, lon2)  

def is_within_bounds(x, y, img_width, img_height):
    return 0 <= x < img_width and 0 <= y < img_height

def latlon_to_pixel(lat, lon, top_left, bottom_right, img_width, img_height):
    lat_range = top_left[0] - bottom_right[0]  
    lon_range = bottom_right[1] - top_left[1]  
    
    x_percent = (lon - top_left[1]) / lon_range
    y_percent = (top_left[0] - lat) / lat_range
    
    x_pixel = int(x_percent * img_width)
    y_pixel = int(y_percent * img_height)
    
    return x_pixel, y_pixel

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir

    working_dir = work_dir + f"dev-{city}/"
    for zl in ['zl15','zl17']:
        os.makedirs(working_dir + "clipped_results_pixel_"+zl, exist_ok=True)
        # df = pd.read_csv('../SAT_BJ_wudaokou_zl15.csv')
        df = pd.read_csv(working_dir + f'SAT_{city}_'+zl+'.csv')
        for cnt in trange(len(df)):
            top_left = (df.at[cnt,'tl_lat'],df.at[cnt,'tl_lng'])
            bottom_right = (df.at[cnt,'bt_lat'],df.at[cnt,'bt_lng'])
            img_name = df.at[cnt,'img_name'].split('.')[0]
        
            for typ in ['buildings','driving','pois','landuse']:
                if not os.path.exists(working_dir + "clipped_results_"+zl+"/clipped_"+typ+"_"+img_name+".geojson"):
                    continue
                with open(working_dir + "clipped_results_"+zl+"/clipped_"+typ+"_"+img_name+".geojson", "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)
                
                features = geojson_data['features']

                # Extract all the polygons
                polygons = []
                for feature in features:
                    geometry = feature['geometry']
                    if geometry['type'] == 'Polygon':
                        polygon_shape = shape(geometry)  
                        polygon_name = feature['properties'].get('name', 'Unnamed Polygon')  
                        polygons.append({'name': polygon_name, 'shape': polygon_shape})

                for feature in features:
                    geometry = feature['geometry']
                    if geometry['type'] == 'Point':
                        point = shape(geometry)  
                        point_name = feature['properties'].get('name', 'Unnamed Point')  
                        
                        for polygon in polygons:
                            if polygon['shape'].contains(point):
                                feature['properties']['name'] = f"{point_name}-{polygon['name']}"
                                break  
                
                
                if not os.path.exists(working_dir + "clipped_results_"+zl+"_updated"):
                    os.makedirs(working_dir + "clipped_results_"+zl+"_updated", exist_ok=True)

                with open(working_dir + "clipped_results_"+zl+"_updated/clipped_"+typ+"_"+img_name+"_updated.geojson", "w", encoding="utf-8") as f:
                    json.dump(geojson_data, f, ensure_ascii=False, indent=4)


            for typ in ['buildings','driving','pois','landuse']:
                # top_left = (39.96870074491694,116.356201171875)
                # bottom_right = (39.9602803542957,116.3671875)
                img_width, img_height = 256, 256 

                # with open("clipped_buildings_12409_26975.png.geojson", "r", encoding="utf-8") as f:
                # with open("clipped_results_wudaokou/clipped_"+typ+"_"+img_name+".geojson", "r", encoding="utf-8") as f:
                if not os.path.exists(working_dir + "clipped_results_"+zl+"_updated/clipped_"+typ+"_"+img_name+"_updated.geojson"):
                    continue
                with open(working_dir + "clipped_results_"+zl+"_updated/clipped_"+typ+"_"+img_name+"_updated.geojson", "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)

                for feature in geojson_data['features']:
                    geometry = feature['geometry']
                    coords = geometry['coordinates']

                    if geometry['type'] == 'Polygon':
                        new_coords = []
                        for ring in coords:
                            filtered_ring = [coord for coord in [latlon_to_pixel(lat, lon, top_left, bottom_right, img_width, img_height) for lon, lat in ring] if is_within_bounds(*coord, img_width, img_height)]
                            if filtered_ring:
                                new_coords.append(filtered_ring)
                        if new_coords:
                            feature['geometry']['coordinates'] = new_coords
                    
                    elif geometry['type'] == 'LineString':
                        filtered_coords = [coord for coord in [latlon_to_pixel(lat, lon, top_left, bottom_right, img_width, img_height) for lon, lat in coords] if is_within_bounds(*coord, img_width, img_height)]
                        if filtered_coords:
                            feature['geometry']['coordinates'] = filtered_coords
                    
                    elif geometry['type'] == 'MultiLineString':
                        new_multilines = []
                        for linestring in coords:
                            filtered_linestring = [coord for coord in [latlon_to_pixel(lat, lon, top_left, bottom_right, img_width, img_height) for lon, lat in linestring] if is_within_bounds(*coord, img_width, img_height)]
                            if filtered_linestring:
                                new_multilines.append(filtered_linestring)
                        if new_multilines:
                            feature['geometry']['coordinates'] = new_multilines

                    if geometry['type'] == 'Point':
                        lat, lon = coords[1], coords[0]
                        x_pixel, y_pixel = latlon_to_pixel(lat, lon, top_left, bottom_right, img_width, img_height)
                        geojson_data['features'][geojson_data['features'].index(feature)]['geometry']['coordinates'] = [x_pixel, y_pixel]


                with open(working_dir + "clipped_results_pixel_"+zl+"/clipped_"+typ+"_"+img_name+"_pixel.geojson", "w") as f:
                    json.dump(geojson_data, f, ensure_ascii=False)