# Function: Clip POI/driving/landuse/natural/buildings data from GeoJSON file based on the shapefile of the region.

import geopandas as gpd
import os
from tqdm import tqdm
import argparse

def clip(shp_file, geojson_file, output_dir, typ):

    shp_gdf = gpd.read_file(shp_file)
    geojson_gdf = gpd.read_file(geojson_file)

    if shp_gdf.crs is None:
        shp_gdf = shp_gdf.set_crs(epsg=4326)

    if shp_gdf.crs != geojson_gdf.crs:
        geojson_gdf = geojson_gdf.to_crs(shp_gdf.crs)

    os.makedirs(output_dir, exist_ok=True)

    for index, polygon in shp_gdf.iterrows():
        clipped_gdf = geojson_gdf[geojson_gdf.geometry.intersects(polygon.geometry)]
        
        # output_filename = os.path.join(output_dir, f"clipped_part_{index}.geojson")
        output_filename = os.path.join(output_dir, f"clipped_{typ}_{polygon['region_nam'].split('.')[0]}.geojson")
        
        if not clipped_gdf.empty:
            clipped_gdf.to_file(output_filename, driver="GeoJSON")
            print(f"Saved clipped data to {output_filename}")
        else:
            print(f"No intersecting features for Polygon {index}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir

    for zl in ["zl15", "zl17"]:
        shp_file = os.path.join(work_dir, f'dev-{city}/SAT_{city}_{zl}.shp')
        output_dir = os.path.join(work_dir, f'dev-{city}/clipped_results_{zl}')
        
        for typ in ['buildings','pois','landuse','natural','driving']:
            # TODO: Change the path to the actual geojson file
            geojson_dir = "....../ThreeCityImage/city_geojson_three_cities"
            geojson_file = os.path.join(geojson_dir, f'{city}_{typ}.geojson')
            clip(shp_file, geojson_file, output_dir, typ)
