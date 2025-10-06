import geopandas as gpd
import pandas as pd
import argparse
import os
from tqdm import trange


# Original Implementation assumes that the shapefile and the CSV file are in the same CRS.
# This implementation checks the CRS of both files and reprojects them if necessary.
import geopandas as gpd
import pandas as pd
import json

def process_spatial_join(shp_file, stv_index_path, output_file):
    gdf_polygon = gpd.read_file(shp_file)
    
    if gdf_polygon.crs is None:
        gdf_polygon.set_crs(epsg=4326, inplace=True)  # Assuming WGS84 as default    


    df_points = pd.read_csv(stv_index_path)
    geometry = gpd.points_from_xy(df_points['longitude'], df_points['latitude'])
    gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry)
    
    # Set CRS for points if not already set (assuming WGS84)    
    if gdf_points.crs is None:
        gdf_points.set_crs(epsg=4326, inplace=True)


    if gdf_polygon.crs != gdf_points.crs:
        gdf_points = gdf_points.to_crs(gdf_polygon.crs)

    result = gpd.sjoin(gdf_polygon, gdf_points, how='inner')

    print(result.head())

    # result[['region_nam', 'sid', 'sid_84_long', 'sid_84_lat']].to_csv(output_file, index=False)
    result[['image_name', 'longitude', 'latitude']].to_csv(output_file, index=False)
    print(f"Saved result to {output_file}")
    print(f"Total records: {len(result)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    
    # TODO: Change the path to the actual streetview index file
    stv_index_path = f"....../Tricity/index_{city}.csv"

    for zl in ['zl15', 'zl17']:
        process_spatial_join(f'{work_dir}dev-{city}/SAT_{city}_{zl}.shp', stv_index_path, f'{work_dir}dev-{city}/stv_in_sat_{city}_{zl}.csv')
        print(f"Processed {zl} for {city}")

        # sanity check and copy the stv images
        print(f"Copying streetview images for {city} {zl}")
        # TODO: Change the path to the actual streetview image directory
        stv_img_all_dir = f'....../ThreeCityImage/{city}/StreetView/'

        df = pd.read_csv(f'{work_dir}dev-{city}/stv_in_sat_{city}_{zl}.csv')
        print(f"Total records: {len(df)}")
        target_dir = f'{work_dir}dev-{city}/sampled_stv_images/'
        os.makedirs(target_dir, exist_ok=True)

        for i in trange(len(df)):
            row = df.iloc[i]
            img_name = row["image_name"]
            # Bug here, OS and python treat paths differently
            og_path = f'\"{os.path.join(stv_img_all_dir, img_name)}\"'
            og_path_py = os.path.join(stv_img_all_dir, img_name)
            assert os.path.exists(og_path_py), f"Image {og_path_py} not found"
            new_image_path = f'\"{os.path.join(target_dir, img_name)}\"'
            os.system(f"cp {og_path} {new_image_path}")

        # assert target_dir is not empty
        assert len(os.listdir(target_dir)) > 0, f"No images copied to {target_dir}"   
        print(f"Streetview images copied for {city} {zl}")
        print(f"Total images: {len(os.listdir(target_dir))} in {target_dir}")         
            