# # Function: Generate shapefile for satellite images in order to visualize them in GIS.

import shapefile  # Using pyshp
import pandas as pd
import argparse
import os

def make_sat_shp(data_address, csv_address) -> None:
    sat = pd.read_csv(csv_address, header=0, sep=',')
    file = shapefile.Writer(data_address)
    file.field('num')
    file.field('region_name')
    file.field('type', 'C', '40')

    for i in range(len(sat)):
        # Extract image name
        img_name = sat.at[i, 'img_name']
        
        # Define the polygon coordinates
        polygon = [
            [sat.at[i, 'tl_lng'], sat.at[i, 'tl_lat']],
            [sat.at[i, 'bt_lng'], sat.at[i, 'tl_lat']],
            [sat.at[i, 'bt_lng'], sat.at[i, 'bt_lat']],
            [sat.at[i, 'tl_lng'], sat.at[i, 'bt_lat']],
            [sat.at[i, 'tl_lng'], sat.at[i, 'tl_lat']]  # Close the polygon
        ]
        
        # Add the polygon to the shapefile
        file.poly([polygon])
        file.record(str(i), img_name, 'Polygon')

    file.close()

    # Write the projection file with WKT for EPSG:4326
    wkt = """GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""

    # Write the WKT to the .prj file
    with open(data_address.replace(".shp", ".prj"), 'w') as f:
        f.write(wkt)

    print(f"Shapefile and projection file have been created at {data_address}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')    
    args = parser.parse_args()

    city = args.city
    work_dir = args.work_dir

    for zl in ["zl15", "zl17"]:
        csv_path = os.path.join(work_dir, f'dev-{city}/SAT_{city}_{zl}.csv')
        shp_path = os.path.join(work_dir, f'dev-{city}/SAT_{city}_{zl}.shp')
        make_sat_shp(shp_path, csv_path)