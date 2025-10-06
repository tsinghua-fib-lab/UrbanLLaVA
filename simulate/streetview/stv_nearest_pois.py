import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from math import radians, sin, cos, asin, sqrt
import argparse



import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import KDTree

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir        
    for zl in ['zl15', 'zl17']:
        csv_file = work_dir + f"dev-{city}/stv_in_sat_{city}_{zl}.csv"

        output_dir = work_dir + f"dev-{city}/"

        csv_data = pd.read_csv(csv_file)

        csv_data['geometry'] = csv_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        csv_gdf = gpd.GeoDataFrame(csv_data, geometry='geometry', crs="EPSG:4326")

        # TODO: Change the following paths to the actual paths
        if city == "Beijing":
            geojson_file = "....../MLLM-wudaokou_new/make_dataset/beijing_pois_five_ring.geojson"
        elif city == "London":
            geojson_file = "....../ThreeCityImage/city_geojson/London_geojson/London_pois_five_ring.geojson"
        elif city == "NewYork":
            geojson_file = "....../ThreeCityImage/city_geojson/NewYork_geojson/NewYork_pois_five_ring.geojson"

        assert geojson_file is not None, "Please specify the path to the GeoJSON file."

        geojson_gdf = gpd.read_file(geojson_file)

        points_gdf = geojson_gdf[geojson_gdf.geometry.type == 'Point'].copy()
        polygons_gdf = geojson_gdf[geojson_gdf.geometry.type == 'Polygon'].copy()

        polygons_gdf['geometry'] = polygons_gdf['geometry'].centroid

        combined_gdf = pd.concat([points_gdf, polygons_gdf])

        geojson_coords = [(geom.x, geom.y) for geom in combined_gdf.geometry]

        tree = KDTree(geojson_coords)

        results = []

        for idx, row in csv_gdf.iterrows():
            point_coords = (row.geometry.x, row.geometry.y)
            
            distances, indices = tree.query(point_coords, k=20)
            
            for distance, index in zip(distances, indices):
                nearest_feature = combined_gdf.iloc[index]
                feature_name = nearest_feature.get('name', 'Unknown')  
                
                results.append({
                    'image_name': row.image_name,
                    'csv_latitude': row.geometry.y,
                    'csv_longitude': row.geometry.x,
                    'nearest_feature_name': feature_name,
                    'nearest_feature_type': nearest_feature.geometry.type,
                    'distance': distance
                })

        output_df = pd.DataFrame(results)
        # output_df.to_csv(output_dir + "stv_in_sat_zl17_wudaokou_nearest_features.csv", index=False)
        output_df.to_csv(output_dir + f"stv_in_sat_nearest_features_{city}_{zl}.csv", index=False)

        print("Results saved to:", output_dir + f"stv_in_sat_nearest_features_{city}_{zl}.csv")