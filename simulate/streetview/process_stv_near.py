import pandas as pd
import argparse

def process_stv_near(input_file, output_file):
    df = pd.read_csv(input_file)

    result = pd.DataFrame(columns=['image_name', 'feature_names'])

    for image_name, group in df.groupby('image_name'):
        feature_names = group['nearest_feature_name'].dropna().head(10)
        
        feature_names_str = ','.join(feature_names)
        
        result_tmp = pd.DataFrame({'image_name': [image_name], 'feature_names': [feature_names_str]})
        result = pd.concat([result, result_tmp], ignore_index=True)

    result.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir        
    for zl in ['zl15', 'zl17']:
        input_path = work_dir + f'dev-{city}/stv_in_sat_nearest_features_{city}_{zl}.csv'
        output_path = work_dir + f'dev-{city}/stv_in_sat_nearest_features_update_{city}_{zl}.csv'
        process_stv_near(input_path, output_path)
