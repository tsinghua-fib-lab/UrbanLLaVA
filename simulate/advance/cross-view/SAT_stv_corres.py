import pandas as pd
import argparse
from tqdm import tqdm, trange

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # Long radius
ee = 0.00669342162296594323  # Square of eccentricity



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f"dev-{city}/"

    for zl in ['zl15','zl17']:
        df_stv = pd.read_csv(working_dir + f'stv_in_sat_{city}_{zl}.csv')
        df_sat = pd.read_csv(working_dir + f'SAT_{city}_{zl}.csv')

        df_stv['x_pixel'] = 0
        df_stv['y_pixel'] = 0
        df_stv['sat'] = 's'

        for i_stv in trange(len(df_stv)):
            lng = df_stv.at[i_stv,'longitude']
            lat = df_stv.at[i_stv,'latitude']

            for i_sat in range(len(df_sat)):
                sat_tl_lat = df_sat.at[i_sat,'tl_lat'] #tl_lat,tl_lng,bt_lat,bt_lng
                sat_tl_lng = df_sat.at[i_sat,'tl_lng']
                sat_bt_lat = df_sat.at[i_sat,'bt_lat']
                sat_bt_lng = df_sat.at[i_sat,'bt_lng']

                y_pixel = int(255*((sat_tl_lat-lat)/(sat_tl_lat-sat_bt_lat)))
                x_pixel = int(255*((lng-sat_tl_lng)/(sat_bt_lng-sat_tl_lng)))
                # print(x_pixel, y_pixel)
                if 0<=x_pixel and x_pixel<=255 and 0<=y_pixel and y_pixel<=255:
                    df_stv.at[i_stv,'x_pixel'] = x_pixel
                    df_stv.at[i_stv,'y_pixel'] = y_pixel
                    df_stv.at[i_stv,'sat_img_name'] = df_sat.at[i_sat,'img_name']
                    break
        df_stv.to_csv(working_dir + f'sat_stv_corr_{city}_{zl}.csv', index=False)
        print(f'{working_dir}sat_stv_corr_{city}_{zl}.csv saved. {len(df_stv)} records processed.')

