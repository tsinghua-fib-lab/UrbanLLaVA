import os
import pandas as pd
import argparse
import tqdm 
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir
    working_dir = work_dir + f"dev-{city}/"

    for zl in ['zl15','zl17']:

        # df = pd.read_csv('sat_stv_corr_'+zl+'_'+area+'.csv')
        df = pd.read_csv(working_dir + f'sat_stv_corr_{city}_{zl}.csv')

        df['partition'] = 's'
        df['x_min'] = 0
        df['x_max'] = 0
        df['y_min'] = 0
        df['y_max'] = 0

        for i in trange(len(df)):
            x_pixel = df.at[i,'x_pixel']
            y_pixel = df.at[i,'y_pixel']
            if x_pixel<=127:
                if y_pixel<=127:
                    df.at[i,'partition'] = 'Top_left'
                else:
                    df.at[i,'partition'] = 'Bottom_left'
            else:
                if y_pixel<=127:
                    df.at[i,'partition'] = 'Top_right'
                else:
                    df.at[i,'partition'] = 'Bottom_right'
            df.at[i,'x_min'] = max(0,x_pixel-10)
            df.at[i,'x_max'] = min(255,x_pixel+10)
            df.at[i,'y_min'] = max(0,y_pixel-10)
            df.at[i,'y_max'] = min(255,y_pixel+10)

        # df.to_csv('sat_stv_corr_'+zl+'_'+area+'_partition.csv',index=False)
        df.to_csv(working_dir + f'sat_stv_corr_{city}_{zl}_partition.csv',index=False)
        print(f'{working_dir}sat_stv_corr_{city}_{zl}_partition.csv saved. {len(df)} records processed.')

