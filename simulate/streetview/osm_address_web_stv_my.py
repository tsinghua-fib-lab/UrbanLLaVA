import os
from tqdm import trange

import pandas as pd 
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import argparse

def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="MyGeocodingApp2",timeout=1,proxies="http://127.0.0.1:10190")  
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)  
    location = geocode((lat, lon), exactly_one=True,language='en')
    return location.address if location else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir    

    for zl in ['zl15','zl17']:
        working_dir = work_dir + f'dev-{city}/'

        input_path= working_dir + f'stv_in_sat_{city}_{zl}.csv'
        assert os.path.exists(input_path)

        output_path = working_dir + f'stv_in_sat_address_deploy_{zl}.csv'

        df = pd.read_csv(input_path)
        df['adr'] = 's'

        for i in trange(len(df)):
            # lng = (df.at[i,'tl_lng']+df.at[i,'bt_lng'])/2
            # lat = (df.at[i,'tl_lat']+df.at[i,'bt_lat'])/2
            # lng = (df.at[i,'lng']),
            # lat = (df.at[i,'lat'])
            lng = (df.at[i,'longitude'])
            lat = (df.at[i,'latitude'])

        # lat, lng= 51.58425973969619,0.13408350251072 # 39.882027527944864, 116.38185151260446
            try:
                address = reverse_geocode(lat, lng)
            # print(address)
                df.at[i,'adr'] = str(address)
            except Exception as e:
                pass
                # continue
        df.to_csv(output_path, index=False)