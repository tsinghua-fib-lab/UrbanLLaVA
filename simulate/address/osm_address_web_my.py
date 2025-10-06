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

    working_dir = work_dir + f'dev-{city}/'

    for zl in ['zl15','zl17']:
        input_path = working_dir + f'SAT_interpolate_{city}_{zl}.csv'
        assert os.path.exists(input_path)
        output_path = working_dir + f'SAT_interpolate_address_{city}_{zl}.csv'

        df = pd.read_csv(input_path)
        df['adr'] = 's'

        for i in trange(len(df)):
            lng = (df.at[i,'lng'])
            lat = (df.at[i,'lat'])

            try:
                address = reverse_geocode(lat, lng)
                df.at[i,'adr'] = str(address)
            except Exception as e:
                pass
                # continue
        df.to_csv(output_path, index=False)