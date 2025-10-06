import os
from typing import Optional, List

import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split


def match_pos(city_poi, city_checkin):
    mid1 = {} 
    result = []
    for idx, row in city_poi.iterrows():
        venue = row['Venue ID']
        key = venue
        if row.size < 5:
            continue
        if key not in mid1:
            mid1[key] = {"lat":row['Latitude'],'lon':row['Longitude']}
        else:
            print('poi venue:', key)
    for idx, row in city_checkin.iterrows():
        venue = row['Venue ID']
        user = row['User ID']
        key = f"{user}_{venue}"
        if row.size < 4:
            continue
        result.append({
            "user_id":row['User ID'],
            "location":row['Venue ID'], 
            "utc":row['UTC Time'],
            "offset":row['Timezone Offset'],
            "Latitude":mid1[venue]["lat"],
            'Longitude': mid1[venue]["lon"]
        })
        # else:
            # print('checkin venue:', key)
    result_df = pd.DataFrame(result)
    return result_df
def get_day(day):
    tmp = {'Mon': 'Monday', 'Tue': 'Tuesday', 'Wed': 'Wednesday',
           'Thu': 'Thursday', 'Fri': 'Friday', 'Sat': 'Saturday', 'Sun': 'Sunday'}
    pro_day = tmp[day.split()[0]]
    return pro_day
def get_time(time):
    pre_time = time.split()[3]
    hour = int(pre_time.split(':')[0])
    min = int(pre_time.split(':')[1])
    pro_time = 60*hour + min
    return pro_time

def get_start_day(time):
    mon2day_13 = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    mon2day_12 = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    mon2order = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    year = int(time.split()[-1])
    mon = time.split()[1]
    day = int(time.split()[2])
    day_mon = 0
    order = mon2order[mon]
    if year == 2012:
        for i in range(order):
            day_mon += mon2day_12[i+1]
    elif year == 2013:
        for i in range(mon2order[mon]):
            day_mon += mon2day_13[i+1]
    else:
        print("other year:",year)
    day_mon += day
    return day_mon
def get_dup(group,threshold): 
    cnt_set = set()
    for idx, row in group.iterrows():
        venue = row['location_id']
        user = row['user_id']
        key = f"{user}_{venue}"
        if key not in cnt_set:
            cnt_set.add(key)
    if (group.shape[0]-len(cnt_set))/group.shape[0] > threshold:
        return True
    else:
        return False

def get_split(data, city, output_path, threshold, context_step, historical_step):
    data['utc'] = pd.to_datetime(data['utc'])
    data.sort_values(by=['user_id', 'utc'], inplace=True)

    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for user, group in data.groupby('user_id'):
        if group.shape[0] < context_step + historical_step + 5:
            continue
        if not get_dup(group=group, threshold=threshold):
            continue
        train, temp = train_test_split(group, test_size=0.4, shuffle=False)
        val, test = train_test_split(temp, test_size=0.5, shuffle=False)

        train_data = pd.concat([train_data, train])
        val_data = pd.concat([val_data, val])
        test_data = pd.concat([test_data, test])

    train_data.to_csv(os.path.join(output_path,'{}_train.csv'.format(city)), index=False)
    val_data.to_csv(os.path.join(output_path,'{}_val.csv'.format(city)), index=False)
    test_data.to_csv(os.path.join(output_path,'{}_test.csv'.format(city)), index=False)

def dow2int(str_day):
    tmp = {'Monday':0, 'Tuesday':1,'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5,'Sunday':6}
    return tmp[str_day]


class UserPredict:

    def __init__(self, user_id, historical_steps):
        self.user_id = user_id
        self.steps = []
        self.diff = []

        self.historical_steps = historical_steps

    def expand(self, location_id, week_day, start_min):
        self.steps.append((location_id, week_day, start_min))

    def to_list(self) -> List:
        if len(self.steps) > self.historical_steps:
            dicts = []
            # Slipping window
            for start in range(0, len(self.steps) - self.historical_steps):
                window = self.steps[start: start + self.historical_steps]
                y = self.steps[start + self.historical_steps]

                location_ids, week_days, start_mins = zip(*window)
                location_y, weekday_y, start_min_y = y

                dicts.append({
                    'X': np.array(location_ids),
                    'user_X': self.user_id,
                    'weekday_X': np.array([dow2int(weekday) for weekday in week_days]),
                    'start_min_X': np.array(start_mins),
                    'diff': np.array(self.diff, dtype=np.float64),
                    'Y': location_y,
                    'weekday_Y': weekday_y,
                    'start_min_Y': start_min_y,
                })

            return dicts
        else:
            return []


def get_test_dict(test, historical_step):
    user_id_to_predict = {}
    for _, row in test.iterrows():
        user_id = row.user_id
        location_id = row.location_id
        start_min = row.start_min
        week_day = row.week_day
        if user_id not in user_id_to_predict:
            predict = UserPredict(user_id, historical_step)
            user_id_to_predict[user_id] = predict
        predict = user_id_to_predict[user_id]
        predict.expand(location_id, week_day, start_min)

    user_id_to_predict_list = []
    for predict in user_id_to_predict.values():
        user_id_to_predict_list.extend(predict.to_list())

    return user_id_to_predict_list


def filter_checkints_from_original_file(input_path=None, output_path=None, bounds_dict=None):
    checkins_file = os.path.join(input_path, "dataset_TIST2015_Checkins.txt")
    pois_file = os.path.join(input_path, "dataset_TIST2015_POIs.txt")

    for city_name in bounds_dict:
        bounds = bounds_dict[city_name]

        pois_df = pd.read_csv(pois_file, sep='\t', header=None, names=[
            "Venue ID", "Latitude", "Longitude", "Venue Category Name", "Country Code"
        ])
        filtered_pois = pois_df[
            (pois_df["Latitude"] <= bounds[0]) & (pois_df["Latitude"] >= bounds[1]) &
            (pois_df["Longitude"] >= bounds[2]) & (pois_df["Longitude"] <= bounds[3])
        ]
        filtered_venue_ids_set = set(filtered_pois["Venue ID"])
        checkins_df = pd.read_csv(checkins_file, sep='\t', header=None, names=[
            "User ID", "Venue ID", "UTC Time", "Timezone Offset"
        ])

        filtered_checkins = checkins_df[checkins_df["Venue ID"].apply(lambda x: x in filtered_venue_ids_set)]

        print("Filtered POIs:")
        print(filtered_pois)

        print("\nFiltered Check-ins:")
        print(filtered_checkins)

        filtered_pois.to_csv(os.path.join(output_path, "{}_filtered_pois.csv".format(city_name)), index=False)
        filtered_checkins.to_csv(os.path.join(output_path, "{}_filtered_checkins.csv".format(city_name)), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./mobility")
    parser.add_argument('--filter_path', type=str, default="./mobility/checkin")
    parser.add_argument("--output_path", type=str, default="./mobility/checkin_split")
    parser.add_argument("--final_output", type=str, default="./mobility/checkin_test_pk")
    parser.add_argument("--repeat_THR", type=float, default=0.05)
    parser.add_argument("--context_step", type=int, default=40)
    parser.add_argument("--historical_step", type=int, default=5)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    historical_step = args.historical_step
    context_step = args.context_step
    step = args.step
    repeat_THR = args.repeat_THR
    input_path = args.input_path
    filter_path = args.filter_path
    output_path = args.output_path
    final_output = args.final_output

    bounds_dict = {
        "Shanghai": [31.4193, 30.7300, 121.1215, 121.9730],
        "Beijing": [40.0891, 39.7442, 116.1536, 116.6082],
        "Mumbai": [19.2729, 18.9797, 72.7576, 72.9836],
        "Tokyo": [35.8712, 35.5859, 139.6005, 139.9713],
        "London": [51.6146, 51.3598, -0.3159, 0.1675],
        "Paris": [48.9038, 48.8115, 2.249, 2.4239],
        "NewYork": [40.7314, 40.7028, -74.0128, -73.9445], 
        "Moscow": [55.8792, 55.6319, 37.4016, 37.8067],
        "San_Francisco": [37.8076, 37.6153, -122.5099, -122.3630],
        "Sao_Paulo": [-23.4242, -23.7765, -46.8251, -46.4365],
        "Nairobi": [-1.1906, -1.3381, 36.6868, 36.9456],
        "Cape_Town": [-33.8179, -34.0674, 18.3472, 18.6974],
        "Sydney": [-33.6450, -34.0447, 150.8382, 151.2982]
    }

    filter_checkints_from_original_file(
        input_path=input_path,
        output_path=filter_path,
        bounds_dict=bounds_dict
        )

    datas = os.listdir(filter_path)
    poi_checkin = []
    city_type = {}
    for data in datas:
        city  = data.split("_")[0]
        type_ = data.split("_")[-1]
        if city not in city_type:
            city_type[city] = {'pois.csv': "", "checkins.csv": ""}
        city_type[city][type_] = data
    for type_dict in city_type.values():
        poi_checkin.append((type_dict['pois.csv'],type_dict['checkins.csv']))
        
    for poi, checkin in poi_checkin:
        city = poi.split("_")[0]
        city_poi = pd.read_csv(os.path.join(filter_path, poi))
        city_checkin = pd.read_csv(os.path.join(filter_path, checkin))
        result = match_pos(city_poi, city_checkin)
        result['start_min'] = result.apply(lambda x: get_time(x.utc), axis=1)
        result['start_day'] = result.apply(lambda x: get_start_day(x.utc), axis=1)
        result['week_day'] = result.apply(lambda x: get_day(x.utc), axis=1)
        result['location_id'], unique_venues = pd.factorize(result['location'])
        get_split(result, city, output_path, repeat_THR, context_step, historical_step)
        print('{} Done!'.format(city))

    output_data = os.listdir(output_path)
    for data in output_data:
        city = data.split("_")[0]
        data_type = data.split("_")[-1]
        if data_type == "test.csv":
            tar_data = pd.read_csv(os.path.join(output_path, data))
            result = get_test_dict(tar_data, historical_step)
            output_name = os.path.join(final_output, '{}_fin'.format(city))
            with open('{}.pk'.format(output_name), 'wb') as f:
                pickle.dump(result, f)
