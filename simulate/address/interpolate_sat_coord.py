# Function: Interpolate the coordinates of satellite images for zoom level 15 and 17
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['London', 'NewYork', 'Beijing'])
    parser.add_argument('--work_dir', type=str, default='../../data/')
    args = parser.parse_args()
    city = args.city
    work_dir = args.work_dir

    for zl in ['zl15','zl17']:


        input_csv_path = work_dir + f'dev-{city}/SAT_{city}_{zl}.csv'

        output_csv_path = work_dir + f'dev-{city}/SAT_interpolate_{city}_{zl}.csv'


        if 'zl17' in input_csv_path:
            data = pd.read_csv(input_csv_path)

            img_list = []
            lng_list = []
            lat_list = []

            for i in  range(len(data)):
                tl_lat = data.at[i,'tl_lat']
                tl_lng = data.at[i,'tl_lng']
                bt_lat = data.at[i,'bt_lat']
                bt_lng = data.at[i,'bt_lng']
                # evenly sample 3*3 points for zoom level 17
                for j in range(3):
                    for k in range(3):
                        if j == 0 and k==0:
                            img_list.append(data.at[i,'img_name'])
                            lng_list.append(data.at[i,'tl_lng'])
                            lat_list.append(data.at[i,'tl_lat'])
                            continue
                        if j ==2 and k==2:
                            img_list.append(data.at[i,'img_name'])
                            lng_list.append(data.at[i,'bt_lng'])
                            lat_list.append(data.at[i,'bt_lat'])
                            continue
                        img_list.append(data.at[i,'img_name'])
                        lng_list.append(data.at[i,'tl_lng']+j/2*(data.at[i,'bt_lng']-data.at[i,'tl_lng']))
                        lat_list.append(data.at[i,'tl_lat']-k/2*(data.at[i,'tl_lat']-data.at[i,'bt_lat']))

            new_df = pd.DataFrame({'img_name':img_list,'lng':lng_list,'lat':lat_list})
            new_df.to_csv(output_csv_path, index=False)

        elif 'zl15' in input_csv_path:    


            data = pd.read_csv(input_csv_path)

            img_list = []
            lng_list = []
            lat_list = []

            for i in  range(len(data)):
                tl_lat = data.at[i,'tl_lat']
                tl_lng = data.at[i,'tl_lng']
                bt_lat = data.at[i,'bt_lat']
                bt_lng = data.at[i,'bt_lng']
        # evenly sample 5*5 points for zoom level 15
                for j in range(5):
                    for k in range(5):
                        if j == 0 and k==0:
                            img_list.append(data.at[i,'img_name'])
                            lng_list.append(data.at[i,'tl_lng'])
                            lat_list.append(data.at[i,'tl_lat'])
                            continue
                        if j ==4 and k==4:
                            img_list.append(data.at[i,'img_name'])
                            lng_list.append(data.at[i,'bt_lng'])
                            lat_list.append(data.at[i,'bt_lat'])
                            continue
                        img_list.append(data.at[i,'img_name'])
                        lng_list.append(data.at[i,'tl_lng']+j/4*(data.at[i,'bt_lng']-data.at[i,'tl_lng']))
                        lat_list.append(data.at[i,'tl_lat']-k/4*(data.at[i,'tl_lat']-data.at[i,'bt_lat']))

            new_df = pd.DataFrame({'img_name':img_list,'lng':lng_list,'lat':lat_list})
            new_df.to_csv(output_csv_path, index=False)


        else:
            print("Please input the correct csv file path!")
            raise NotImplementedError


        # sanity check
        print("Before interpolation, the number of images is ",len(data))
        print("After interpolation, the number of images is ",len(new_df))
        print("The ratio of the number of images after interpolation to the number of images before interpolation is ",len(new_df)/len(data))