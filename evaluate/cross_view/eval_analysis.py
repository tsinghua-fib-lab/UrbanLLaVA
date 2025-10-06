
import jsonlines
import os
import argparse
import pandas as pd
# from setproctitle import setproctitle
from tqdm import tqdm
import json

from config import CROSS_VIEW_PATH, CROSS_VIEW_RESULTS_PATH


def calculate_acc(city_name_list, model_name_list,task_name,save_name):
    all_acc_list = []
    all_city_list = []
    all_model_name_list = []
    for model_name in model_name_list:
        for city_name in city_name_list:
            city_pred_list = []
            city_GT_list = []
            print(model_name, city_name)
            try:
                json_file_path = os.path.join(CROSS_VIEW_RESULTS_PATH, city_name+'_'+model_name+'_'+task_name+'_eval.jsonl')
                with jsonlines.open(json_file_path) as reader:
                    for obj in reader:
                        city_pred_list.append(obj['text'])
                        city_GT_list.append(obj['GT'])
            except FileNotFoundError as e:
                    print("File not found! City:{} Model:{}".format(city_name, model_name))
                    continue

            if len(city_pred_list) != len(city_GT_list):
                raise ValueError("different length")

            city_pred_list_lower = [item.lower() for item in city_pred_list]
            city_GT_list_lower = [item.lower() for item in city_GT_list]

            count = sum(item1 == item2 for item1, item2 in zip(city_pred_list_lower, city_GT_list_lower))
            total = len(city_GT_list_lower)

            acc = count / total * 100

            all_acc_list.append(acc)
            all_city_list.append(city_name)
            all_model_name_list.append(model_name)

    df = pd.DataFrame({'city': all_city_list, 'model': all_model_name_list, 'acc': all_acc_list})
    df.to_csv(os.path.join(CROSS_VIEW_RESULTS_PATH, save_name), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')#InternVL2-40B  GPT4o_MINI  Qwen2-VL-2B-Instruct
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name') #Beijing, London, NewYork
    parser.add_argument('--eval_all_city_all_model', type=str, default="no", help='If yes, automatically evaluate data from all cities on all models.')
    parser.add_argument('--task_name', type=str, default='IR', help='task name', choices=["IR", "CL","SC_Buildings","SC_POIs"])
    #task_name include: Image Retrieval, Camera Localization, Scene Comparision Buildings, Scene Comparison POIs(restaurant, education, shopping)

    args = parser.parse_args()

    if args.eval_all_city_all_model == 'yes':
        args.city_name="Beijing,London,NewYork"
        args.model_name="InternVL2-40B,GPT4o_MINI"

        city_name_list = args.city_name.split(",")
        model_name_list = args.model_name.split(",")
    
    else:
        city_name_list = [args.city_name]
        model_name_list = [args.model_name]
    
    save_name = 'summary_all_models_all_cities_{}.csv'.format(args.task_name)

    calculate_acc(city_name_list,model_name_list,args.task_name,save_name)
