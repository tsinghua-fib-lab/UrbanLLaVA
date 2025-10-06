import os
import argparse
import pandas as pd
# from setproctitle import setproctitle
import jsonlines

from tqdm import tqdm
import json

from config import CROSS_VIEW_PATH, CROSS_VIEW_RESULTS_PATH
from serving.vlm_serving import VLMWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')#InternVL2-40B  GPT4o_MINI  Qwen2-VL-2B-Instruct
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name') #Beijing, London, NewYork
    parser.add_argument('--data_name', type=str, default="mini", help='dataset size')
    parser.add_argument('--task_name', type=str, default='IR', help='task name', choices=["IR", "CL","SC_Buildings","SC_POIs"]) 
    #task_name include: Image Retrieval, Camera Localization, Scene Comparision Buildings, Scene Comparison POIs(restaurant, education, shopping)

    args = parser.parse_args()

    print("Load the model")
    model_wrapper = VLMWrapper(args.model_name)
    model = model_wrapper.get_vlm_model()

    print("Load the test data jsonl")

    with jsonlines.open(os.path.join(CROSS_VIEW_PATH, f"{args.city_name}_{args.task_name}_eval.jsonl")) as reader:
        eval_data = list(reader)

    if args.data_name == 'mini':
        eval_data = eval_data[:int(0.1*len(eval_data))]

    
    response_list = []

    ###Model inference
    for obj in eval_data:
        img_names = obj['image']
        prompt = obj['conversations'][0]['value'].replace("<image>", "")
        GT = obj['conversations'][1]['value']
        ret = model.generate(img_names+[prompt])
        response_list.append([img_names, ret, GT])
    
    os.makedirs(os.path.dirname(CROSS_VIEW_RESULTS_PATH), exist_ok=True)
    # # Save the response
    with open(os.path.join(CROSS_VIEW_RESULTS_PATH, f"{args.city_name}_{args.model_name}_{args.task_name}_eval.jsonl"), "w") as fout:
        
        for i in range(len(response_list)):
            value = {
                "img_name": response_list[i][0],
                "text": response_list[i][1],
                "GT": response_list[i][2],  ##saving GT for quick human evaluation
            }
            fout.write(json.dumps(value) + "\n")
    
    model_wrapper.clean_proxy()
