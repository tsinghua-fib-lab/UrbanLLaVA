import os
import tqdm
import argparse
import pandas as pd
from multiprocessing import Pool

from config import CITY_BOUNDARY, VLM_MODELS, LLM_MODELS, TASK_DEST_MAPPING


class Evaluator:
    def __init__(self, city_name, model_name, data_name, task_name, workers=1) -> None:
        self.city_list = list(CITY_BOUNDARY.keys())
        self.model_list = {"vlm": VLM_MODELS, "llm": LLM_MODELS}
        self.task_list = list(TASK_DEST_MAPPING.keys())
        self.workers = workers
        
        self.city_name_list = city_name.split(",")
        self.model_name_list = model_name.split(",")
        self.task_name_list = task_name.split(",")
        self.data_name = data_name

    def evaluate(self):
        # TODO: run single task or run task sets
        self.multiple_task_wrapper(self.task_name_list, self.model_name_list, self.city_name_list)

    def valid_inputs(self):
        # TODO: check if the inputs are valid
        pass

    @staticmethod
    def single_task_wrapper(task_name, model_name, city_name, data_name):
        # run single task 
        task_desc = TASK_DEST_MAPPING[task_name]
        if task_name in ["population", "objects"]:
            eval_scipt = "python -m {} --city_name={} --data_name={} --model_name={} --task_name={}".format(task_desc, city_name, data_name, model_name, task_name)
        else:
            eval_scipt = "python -m {} --city_name={} --data_name={} --model_name={}".format(task_desc, city_name, data_name, model_name)

        return os.system(eval_scipt)

    # TODO: run multiple tasks
    def multiple_task_wrapper(self, task_list, model_list, city_list):
        # TODO running multi tasks efficiently
        para_group = []
        for task in task_list:
            for model in model_list:
                for city in city_list:
                    para_group.append([task, model, city, self.data_name])
        
        if self.workers==1:
            for para in para_group:
                self.single_task_wrapper(*para)
        else:
            with Pool(args.workers) as pool:
                pool.starmap(self.single_task_wrapper, para_group)


    def analyze_results(self):
        # TODO: analyze the results
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default="Beijing")
    parser.add_argument('--task_name', type=str, default='traffic_signal')
    parser.add_argument('--data_name', type=str, default='mini')
    parser.add_argument('--model_name', type=str, default="GPT4o")
    args = parser.parse_args()

    # Evaluator Initialization
    Eval = Evaluator(
        city_name=args.city_name,
        model_name=args.model_name,
        data_name=args.data_name,
        task_name=args.task_name)
    # Running Evalautor 
    Eval.evaluate()
