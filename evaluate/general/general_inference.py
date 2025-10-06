# This script is used to convert a address QA into a multi-choice question for evaluation.

import os
import argparse

from tqdm import tqdm, trange
import json

from config import RESULTS_PATH, EVAL_COMMON_PATH
from serving.vlm_serving import VLMWrapper

datasets_path_mapping = {
    "llava-bench": "llava-bench-in-the-wild",
    "MMStar": "MMStar",
    "realworldqa": "realworldqa",
    "mm-vet": "MM-vet/mm-vet",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--task_name', type=list, default=['llava-bench', 'MMStar', 'realworldqa', 'mm-vet'], help='dataset name list')
    parser.add_argument('--data_name', type=str, default='all', help='data name', choices=["all", "mini"])
    args = parser.parse_args() 

    model_name = args.model_name
    task_name_list = args.task_name

    print("Load the model")
    model_wrapper = VLMWrapper(args.model_name)
    model = model_wrapper.get_vlm_model()
    
    for dataset_name in task_name_list:
        dir_name = datasets_path_mapping[dataset_name]
        assert os.path.exists(os.path.join(EVAL_COMMON_PATH, dir_name)), f"{dir_name} does not exist"


    if "llava-bench" in task_name_list:

        # Do the inference on llava-bench
        dir1 = os.path.join(EVAL_COMMON_PATH, datasets_path_mapping["llava-bench"])
        image_dir1 = os.path.join(dir1, "images")
        questions_file = os.path.join(dir1, "questions.jsonl")
        with open(questions_file, 'r') as f:
            questions = [json.loads(line) for line in f.readlines()]

        reference_file = os.path.join(dir1, "answers_gpt4.jsonl")
        with open(reference_file, 'r') as f:
            references = [json.loads(line) for line in f.readlines()]

        assert len(questions) == len(references), "The number of questions and references should be the same"

        response_list = []
        print("Start inference on llava-bench")
        for i in trange(len(questions)):
            question_item = questions[i]
            reference_item = references[i]
            assert question_item["question_id"] == reference_item["question_id"], "The question_id should be the same"
            gt = reference_item["text"]
            prompt = question_item["text"]
            assert question_item["text"] == reference_item["prompt"], f"The prompt should be the same: First: {question_item['text']} , Second: {reference_item['text']}"
            image_path = os.path.join(image_dir1, question_item["image"])
            assert os.path.exists(image_path), f"{image_path} does not exist"
            response = model.generate([image_path, prompt])
            response_item = {
                "image": question_item["image"],
                "prompt": prompt,
                "response": response,
                "gt": gt
            }
            response_list.append(response_item)

        print("Inference on llava-bench is done")
        output_path1 = os.path.join(dir1, f"{model_name}_response.jsonl")
        with open(output_path1, 'w') as f:
            for item in response_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


    if "MMStar" in task_name_list:

        # Do the inference on MMStar
        dir2 = os.path.join(EVAL_COMMON_PATH, datasets_path_mapping["MMStar"])
        # MMStar is using parquet and tsv file, but we can convert them into basic files

        benchmark_file = os.path.join(dir2, "val.json")
        with open(benchmark_file, 'r') as f:
            mmstar_data = json.load(f)

        response_list = []
        print("Start inference on MMStar")
        for item in tqdm(mmstar_data):
            index = item["index"]
            prompt = item["question"]
            gt = item["answer"]
            image_path = os.path.join(dir2, item["image"])
            assert os.path.exists(image_path), f"{image_path} does not exist"
            response = model.generate([image_path, prompt])
            response_item = {
                "image": item["image"],
                "prompt": prompt,
                "response": response,
                "gt": gt
            }
            response_list.append(response_item)
        
        assert len(response_list) == len(mmstar_data), "The number of responses should be the same as the number of data"
        print("Inference on MMStar is done")
        output_path2 = os.path.join(dir2, f"{model_name}_response.jsonl")
        with open(output_path2, 'w') as f:
            for item in response_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if "realworldqa" in task_name_list:

        # Do the inference on realworldqa
        dir3 = os.path.join(EVAL_COMMON_PATH, datasets_path_mapping["realworldqa"])
        image_dir3 = os.path.join(dir3, "images")
        questions_file = os.path.join(dir3, "annotations.json")
        with open(questions_file, 'r') as f:
            questions = json.load(f)

        response_list = []
        print("Start inference on realworldqa")
        for item in tqdm(questions):
            prompt = item["question"]
            gt = item["answer"]
            image_path = os.path.join(image_dir3, item["image"])
            assert os.path.exists(image_path), f"{image_path} does not exist"
            response = model.generate([image_path, prompt])
            response_item = {
                "image": item["image"],
                "prompt": prompt,
                "response": response,
                "gt": gt
            }
            response_list.append(response_item)

        assert len(response_list) == len(questions), "The number of responses should be the same as the number of data"
        print("Inference on realworldqa is done")
        output_path3 = os.path.join(dir3, f"{model_name}_response.jsonl")
        with open(output_path3, 'w') as f:
            for item in response_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if "mm-vet" in task_name_list:

        # Do the inference on mm-vet
        dir4 = os.path.join(EVAL_COMMON_PATH, datasets_path_mapping["mm-vet"])
        image_dir4 = os.path.join(dir4, "images")
        questions_file = os.path.join(dir4, "mm-vet.json")
        with open(questions_file, 'r') as f:
            questions = json.load(f)

        response_list = []
        print("Start inference on mm-vet")
        for key, item in tqdm(questions.items()):
            prompt = item["question"]
            gt = item["answer"]
            image_path = os.path.join(image_dir4, item["imagename"])
            assert os.path.exists(image_path), f"{image_path} does not exist"
            response = model.generate([image_path, prompt])
            response_item = {
                "image": item["image"],
                "prompt": prompt,
                "response": response,
                "gt": gt
            }
            response_list.append(response_item)

        assert len(response_list) == len(questions), "The number of responses should be the same as the number of data"
        print("Inference on mm-vet is done")
        output_path4 = os.path.join(dir4, f"{model_name}_response.jsonl")
        with open(output_path4, 'w') as f:
            for item in response_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


    


    model_wrapper.clean_proxy()
