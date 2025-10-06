import os
import tqdm
import copy
import jsonlines
import argparse
import pandas as pd
from multiprocessing import Pool


from config import RESULTS_PATH, GEOQA_SAMPLE_RATIO, GEOQA_DATA_PATH, LLM_MODEL_MAPPING
from serving.llm_api import extract_choice, get_chat_completion, get_model_response_hf, get_model_response_vllm, get_response_mllm_api
from serving.vlm_serving import VLMWrapper

TASK_FILES_EXTEND = {
    "aoi2addr":"aoi2addr.csv",
    "aoi2type":"aoi2type.csv",
    "aoi_boundary_poi":"aoi_boundary_poi.csv",
    "aoi_group":"aoi_group.csv",
    "AOI_POI":"AOI_POI.csv",
    "aoi_poi":"aoi_poi.csv",
    "AOI_POI2":"AOI_POI2.csv",
    "AOI_POI3":"AOI_POI3.csv",
    "AOI_POI4":"AOI_POI4.csv",
    "AOI_POI5":"AOI_POI5.csv",
    "AOI_POI6":"AOI_POI6.csv",
    "AOI_POI_road1":"AOI_POI_road1.csv",
    "AOI_POI_road2":"AOI_POI_road2.csv",
    "AOI_POI_road3":"AOI_POI_road3.csv",
    "AOI_POI_road4":"AOI_POI_road4.csv",
    "boundary_road":"boundary_road.csv",
    "districts_poi_type":"districts_poi_type.csv",
    "landmark_env":"landmark_env.csv",
    "landmark_path":"landmark_path.csv",
    "poi2addr":"poi2addr.csv",
    "poi2coor":"poi2coor.csv",
    "poi2type":"poi2type.csv",
    "poi_aoi":"poi_aoi.csv",
    "road_arrived_pois":"road_arrived_pois.csv",
    "road_length":"road_length.csv",
    "road_link":"road_link.csv",
    "road_od":"road_od.csv",
    "type2aoi":"type2aoi.csv",
    "type2poi":"type2poi.csv",
}

def task_files_adaption(task_file, region_exp, evaluate_version):
    task_files = copy.deepcopy(task_file)
    path_prefix = os.path.join(GEOQA_DATA_PATH, "{}/{}".format(region_exp, evaluate_version))
    for k in task_files:
        if path_prefix not in task_files[k]:
            task_files[k] = os.path.join(path_prefix, task_files[k])
    os.makedirs(path_prefix, exist_ok=True)
    return task_files

INIT_PROMPT = "The following is a multiple-choice question about the geospatial knowledge of city. Please choose the most suitable one among A, B, C and D as the answer to this question. Please output the option directly. No need for explaination.\n"

def format_example(line, include_answer=True, max_choices=4):
    choices = ["A", "B", "C", "D"]
    prompt=INIT_PROMPT
    if max_choices>=5:
        choices.append("E")
        prompt = prompt.replace("A, B, C and D", "A, B, C, D and E")
    if max_choices>=6:
        choices.append("F")
        prompt = prompt.replace("A, B, C, D and E", "A, B, C, D, E and F")
    
    example = prompt + 'Question: ' + line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += '\nAnswer: ' + line["answer"] + '\n\n'
    else:
        example += '\nAnswer:'
    return example


###################### Evaluation interface
def run_evaluate_api(task_file_path, model_name, task_name, max_validation, temperature, max_tokens, infer_server, region_exp, data_version, model=None):
    try:
        test_df = pd.read_csv(task_file_path, header=0)
    except:
        return []
    
    columns = test_df.columns.to_list()
    if "F" in columns:
        max_choices = 6
    elif "E" in columns:
        max_choices = 5
    elif "D" in columns:
        max_choices = 4
    else:
        max_choices = 4
    
    if data_version == "mini":
        test_df = test_df.sample(min(max_validation,int(test_df.shape[0]*GEOQA_SAMPLE_RATIO)), random_state=42)
    else:
        if test_df.shape[0]>max_validation*2:
            test_df = test_df.sample(max_validation, random_state=42)
    correct_count, count = 0, 0
    res = []
    for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, include_answer=False, max_choices=max_choices)
        if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
            output = get_response_mllm_api(session=[{"role":"user", "content": question}], 
                    model_name=model_name, temperature=temperature, max_tokens=max_tokens, infer_server=infer_server)
        else:
            if model is not None:
                output = get_model_response_hf(question, model)
            else:
                output, _ = get_chat_completion(
                    session=[{"role":"user", "content": question}], 
                    model_name=model_name, temperature=temperature, max_tokens=max_tokens, infer_server=infer_server
                    )
        res.append([{"role":"user", "content": question}, {"role":"assistant", "content": output}, {"role":"ref", "content": row["answer"]}])

        if len(output) == 0:
            pass
        else:
            ans = extract_choice(output, ["A", "B", "C", "D", "E", "F"]) 
            if ans==row["answer"]:
                correct_count += 1
        count += 1

    if count == 0:
        count = 1
    print("Success rate:{}({}/{})".format(correct_count/count, correct_count, count))

    os.makedirs(os.path.join(RESULTS_PATH, "logs_geo_knowledge/"), exist_ok=True)
    if "/" in model_name:
        model_name = model_name.replace("/", "_")
    with jsonlines.open(os.path.join(RESULTS_PATH, "logs_geo_knowledge", "{}_{}_{}.jsonl".format(region_exp, model_name, task_name)), "w") as wid:
        for r in res:
            wid.write(r)
    return [model_name, task_name, correct_count, count, correct_count/count]

if __name__ == "__main__":
    print("start model evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GPT4omini")
    parser.add_argument("--city_eval_version", type=str, default="v82")
    parser.add_argument("--city_name", type=str, default="Beijing")
    parser.add_argument("--max_tokens", default=200, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_valid", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--infer_server", type=str, default=None)
    parser.add_argument("--data_name", choices=["all","mini"], default="mini")
    args = parser.parse_args()

    KNOWLEDGE_TASK_FILES_ = task_files_adaption(TASK_FILES_EXTEND, args.city_name, args.city_eval_version)
    
    para_group = []
    for model_name in [args.model_name]: 
        model = None
        if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
            model_full = model_name
        else:
            try:
                model_full = LLM_MODEL_MAPPING[model_name]
            except KeyError:
                model_full = model_name
                model_wrapper = VLMWrapper(model_full)
                model = model_wrapper.get_vlm_model()

        for task_name in KNOWLEDGE_TASK_FILES_.keys():
            print("evaluate model:{} task:{}".format(model_name, task_name))
            if "csv" not in KNOWLEDGE_TASK_FILES_[task_name]:
                print("task:{} is not ready, ignore it!".format(task_name))
                continue
            para_group.append((
                KNOWLEDGE_TASK_FILES_[task_name], 
                model_name,  
                task_name,
                args.max_valid, 
                args.temperature, 
                args.max_tokens,
                args.infer_server,
                args.city_name,
                args.data_name,
                model
            ))

    res_df = []
    if args.workers == 1:
        for para in para_group:
            print(para)
            res = run_evaluate_api(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7], para[8], para[9])
            if len(res)<1:
                continue
            res_df.append(res)
    else:
        with Pool(args.workers) as pool:
            results = pool.starmap(run_evaluate_api, para_group)
        for res in results:
            if len(res)<1:
                continue
            res_df.append(res)
    
    res_df = pd.DataFrame(res_df, columns=["model_name", "task_name", "corrct", "count", "accuracy"])
    print(res_df.head())
    os.makedirs(os.path.join(RESULTS_PATH, "geo_knowledge_result"), exist_ok=True)
    res_df.to_csv(os.path.join(RESULTS_PATH, "geo_knowledge_result", "geo_knowledge_{}_{}_summary_{}.csv".format(
        args.city_name, 
        args.city_eval_version, 
        args.model_name.replace("/", "_"))))
