import os
import argparse
import pandas as pd
import random
import jsonlines
import json
from datetime import datetime
from multiprocessing import Pool
from openai import OpenAI
from tqdm import tqdm

from pycitydata.map import Map

from serving.vlm_serving import VLMWrapper
from config import MAP_CACHE_PATH, RESOURCE_PATH, RESULTS_PATH, MONGODB_URI, MAP_DICT, PROXY, NAVIGATION_IMAGE_FOLDER, NAVIGATION_URL_PATH, NAVIGATION_FILE_PATH, NAVIGATION_URL_PATH_now
from .utils import calculate_direction, calculate_distance, get_prompt_eval, get_basic_prompt
from serving.llm_api import  get_model_response_gpt, get_model_response_hf_image, match_response, get_response_mllm_api

train_data_gen=False
conversation_counter = 0
def route_process_nav(city_map, city, road_ids, match_df, url_df, meta_info_df):
    # one route landmark navigation
    url_map = dict(zip(url_df['image_name'], url_df['image_url']))
    instructions = []
    basic_prompt = get_basic_prompt()
    instructions.append({
        "type": "text",
        "text": basic_prompt
    })
    steps = []

    for i, road_id in enumerate(road_ids):
        road_matches = match_df[match_df['road_id'] == road_id]
        if road_matches.empty:
            print(f"No matches found for road_id: {road_id}")
            continue
        road_info = city_map.get_road(road_id)
        road_name = road_info['name']
        if not road_name: 
            road_name = "unknown road"
        road_len = road_info['length']
        lane_id = road_info['lane_ids'][0]

        sorted_matches = road_matches.sort_values(by='distance')
        for count, (idx, row) in enumerate(sorted_matches.iterrows()):
            walk_len = int(road_len - row['distance'])
            image_path_suffix = row['file_name']
            url_image = url_map.get(image_path_suffix)
            
            if count == 0:
                action = "forward"
                step_instruction = f"When you see this image, your current action is 'forward':"
                instructions.append({
                        "type": "text",
                        "text": step_instruction
                    })
                instructions.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url_image
                    }
                })
                steps.append({
                            "action": action,
                            "image_url": url_image
                        })
                
            # if last element
            elif count == len(sorted_matches) - 1:
                # print("enter last image")
                # check if it is the last road_id
                if i == len(road_ids) - 1:
                    action = "stop"
                    step_instruction = f"When you see this image, your current action is '{action}':"
                    instructions.append({
                        "type": "text",
                        "text": step_instruction
                    })
                    instructions.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url_image
                        }
                    })
                    steps.append({
                                "action": action,
                                "image_url": url_image
                            })
                else:
                    next_road_id = road_ids[i+1]
                    next_road_info = city_map.get_road(next_road_id)
                    next_road_start = next_road_info['shapely_lnglat'].coords[0]
                    current_road_end = road_info['shapely_lnglat'].coords[-1]
                    action = calculate_direction(current_road_end, next_road_start)
                    step_instruction = f"When you see this image, your current action is '{action}':"
                    instructions.append({
                        "type": "text",
                        "text": step_instruction
                    })
                    instructions.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url_image
                        }
                    })
                    steps.append({
                                "action": action,
                                "image_url": url_image
                            })
            
    last_instruction = f"ATTENTION: Your should describe the image and integrate the action decision into your description for EACH image."
    instructions.append({
        "type": "text",
        "text": last_instruction
    })
    return instructions, steps


def nav_gen(city, route_file, output_file, city_map, model_name, data_sample=600):
    match_file = os.path.join(NAVIGATION_FILE_PATH, '{}_matched_images.csv'.format(city))
    meta_file = os.path.join(NAVIGATION_IMAGE_FOLDER, f"{city}_StreetView_Images/combined_stitch_meta_info.csv")

    url_file = NAVIGATION_URL_PATH
    match_data_df = pd.read_csv(match_file)
    meta_info_df = pd.read_csv(meta_file)
    url_df = pd.read_csv(url_file)  

    sample = 0
    with jsonlines.open(output_file, mode='a') as writer:
        with jsonlines.open(route_file) as reader:
            for obj in reader:
                start_aoi_id = obj.get('start_aoi_id')
                dest_aoi_id = obj.get('dest_aoi_id')
                road_ids = obj.get('road_ids')
                instructions, steps = route_process_nav(city_map, city, road_ids, match_data_df, url_df, meta_info_df)
                response = get_model_response_gpt(json.dumps(instructions), model_name)

                record = {
                        "route": road_ids,
                        "response": response,
                        "steps": steps
                    }
                writer.write(record)
                sample += 1
                
                if sample == data_sample:
                    break

def single_route_process(city, route, navigation, steps, model_name):
    # print("enter single_route_process")
    success_flag = True
    basic_prompt = get_prompt_eval()
    basic_prompt = f"{basic_prompt}\n{navigation}"
    prompts = []
    prompts.append({
    "type": "text",
    "text": basic_prompt
    })
    for step in steps:
        text = f"Here is the street view image of your current location."
        image_url = step['image_url']
        prompts.append({
            "type": "text",
            "text": text
        })
        prompts.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
        last_text = f"Please provide the next action('forward', 'left', 'right', or 'stop') based on the image and the navigation instruction:"
        prompts.append({
            "type": "text",
            "text": last_text
        })
        action_true = step['action']
        action = get_model_response_gpt(prompts)
        print(f"Action: {action}, True Action: {action_true}")
        if action != action_true:
            success_flag = False
            print("false")
            break
        prompts.append({
            "type": "text",
            "text": f"Action: {action}"
        })
    
    if success_flag:
        print("right")
    return success_flag
    

def process_single_route(args):
    # try 5 times
    city, route, response, steps, model_name = args

    for i in range(5):
        success_time = single_route_process(city, route, response, steps, model_name)
        if success_time == 1:
            print(f"Success found for route {route}")
            return {"route": route, "response": response, "steps": steps}  

    return None  


def validate_eval(city, instruction_file, instruction_validate_file, model_name, num_processes):
    results_to_save = []
    print(f"Validating routes in {instruction_file} using model {model_name}...")
    with jsonlines.open(instruction_file) as reader:
        records = list(reader)  

    print(f"Validating {len(records)} routes...")
    args_list = [(city, record['route'], record['response'], record['steps'], model_name) for record in records]
    with Pool(processes=num_processes) as pool:  
        print("Processing routes...enter parallel processing")
        for result in pool.imap(process_single_route, args_list):
            if result:
                results_to_save.append(json.dumps(result))
    with open(instruction_validate_file, 'a') as file:
        for item in results_to_save:
            file.write(item + '\n')

    print(f"Validated routes saved to {instruction_validate_file}")


def single_route_process_eval(city, route, navigation, steps, model_name, model=None):
    # print("enter single_route_process_eval")
    url_file = NAVIGATION_URL_PATH
    url_to_name_mapping = pd.read_csv(url_file).set_index('image_url')['image_name'].to_dict()
    url_file_now = NAVIGATION_URL_PATH_now
    name_to_url_mapping = pd.read_csv(url_file_now).set_index('image_name')['image_url'].to_dict()
    success_flag = True
    step_count = 0  
    basic_prompt = get_prompt_eval()
    basic_prompt = f"{basic_prompt}\n{navigation}"
    prompts = []
    if "gpt4o" in model_name:
        prompts.append({
        "type": "text",
        "text": basic_prompt
        })
    else:
        prompts.append({
        "type": "text",
        "value": basic_prompt
        })
    last_image_url = steps[-1]['image_url']
    actions_history = []  
    
    for step in steps:
        step_actions = []
        text = f"Here is the street view image of your current location."
        image_url = step['image_url']
        # current_prompts = prompts.copy()
        if actions_history:
            previous_actions = "Previous actions: " + ", ".join(actions_history)
        else:
            previous_actions = ""
        
        image_name = url_to_name_mapping[image_url]
        image_path = os.path.join(NAVIGATION_IMAGE_FOLDER, f"{city}_StreetView_Images/{image_name}")
            
        last_text = f"""
        Please provide the Reason and next Action('forward', 'left', 'right', or 'stop') based on the image and the navigation instruction.\n"""

        current_prompts = f"{basic_prompt}\n{previous_actions}\n{text}\n{last_text}"

        
        action_true = step['action']
        if "gpt4o" in model_name:
            gpt_prompts = []
            gpt_prompts.append({
                 "type": "text",
                 "text": current_prompts
            })
            gpt_prompts.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
            
            model_response = get_model_response_gpt(gpt_prompts, model_name)
            reason, action = match_response(model_response)
            # action = get_model_response_gpt(current_prompts, model_name)
            # action = match_response(action)
        elif "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
            image_url_now = name_to_url_mapping[image_name]
            gpt_prompts = []
            gpt_prompts.append({
                 "type": "text",
                 "text": current_prompts
            })
            gpt_prompts.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url_now
                }
            })
            model_response = get_response_mllm_api(session=[{"role": "user", "content": gpt_prompts}], model_name=model_name)
            reason, action = match_response(model_response)
        else:
            model_response = get_model_response_hf_image(image_path, current_prompts, model)
            # print(f"model response: {model_response}")
            reason, action = match_response(model_response)
            # action = get_model_response_hf(current_prompts, model)
            # action = match_response(action)
        
        step_actions.append({"response": model_response, "true_action": step['action']})

        # print(f"Action: {action}, True Action: {action_true}")
        if action != action_true:
            success_flag = False
            # print("false")
            distance = calculate_distance(city, last_image_url, image_url)
            break

        actions_history.append(action)
        step_count += 1
        

    if success_flag:
        distance = 0
        print("right")
    logs_file = os.path.join(RESULTS_PATH, f'outdoor_navigation_results/logs/{city}/{city}_{model_name}.jsonl')
    os.makedirs(os.path.dirname(logs_file), exist_ok=True)  
    with jsonlines.open(logs_file, mode='a') as writer:
        writer.write({"interactions": current_prompts, "step_actions": step_actions})
    return success_flag, step_count, distance

def process_single_route_eval(args):
    """"""
    city, route, response, steps, model_name = args
    success_found, step_count, distance = single_route_process_eval(city, route, response, steps, model_name)
    return success_found, step_count, distance

def eval_gen(city, instruction_validate_file, model_name, num_processes, samples):
    success_time = 0
    step_sum = 0
    distance_sum = 0

    if "gpt4o" in model_name:
        with jsonlines.open(instruction_validate_file) as reader:
            records = list(reader)  
            record_count = len(records)

            if samples < record_count:
                records = random.sample(records, samples)
            else:
                print(f"Requested sample size {samples} exceeds available records {record_count}. Evaluating all records.")

        args_list = [(city, record['route'], record['response'], record['steps'], model_name) for record in records]

        with Pool(processes=num_processes) as pool:  
            for result in pool.imap(process_single_route_eval, args_list):
                success_found, step_count, distance = result
                if success_found:
                    success_time += 1
                step_sum += step_count
                distance_sum += distance
    else:
        if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
            model=None
        else:
            print("other model")
            model_wrapper = VLMWrapper(model_name)
            print(model_name)
            model = model_wrapper.get_vlm_model()
        with jsonlines.open(instruction_validate_file) as reader:
            records = list(reader)  
            record_count = len(records)  
            
            if samples < record_count:
                records = random.sample(records, samples)
            else:
                print(f"Requested sample size {samples} exceeds available records {record_count}. Evaluating all records.")

        for record in tqdm(records):
            route = record["route"]
            response = record["response"]
            steps = record["steps"]
            success_found, step_count, distance = single_route_process_eval(city, route, response, steps, model_name, model)
            if success_found:
                success_time += 1
            step_sum += step_count
            distance_sum += distance

    success_ratio = success_time / record_count
    avg_step = step_sum / record_count
    avg_distance = distance_sum / record_count

    result_data = {
        "city": city,
        "model_name": model_name,
        "success_ratio": success_ratio,
        "average_steps": avg_step,
        "average_distance": avg_distance
    }
    print(f"Success ratio: {success_ratio}, Average steps: {avg_step}, Average distance: {avg_distance}")
    
    today_date = datetime.now().strftime('%Y-%m-%d')
    results_file = os.path.join(RESULTS_PATH, f'outdoor_navigation_results/{city}/{today_date}.jsonl')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)  

    with jsonlines.open(results_file, mode='a') as writer:
        writer.write(result_data)

    print(f"Results saved to {results_file}")


def main(args):
    m = Map(
            mongo_uri=MONGODB_URI,
            mongo_db="llmsim",
            mongo_coll=MAP_DICT[args.city_name],
            cache_dir=MAP_CACHE_PATH,
        )

    route_file = os.path.join(NAVIGATION_FILE_PATH, '{}_navigation_tasks.jsonl'.format(args.city_name))
    instruction_file = os.path.join(NAVIGATION_FILE_PATH, '{}_navigation_instructions-v2.4.jsonl'.format(args.city_name))
    instruction_validate_file = os.path.join(NAVIGATION_FILE_PATH, '{}_navigation_instructions_validate-v2.5.jsonl'.format(args.city_name))

    if args.data_name == "all":
        default_samples = 50
    else:
        default_samples = 5
    SAMPLES = args.samples if args.samples is not None else default_samples

    num_processes = 10
    if args.mode=="gen":
        data_sample = 600
        # generate
        nav_gen(args.city_name, route_file, instruction_file, m, args.model_name, data_sample)
        # validate
        validate_eval(args.city_name, instruction_file, instruction_validate_file, args.model_name, num_processes)

    elif args.mode=="eval":
        # evaluate
        eval_gen(args.city_name, instruction_validate_file, args.model_name, num_processes, SAMPLES)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Shanghai")
    parser.add_argument("--model_name", type=str, default="GPT4omini")
    parser.add_argument("--data_name", type=str, default="mini", choices=["all", "mini"])
    parser.add_argument("--samples", type=int)
    parser.add_argument("--mode", type=str, default="eval")
    args = parser.parse_args()

    main(args)
