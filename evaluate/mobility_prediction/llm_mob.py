import os
import pickle
import time
import ast
import json
import logging
import re
import random
import argparse
import pandas as pd
from datetime import datetime

from serving.llm_api import get_chat_completion, get_model_response_hf, get_response_mllm_api
from .metrics import get_acc1_f1, cal_metrics
from config import RESULTS_PATH, MOBILITY_SAMPLE_RATIO, MOBILITY_SPLIT_PATAH, MOBILITY_TEST_PATH, LLM_MODEL_MAPPING
from serving.vlm_serving import VLMWrapper

random.seed(32)

def get_dataset(dataname, split_path="./checkin_split/", test_path="./checkint_test_pk/", data_version="all"):
    
    # Get training and validation set and merge them
    train_data = pd.read_csv(os.path.join(split_path, f"{dataname}_train.csv"))
    valid_data = pd.read_csv(os.path.join(split_path, f"{dataname}_val.csv"))

    # Get test data
    with open(os.path.join(test_path, f"{dataname}_fin.pk"), "rb") as f:
        test_file = pickle.load(f)  # test_file is a list of dict
    all_users = list(set(train_data["user_id"]))

    if data_version == "mini":
        sample_users = random.choices(all_users, k=int(len(all_users)*MOBILITY_SAMPLE_RATIO))
        train_data = train_data[train_data["user_id"].isin(sample_users)]
        valid_data = valid_data[valid_data["user_id"].isin(sample_users)]
        test_data = []
        for item in test_file:
            if item["user_X"] in sample_users:
                test_data.append(item)
        test_file = test_data

    # merge train and valid data
    tv_data = pd.concat([train_data, valid_data], ignore_index=True)
    tv_data.sort_values(['user_id', 'start_day', 'start_min'], inplace=True)
    if dataname == 'geolife':
        tv_data['duration'] = tv_data['duration'].astype(int)

    print("Number of total test sample: ", len(test_file))
    return tv_data, test_file


def convert_to_12_hour_clock(minutes):
    if minutes < 0 or minutes >= 1440:
        return "Invalid input. Minutes should be between 0 and 1439."

    hours = minutes // 60
    minutes %= 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    return f"{hours:02d}:{minutes:02d} {period}"



def get_logger(logger_name, log_dir='logs/'):
    # Create log dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its log level
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    log_file = 'log_file' + formatted_datetime + '.log'
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_user_data(train_data, uid, num_historical_stay, logger):
    user_train = train_data[train_data['user_id']==uid]
    logger.info(f"Length of user {uid} train data: {len(user_train)}")
    user_train = user_train.tail(num_historical_stay)
    logger.info(f"Number of user historical stays: {len(user_train)}")
    return user_train


# Organising data
def organise_data(data_version, split_path, dataname, user_train, test_file, uid, logger, num_context_stay=5, sample_single_user=10):
    # Use another way of organising data
    historical_data = []
    for _, row in user_train.iterrows():
        historical_data.append(
            (convert_to_12_hour_clock(int(row['start_min'])),
            row['week_day'],
            row['location_id'])
            )

    logger.info(f"historical_data: {historical_data}")
    logger.info(f"Number of historical_data: {len(historical_data)}")

    # Get user ith test data
    list_user_dict = []
    for i_dict in test_file:
        i_uid = i_dict['user_X']
        if i_uid == uid:
            list_user_dict.append(i_dict)
            
    if data_version == "mini":
        sample_single_user = min(len(list_user_dict),int(sample_single_user*MOBILITY_SAMPLE_RATIO))
        
    list_user_dict = random.choices(list_user_dict, k=sample_single_user)

    predict_X = []
    predict_y = []
    for i_dict in list_user_dict:
        construct_dict = {}
        context = list(zip([convert_to_12_hour_clock(int(item)) for item in i_dict['start_min_X'][-num_context_stay:]],
                        i_dict['weekday_X'][-num_context_stay:],
                        i_dict['X'][-num_context_stay:]))
        target = (convert_to_12_hour_clock(int(i_dict['start_min_Y'])), i_dict['weekday_Y'], None, "<next_place_id>")
        construct_dict['context_stay'] = context
        construct_dict['target_stay'] = target
        predict_y.append(i_dict['Y'])
        predict_X.append(construct_dict)

    #logger.info(f"predict_data: {predict_X}")
    logger.info(f"Number of predict_data: {len(predict_X)}")
    logger.info(f"predict_y: {predict_y}")
    logger.info(f"Number of predict_y: {len(predict_y)}")
    return historical_data, predict_X, predict_y


def single_query_top1_fsq(historical_data, X, model_name, model):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visit to a certain place during certain time.
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and weekday) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (place ID) and "reason" (a concise explanation that supports your prediction)

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    if "meta-llama" in model_name:
        json_mode = True
    else :
        json_mode = False
    #def get_chat_completion(client, prompt, model="gpt-3.5-turbo-0613", json_mode=False, max_tokens=1200):
    if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
        completion = get_response_mllm_api(session=[{"role": "user", "content": prompt}], model_name=model_name, temperature=0, max_tokens=1200, json_mode=json_mode)
        token_usage = 0
    else:
        if model is not None:
            completion = get_model_response_hf(prompt, model)
            token_usage = 0
        else:
            completion, token_usage = get_chat_completion(session=[{"role": "user", "content": prompt}], model_name=model_name, json_mode=json_mode, max_tokens=1200, temperature=0, infer_server="LLM2-vllm")
    return completion, token_usage


def single_query_top1_wot_fsq(historical_data, X, model_name, model):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.    
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user.

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.Do not include other words in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
        completion = get_response_mllm_api(session=[{"role": "user", "content": prompt}], model_name=model_name, temperature=0, max_tokens=1200, json_mode=False)
        token_usage = 0
    else:
        if model is not None:
            completion = get_model_response_hf(prompt, model)
            token_usage = 0
        else:
            completion, token_usage = get_chat_completion(prompt=prompt, model_name=model_name, json_mode=False, max_tokens=1200, temperature=0, infer_server="LLM2-vllm")
    return completion,token_usage


def single_query_top10_fsq(historical_data, X, model_name, model):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times.
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and weekday) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction)

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """

    if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
        completion = get_response_mllm_api(session=[{"role": "user", "content": prompt}], model_name=model_name, temperature=0, max_tokens=1200, json_mode=False)
        token_usage = 0
    else:
        if model is not None:
            completion = get_model_response_hf(prompt, model)
            token_usage = 0
        else:
            completion, token_usage = get_chat_completion(prompt=prompt, model_name=model_name, json_mode=False, max_tokens=1200, temperature=0, infer_server="LLM2-vllm")
    return completion, token_usage


def single_query_top10_wot_fsq(historical_data, X, model_name, model):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times.
    2. the context stays in <context>, which provide more recent activities of this user.
  
    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not use line breaks in the reason.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <next_place_id>: 
    """
    if "Qwen2-VL-72B-Instruct" in model_name or "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
        completion = get_response_mllm_api(session=[{"role": "user", "content": prompt}], model_name=model_name, temperature=0, max_tokens=1200, json_mode=True)
        token_usage = 0
    else:
        if model is not None:
            completion = get_model_response_hf(prompt, model)
            token_usage = 0
        else:
            completion, token_usage = get_chat_completion(prompt=prompt, model_name=model_name, json_mode=True, max_tokens=1200, temperature=0, infer_server="LLM2-vllm")
    return completion, token_usage


def load_results(filename):
    # Load previously saved results from a CSV file    
    results = pd.read_csv(filename)
    return results

def get_response(result):
    match = re.search(r'\{.*?\}', result, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json_str
    else:
        return None

def parse_response_with_method_1(response, uid, predict_y, i):
    # Method1: Try to parse directly, and extract data through regular expressions when failed
    try:
        output = get_response(response)
        res_dict = ast.literal_eval(output)
        res_dict['ground_truth'] = predict_y[i]
        res_dict['user_id'] = uid
        return res_dict
    except Exception as e:
        if e == "'\{' was never closed \(<unknown>, line 1\)" or e == "malformed node or string: None":
            match_pred = re.search(r'"prediction":\s*(\d+)', response)
            match_res = re.search(r'"reason":\s*(\d+)', response)
            if match_pred and match_res:
                prediction_number = match_pred.group(1)
                prediction_res = match_res.group(1)
                return {
                    'prediction': prediction_number,
                    'reason': prediction_res,
                    'ground_truth': predict_y[i],
                    'user_id': uid
                }
            else:
                raise e
        else:
            raise e
        

def parse_response_with_method_2(response, uid, predict_y, i, top_k=1):
    # Method2: Try to parse directly, and process the prediction value according to the condition
    try:
        res_dict = ast.literal_eval(response)
        if top_k != 1:
            res_dict['prediction'] = str(res_dict['prediction'])  
        res_dict['user_id'] = uid
        res_dict['ground_truth'] = predict_y[i]
        return res_dict
    except Exception as e:
        raise e
    
def single_user_query(dataname, uid, historical_data, predict_X, predict_y,logger, top_k, is_wt, output_dir, sleep_query, sleep_crash, model_name, model):
    # Initialize variables
    total_queries = len(predict_X)
    logger.info(f"Total_queries: {total_queries}")

    processed_queries = 0
    current_results = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    out_filename = f"{uid:02d}" + ".csv"
    out_filepath = os.path.join(output_dir, out_filename)

    try:
        # Attempt to load previous results if available
        current_results = load_results(out_filepath)
        processed_queries = len(current_results)
        logger.info(f"Loaded {processed_queries} previous results.")
    except FileNotFoundError:
        logger.info("No previous results found. Starting from scratch.")
    all_token = 0
    # Process remaining queries
    for i in range(processed_queries, total_queries):
    #for query in queries[processed_queries:]:
        
        logger.info(f'The {i+1}th sample: ')
        #logger.info(f"context: {predict_X[i]['context_stay']}")
        #logger.info(f"target stay: {predict_X[i]['target_stay']}")
        if is_wt is True:
            if top_k == 1:
                completions, token_usage = single_query_top1_fsq(historical_data, predict_X[i], model_name, model)
            elif top_k == 10:
                completions, token_usage = single_query_top10_fsq(historical_data, predict_X[i], model_name, model)
            else:
                raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
        else:
            if top_k == 1:
                completions, token_usage = single_query_top1_wot_fsq(historical_data, predict_X[i], model_name, model)
            elif top_k == 10:
                completions, token_usage = single_query_top10_wot_fsq(historical_data, predict_X[i], model_name, model)
            else:
                raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")

        response = completions
        all_token += token_usage

        # Log the prediction results and usage.
        logger.info(f"Pred results: {response}")
        logger.info(f"Ground truth: {predict_y[i]}")
        
        try:
            res_dict = parse_response_with_method_1(response, uid, predict_y, i)
        except Exception as e1:
            logger.info(f"Method 1 failed: {e1}")
            try:
                res_dict = parse_response_with_method_2(response, uid, predict_y, i, top_k)
            except Exception as e2:
                logger.info(f"Method 2 failed: {e2}")
                res_dict = {'user_id': uid, 'ground_truth': predict_y[i], 'prediction': -100, 'reason': None}

        new_row = pd.DataFrame(res_dict, index=[0])  
        current_results = pd.concat([current_results, new_row], ignore_index=True)  

    # Save the current results
    current_results.to_csv(out_filepath, index=False)
    #save_results(current_results, out_filename)
    logger.info(f"Saved {len(current_results)} results to {out_filepath}")
    logger.info(f"token usage for each user: {all_token}")
    logger.info(f"trajories for each user: {len(predict_y)}")

    # Continue processing remaining queries
    if len(current_results) < total_queries:
        #remaining_predict_X = predict_X[len(current_results):]
        #remaining_predict_y = predict_y[len(current_results):]
        #remaining_queries = queries[len(current_results):]
        logger.info("Restarting queries from the last successful point.")
        single_user_query(dataname, uid, historical_data, predict_X, predict_y,
                          logger, top_k, is_wt, output_dir, sleep_query, sleep_crash, model_name, model)



def query_all_user(data_version, split_path, dataname, uid_list, logger, train_data, num_historical_stay,
                   num_context_stay, test_file, top_k, is_wt, output_dir, sleep_query, sleep_crash, model_name, sample_single_user, model):
    all_traj = 0
    for uid in uid_list:  
        # train_data is concatenated train and val, test is a dictionary
        logger.info(f"=================Processing user {uid}==================")
        user_train = get_user_data(train_data, uid, num_historical_stay, logger)  
        historical_data, predict_X, predict_y = organise_data(data_version, split_path, dataname, user_train, test_file, uid, logger, num_context_stay, sample_single_user)   
        all_traj += len(predict_y)
        single_user_query(dataname, uid, historical_data, predict_X, predict_y, logger, top_k=top_k,
                          is_wt=is_wt, output_dir=output_dir, sleep_query=sleep_query, sleep_crash=sleep_crash, model_name=model_name, model=model)
    return all_traj


# Get the remaning user
def get_unqueried_user(dataname, user_cnt, test_path="./checkint_test_pk/", output_dir='output/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(test_path, f"{dataname}_fin.pk"), "rb") as f:
        test_files = pickle.load(f)  # test_file is a list of dict
    user_in_test = set()
    for test_file in test_files:
        X = test_file['user_X']
        user_in_test.add(X)
    all_user_id = random.choices(list(user_in_test), k=user_cnt)
    processed_id = [int(file.split('.')[0]) for file in os.listdir(output_dir) if file.endswith('.csv')]
    remain_id = [i for i in all_user_id if i not in processed_id]
    print(remain_id)
    print(f"Number of the remaining id: {len(remain_id)}")
    return remain_id


def main(city, model_name, user_cnt=50, sample_single_user=10, num_historical_stay=40, num_context_stay=5, split_path="./checkin_split/", test_path="./checkin_test_pk/", data_version="all"):
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

    all_traj = 0  
    datanames = [
        "Beijing", "Cape", "London", "Moscow", "Mumbai", "Nairobi", "NewYork" ,"Paris" ,"San", "Sao", "Shanghai", "Sydney","Tokyo"
    ]

    prediction_results_path = os.path.join(RESULTS_PATH, "prediction_results")
    prediction_logs_path = os.path.join(RESULTS_PATH, "prediction_results", "logs")
    os.makedirs(prediction_results_path, exist_ok=True)
    os.makedirs(prediction_logs_path, exist_ok=True)

    for dataname in datanames:
        
        if dataname != city:
            continue
        top_k = 1  # the number of output places k
        with_time = True  # whether incorporate temporal information for target stay
        sleep_single_query = 0.1  # the sleep time between queries (after the recent updates, the reliability of the API is greatly improved, so we can reduce the sleep time)
        sleep_if_crash = 1  # the sleep time if the server crashes

        if "/" in model_name:
            model_name_back = model_name.replace("/", "_")
        else:
            model_name_back = model_name
        output_dir = os.path.join(prediction_results_path, f"{model_name_back}_{dataname}_top{top_k}_wot")  # the output path
        log_dir = os.path.join(prediction_logs_path, f"{model_name_back}_{dataname}_top{top_k}_wot")  # the log dir

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        tv_data, test_file = get_dataset(dataname, split_path=split_path, test_path=test_path, data_version=data_version)  

        logger = get_logger('my_logger', log_dir=log_dir)

        uid_list = get_unqueried_user(dataname, user_cnt, test_path=test_path, output_dir=output_dir)  
        print(f"uid_list: {uid_list}")

        traj = query_all_user(data_version, split_path, dataname, uid_list, logger, tv_data, num_historical_stay, num_context_stay,
                       test_file, output_dir=output_dir, top_k=top_k, is_wt=with_time,
                       sleep_query=sleep_single_query, sleep_crash=sleep_if_crash, model_name=model_name, sample_single_user=sample_single_user, model=model)
        all_traj += traj

        print("Query done")
    acc1, f1 = cal_metrics(output_dir=output_dir)
    print("city:{} all_traj:{} acc1:{} f1:{}".format(city, all_traj, acc1, f1))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, choices=["Beijing", "Cape", "London", "Moscow", "Mumbai", "Nairobi", "NewYork" ,"Paris" ,"San", "Sao", "Shanghai", "Sydney","Tokyo"])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--user_cnt', type=int, default=50, help="Number of users")
    parser.add_argument('--sample_single_user', type=int, default=10, help="Number of samples for each user")
    parser.add_argument('--data_name',type=str, default='mini', choices=['all','mini'])
    parser.add_argument('--split_path', type=str, default=MOBILITY_SPLIT_PATAH)
    parser.add_argument('--test_path', type=str, default=MOBILITY_TEST_PATH)
    parser.add_argument('--num_historical_stay', type=int, default=40)
    parser.add_argument('--num_context_stay', type=int, default=5)
    
    args = parser.parse_args()

    main(
        args.city_name, 
        args.model_name, 
        user_cnt=args.user_cnt, 
        sample_single_user=args.sample_single_user, 
        num_historical_stay=args.num_historical_stay, 
        num_context_stay=args.num_context_stay,
        split_path=args.split_path,
        test_path=args.test_path,
        data_version=args.data_name
        )
