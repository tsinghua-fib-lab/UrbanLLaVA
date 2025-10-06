import os
import re
import httpx
import time
from thefuzz import process
from openai import OpenAI

from config import PROXY, OPENAI_APIKEY, DEEPINFRA_APIKEY, SILICONFLOW_APIKEY, LLM_MODEL_MAPPING, INFER_SERVER


def get_chat_completion(session, model_name, max_tokens=1200, temperature=0, infer_server=None, json_mode=False):
    client, model_name = get_llm_model_client(model_name, infer_server)

    MAX_RETRIES = 3
    WAIT_TIME = 1
    for i in range(MAX_RETRIES):
        try:
            if json_mode:
                response = client.chat.completions.create(
                            model=model_name,
                            response_format={"type": "json_object"},
                            messages=session,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                token_usage = response.usage.completion_tokens
                client.close()
                return response.choices[0].message.content, token_usage
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=session,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                token_usage = response.usage.completion_tokens
                client.close()
                return response.choices[0].message.content, token_usage
        except Exception as e:
            if i < MAX_RETRIES - 1:
                time.sleep(WAIT_TIME)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "OpenAI API Error.",0

def get_response_mllm_api(session, model_name, temperature=0, max_tokens=1000, infer_server=None,json_mode=False):
    
    if "Qwen2-VL-72B-Instruct" in model_name:
        infer_server = "Siliconflow"
        client = OpenAI(
        api_key=SILICONFLOW_APIKEY,
        base_url="https://api.siliconflow.cn/v1"
        )
    elif "Llama-3.2-90B-Vision-Instruct" in model_name or "Llama-3.2-11B-Vision-Instruct" in model_name:
        infer_server = "DeepInfra"
        client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=DEEPINFRA_APIKEY,
        http_client=httpx.Client(proxies=PROXY),
            )
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            if json_mode:
                response = client.chat.completions.create(
                            model=model_name,
                            response_format={"type": "json_object"},
                            messages=session,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                client.close()
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                        model=model_name,
                        messages=session,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                client.close()
                return response.choices[0].message.content
        except Exception as e:
            retries += 1
            print(f"Error calling GPT API: {e}. Retry {retries}/{max_retries}...")

            if retries < max_retries:
                time.sleep(2 ** retries)  
    client.close()
    return None
    
def get_llm_model_client(model_name, infer_server=None):

    if infer_server is None or infer_server not in INFER_SERVER.keys():
        infer_server = None
        for server_name in INFER_SERVER:
            if model_name in INFER_SERVER[server_name]:
                infer_server=server_name
                break
        if infer_server is None:
            raise NotImplementedError
    
    model_name = LLM_MODEL_MAPPING[model_name]

    if infer_server=='OpenAI':
        client = OpenAI(
            http_client=httpx.Client(proxy=PROXY),
            api_key=OPENAI_APIKEY
            )
    elif infer_server =="DeepInfra":
        client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=DEEPINFRA_APIKEY,
        http_client=httpx.Client(proxies=PROXY),
            )
    elif infer_server =="Siliconflow":
        client = OpenAI(
        api_key=SILICONFLOW_APIKEY,
        base_url="https://api.siliconflow.cn/v1"
        )
    else:
        raise NotImplementedError

    return client, model_name


def get_response_traffic_signal(prompt, model_name, max_tokens=500, temperature=0):

    if model_name in ['fixed-time', 'max-pressure']:
        return "Template for fixed-time/max-pressure control"
    
    dialogs = [{"role": "user", "content": prompt}]
    res, _ = get_chat_completion(session=dialogs, model_name=model_name, max_tokens=max_tokens, temperature=temperature)

    return res


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^A-H]{0,20}?(?:n't|not))[^A-H]{0,10}?\b(?:|is|:|be))\b)[^A-H]{0,20}?\b([A-H])\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b([A-H])\b(?![^A-H]{0,8}?(?:n't|not)[^A-H]{0,5}?(?:correct|right))[^A-H]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^([A-H])(?:\.|,|:|$)", gen)

    # simply extract the first appeared letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])([A-H])(?![a-zA-Z=])", gen)

    if res is None:
        return choice_list[choice_list.index(process.extractOne(gen, choice_list)[0])]
    
    return res.group(1)


def get_model_response_gpt(messages, model_name="gpt-4o-mini-2024-07-18"):
    
    max_retries = 5
    client = OpenAI(
        api_key=OPENAI_APIKEY,
        http_client=httpx.Client(proxies=PROXY)
    )
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                    "role": "user",
                    "content": messages
                    }
                ],
                max_tokens=int(1200),
                )
            # print(prompt)
            # print(completion.choices[0].message.content)            
            return completion.choices[0].message.content
        except Exception as e:
            retries += 1
            print(f"Error calling GPT API: {e}. Retry {retries}/{max_retries}...")

            if retries < max_retries:
                time.sleep(2 ** retries)  

    print(f"Failed after {max_retries} attempts. Returning None.")
    return None

def get_model_response_hf(prompt, model):
    response = model.generate(prompt)
    # print(response)
    return response

def get_model_response_hf_image(image_path, prompt, model):
    response = model.generate([image_path, prompt])
    # print(response)
    return response

def get_model_response_vllm(prompt, model):
    
    model_name, port = model.split(":")
    # TODO: Fill in the correct url and api key
    client = OpenAI(
                base_url="",
                api_key="",
            )
    MAX_RETRIES = 1
    WAIT_TIME = 1
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                    )
            return response.choices[0].message.content
        except Exception as e:
            if i < MAX_RETRIES - 1:
                time.sleep(WAIT_TIME)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "OpenAI API Error."
    # print(response)
    return response


def match_response(response):

    try:
        reason_part = response.split("Reason: ")[1].split("Action: ")[0].strip()
        action_part = response.split("Action: ")[1].strip()
        pattern = r"\b(left|right|forward|stop)\b"
        match = re.search(pattern, action_part, re.IGNORECASE)
        
        if match:
            action = match.group(0).lower()
        else:
            action = None
        # print(f"Reason: {reason_part}, action: {action}")
        return reason_part, action
    except IndexError:
        return None,None
