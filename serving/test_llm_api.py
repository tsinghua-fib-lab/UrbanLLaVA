from openai import OpenAI
import httpx
import os

#### register your API_Key in the environment, do not directly write your key in codes
#export SiliconFlow_API_KEY="xx"
#export DeepInfra_API_KEY="xx"
#export OpenAI_API_KEY="xx"

#### define your proxy
PROXY = "http://127.0.0.1:10190"

#### select platforms
API_KEY_MAPPING = {
    "siliconflow": "SiliconFlow_API_KEY",   # https://siliconflow.cn/models
    "DeepInfra": "DeepInfra_API_KEY",       # https://deepinfra.com/models
    "OpenAI": "OpenAI_API_KEY",             # https://openai.com/api/pricing/
    "vllm": "vllm_KEY"
}
API_URL_MAPPING = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "DeepInfra": "https://api.deepinfra.com/v1/openai",
    "OpenAI": "https://api.openai.com/v1",
    "vllm": "http://your_server_ip:port/v1",
}


API_TYPE = "OpenAI"
API_KEY = os.environ[API_KEY_MAPPING[API_TYPE]]
API_URL = API_URL_MAPPING[API_TYPE]
model_name = "google/gemma-2-9b-it"


#### OpenAI client
if API_TYPE == "OpenAI":
    model_name = "gpt-3.5-turbo-0125"
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY,
        http_client=httpx.Client(proxies=PROXY)
    )
elif API_TYPE == "siliconflow":
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY
    )
elif API_TYPE=="DeepInfra":
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY,
        http_client=httpx.Client(proxies=PROXY),
    )
elif API_TYPE=="vllm":
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY
    )


#### one example
dialogs = [{
        "role": "user",
        "content": "Who are you? Please output your name with JSON format."
    }]

completion = client.chat.completions.create(
  model=model_name,
  messages=dialogs,
  max_tokens=100,
  temperature=0
)

print(completion.choices[0].message.content)
