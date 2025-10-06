import os

PROXY = "http://127.0.0.1:10190"

RESOURCE_PATH=os.path.join("resource/")
RESULTS_PATH="results/"
# TODO: Complete this URL of the MongoDB database
MONGODB_URI = ""

# Common Map data
# TODO: Fill in the path of your evaluation data
EVAL_COMMON_PATH=""
MAP_DATA_PATH=os.path.join(EVAL_COMMON_PATH, "EXP_ORIG_DATA/")
MAP_CACHE_PATH=os.path.join(EVAL_COMMON_PATH, "map_cache/")
ROUTING_PATH=os.path.join(EVAL_COMMON_PATH, "routing_linux_amd64")

# GeoQA
GEOQA_SAMPLE_RATIO=0.1
GEOQA_DATA_PATH=os.path.join(EVAL_COMMON_PATH, "task_Geo_knowledge/")

# mobility predictino task
MOBILITY_SAMPLE_RATIO=0.1
MOBILITY_SPLIT_PATAH=os.path.join(EVAL_COMMON_PATH, "mobility", "checkin_split/")
MOBILITY_TEST_PATH=os.path.join(EVAL_COMMON_PATH, "mobility", "checkin_test_pk/")

# urban exploration task
EXPLORATION_TASK_PATH=os.path.join(EVAL_COMMON_PATH, "navigation_tasks")

# remote_sensing task
REMOTE_SENSING_PATH=os.path.join(EVAL_COMMON_PATH, "remote_sensing/")
REMOTE_SENSING_RESULTS_PATH=os.path.join(RESULTS_PATH, "remote_sensing/")
REMOTE_SENSING_ZOOM_LEVEL=15
WORLD_POP_DATA_PATH="{}ppp_2020_1km_Aggregated.tif".format(REMOTE_SENSING_PATH)

# street view stask
STREET_VIEW_PATH=os.path.join(EVAL_COMMON_PATH, "street_view/")
STEET_VIEW_RESULTS_PATH = os.path.join(RESULTS_PATH, "street_view/")

# outdoor navigation task
NAVIGATION_IMAGE_FOLDER=os.path.join(EVAL_COMMON_PATH, "NEW_StreetView_Images_CUT")
NAVIGATION_FILE_PATH=os.path.join(EVAL_COMMON_PATH, "outdoor_navigation/")
NAVIGATION_URL_PATH=os.path.join(NAVIGATION_IMAGE_FOLDER, "url_mapping_UrbanLLaVA_20241004_2week.csv")
NAVIGATION_URL_PATH_now=os.path.join(NAVIGATION_FILE_PATH, "url_mapping_UrbanLLaVA_20241025_1month.csv")

# TODO: Adjust the path to your own data
# uni-image tasks
UNI_IMAGE_FOLDER= os.path.join(EVAL_COMMON_PATH, "uniimage")
# Source data
BEIJING_STV_IMAGE_FOLDER= os.path.join(EVAL_COMMON_PATH, "ThreeCityImage/Beijing/StreetView")

MULTI_IMAGE_FOLDER=os.path.join(EVAL_COMMON_PATH, "multiimage")

# cross-view task
CROSS_VIEW_PATH = os.path.join(EVAL_COMMON_PATH, "cross_view/")
CROSS_VIEW_RESULTS_PATH = os.path.join(RESULTS_PATH,"cross_view/")

# TODO: We need to define a mapping list to unify the model names deployed on different platforms and frameworks
VLM_MODELS = [
    "QwenVLPlus", "GPT4o", "GPT4o_MINI", "cogvlm2-llama3-chat-19B", "InternVL2-40B", "MiniCPM-Llama3-V-2_5", "llava_next_yi_34b", "llava_next_llama3", "Yi_VL_6B", "Yi_VL_34B", "llava_v1.5_7b", "glm-4v-9b", "InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26B", "Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct", "VILA1.5-3b", "Llama-3-VILA1.5-8b", "VILA1.5-13b", "UrbanLLaVA-8b-mix-v1", "UrbanLLaVA-8b-mix-citywalk", "UrbanLLaVA-8b-mix-citywalk-expand", "UrbanLLaVA-8b-mix-multi", "UrbanLLaVA-8b-mix-single", "UrbanLLaVA-8b-mix-v2", "UrbanLLaVA-8b-mix-v3", "UrbanLLaVA-8b-mix-v4", "UrbanLLaVA-8b-test", "UrbanLLaVA-8b-mix-v5"]


# TODO: Fill in the actual path of your models

VLLM_MODEL_PATH_PREFIX = ""
VLLM_MODEL_PATH_PREFIX2 = ""
VLLM_MODEL_PATH_PREFIX3 = ""
VLLM_MODEL_PATH = {
    "cogvlm2-llama3-chat-19B": os.path.join(VLLM_MODEL_PATH_PREFIX, "cogvlm2-llama3-chat-19B"),
    "InternVL2-40B": os.path.join(VLLM_MODEL_PATH_PREFIX, "InternVL2-40B"),
    "MiniCPM-Llama3-V-2_5": os.path.join(VLLM_MODEL_PATH_PREFIX, "MiniCPM-Llama3-V-2_5"),
    "llava_next_yi_34b": os.path.join(VLLM_MODEL_PATH_PREFIX, "llava-v1.6-34b-hf"),
    "llava_next_llama3": os.path.join(VLLM_MODEL_PATH_PREFIX, "llama3-llava-next-8b-hf"),
    "llava_v1.5_7b": os.path.join(VLLM_MODEL_PATH_PREFIX2, "llava-1___5-7b-hf"),
    "glm-4v-9b": os.path.join(VLLM_MODEL_PATH_PREFIX2, "glm-4v-9b"),
    "Qwen2-VL-2B-Instruct": os.path.join(VLLM_MODEL_PATH_PREFIX, "Qwen2-VL-2B-Instruct"),
    "Qwen2-VL-7B-Instruct": os.path.join(VLLM_MODEL_PATH_PREFIX2, "Qwen2-VL-7B-Instruct"),
    "Yi_VL_6B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "Yi-VL-6B"),
    "Yi_VL_34B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "Yi-VL-34B"),
    "InternVL2-2B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-2B"),
    "InternVL2-4B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-4B"),
    "InternVL2-8B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-8B"),
    "InternVL2-26B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-26B"),
    "VILA1.5-3b": os.path.join(VLLM_MODEL_PATH_PREFIX, "VILA1.5-3b"),
    "Llama-3-VILA1.5-8b": os.path.join(VLLM_MODEL_PATH_PREFIX, "Llama-3-VILA1.5-8b"),
    "VILA1.5-13b": os.path.join(VLLM_MODEL_PATH_PREFIX, "VILA1.5-13b"),
    # TODO: Fill in the name and path of your model
    # if you use vila, the path must contain the size of the model, 3b, 8b or 13b, so that it can be correctly recognized and added by vlmeval
    "UrbanLLaVA-8b-mix-v1": os.path.join(VLLM_MODEL_PATH_PREFIX3, "UrbanLLaVA-8b-20241007-mix-v1"),
}


OPENAI_APIKEY = os.environ["OpenAI_API_KEY"]
DEEPINFRA_APIKEY = os.environ["DeepInfra_API_KEY"]
SILICONFLOW_APIKEY = os.environ["SiliconFlow_API_KEY"]
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]


LLM_MODELS = [
    "Qwen2-7B", "Qwen2-72B", "Intern2.5-7B", "Intern2.5-20B", 
    "Mistral-7B", "Mixtral-8x22B", "LLama3-8B", "LLama3-70B", "Gemma2-9B", "Gemma2-27B", 
    "DeepSeekV2", "GPT3.5-Turbo", "GPT4-Turbo", "GPT4omini"]
INFER_SERVER = {
    "OpenAI": ["GPT3.5-Turbo", "GPT4-Turbo", "GPT4omini", "GPT4o"],
    "DeepInfra": ["Mistral-7B", "Mixtral-8x22B", "LLama3-8B", "LLama3-70B", "Gemma2-9B", "Gemma2-27B",],
    "Siliconflow": ["Qwen2-7B", "Qwen2-72B", "Intern2.5-7B", "Intern2.5-20B", "DeepSeekV2",]
}
LLM_MODEL_MAPPING = {
    "Qwen2-7B":"Qwen/Qwen2-7B-Instruct",
    "Qwen2-72B":"Qwen/Qwen2-72B-Instruct",
    "Intern2.5-7B":"internlm/internlm2_5-7b-chat",
    "Intern2.5-20B":"internlm/internlm2_5-20b-chat",
    "Mistral-7B":"mistralai/Mistral-7B-Instruct-v0.2", 
    "Mixtral-8x22B":"mistralai/Mixtral-8x22B-Instruct-v0.1",
    "LLama3-8B":"meta-llama/Meta-Llama-3-8B-Instruct",
    "LLama3-70B":"meta-llama/Meta-Llama-3-70B-Instruct",
    "Gemma2-9B":"google/gemma-2-9b-it",
    "Gemma2-27B":"google/gemma-2-27b-it",
    "DeepSeekV2":"deepseek-ai/DeepSeek-V2-Chat",
    "GPT3.5-Turbo":"gpt-3.5-turbo-0125",
    "GPT4-Turbo":"gpt-4-turbo-2024-04-09",
    "GPT4omini":"gpt-4o-mini-2024-07-18",
    "GPT4o":"gpt-4o"
}

# This is boundaries of the cities
CITY_BOUNDARY = {
    # Beijing
    # Left bottom: 116.26, 39.96, Right top: 116.40, 40.03
    "Beijing": [(116.26, 39.96), (116.40,39.96), (116.40, 40.03), (116.26, 40.03), (116.26, 39.96)],

    # Left bottom: -0.1868, 51.4874, Right top: -0.10326, 51.5194
    "London": [(-0.1868, 51.4874), (-0.10326, 51.4874), (-0.10326, 51.5194), (-0.1868, 51.5194), (-0.1868, 51.4874)],

    # Left bottom: -74.0128, 40.7028, Right top: -73.9445, 40.7314
    "NewYork":  [(-74.0128, 40.7028), (-73.9445, 40.7028), (-73.9445, 40.7314), (-74.0128, 40.7314), (-74.0128, 40.7028)],
}

TASK_DEST_MAPPING = {
    # text
    "geoqa": "evaluate.geoqa.run_eval",
    "mobility": "evaluate.mobility_prediction.llm_mob",
    "exploration": "evaluate.urban_exploration.eval",
    # visual
    "objects": "evaluate.remoet_sensing.eval_inference",
    "geoloc": "evaluate.street_view.eval_inference",
    "navigation": "evaluate.outdoor_navigation.eval"
    # cross-view
    }

# Original city map data
MAP_DICT = {
    "Beijing":"map_beijing_20240808",
    "Shanghai":"map_shanghai_20240806",
    "Mumbai":"map_mumbai_20240806",
    "Tokyo":"map_tokyo_20240807",
    "London":"map_london_20240807",
    "Paris":"map_paris_20240808",
    "Moscow":"map_moscow_20240807",
    "NewYork":"map_newyork_20240808",
    "SanFrancisco":"map_san_francisco_20240807",
    "SaoPaulo":"map_san_paulo_20240808",
    "Nairobi":"map_nairobi_20240807",
    "CapeTown":"map_cape_town_20240808",
    "Sydney":"map_sydney_20240807"
}
