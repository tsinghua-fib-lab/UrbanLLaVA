import os
import argparse
import transformers

from functools import partial
from config import VLLM_MODEL_PATH, VLM_MODELS


class VLMWrapper:
    def __init__(self, model_name, max_new_tokens=1000):
        self.model_name = model_name
        assert self.model_name in VLM_MODELS

        transformers_version_436=["VILA1.5-3b", "Llama-3-VILA1.5-8b", "VILA1.5-13b"]
        transformers_version_437=["cogvlm2-llama3-chat-19B", "InternVL2-40B", "llava_v1.5_7b", "Yi_VL_6B", "Yi_VL_34B",
                                  "InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26B"]
        transformers_version_440=["MiniCPM-Llama3-V-2_5"]
        transformers_version_444=["llava_next_yi_34b", "llava_next_llama3", "glm-4v-9b"]
        trainformers_version_latest = ["Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct"]
        # The following is an example of the model name of UrbanLLaVA
        # TODO: Add your model name of UrbanLLaVA here
        transformers_version_UrbanLLaVA=["UrbanLLaVA-8b-mix-v1"]

        # Install the correct version of transformers
        if self.model_name in transformers_version_UrbanLLaVA:
            if transformers.__version__ != "4.36.2":
                os.system("pip install transformers==4.36.2")
        elif self.model_name in transformers_version_436:
            if transformers.__version__ != "4.36.2":
                os.system("pip install transformers==4.36.2")
        elif self.model_name in transformers_version_437:
            if transformers.__version__ != "4.37.0":
                os.system("pip install transformers==4.37.0")
        elif self.model_name in transformers_version_440:
            if transformers.__version__ != "4.40.0":
                os.system("pip install transformers==4.40.0")
        elif self.model_name in transformers_version_444:
            if transformers.__version__ != "4.44.2":
                os.system("pip install transformers==4.44.2")
        elif self.model_name in trainformers_version_latest:
            if transformers.__version__ != "4.45.0.dev0":
                os.system("pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate")
        else:
            print("no need to update transformers")

        # place this line after the command "pip install"
        try:
            from vlmeval.config import supported_VLM
            from vlmeval.vlm import VILA
            from functools import partial
        except Exception as e:
            print(e)
            print("need to run this script in vlmeval")


        # only update local model path
        for model_name in transformers_version_436 + transformers_version_437 + transformers_version_440 + transformers_version_444 + trainformers_version_latest:
            original_func = supported_VLM[model_name]  
            if "glm" in model_name or "cogvlm" in model_name:
                supported_VLM[model_name] = partial(original_func.func, 
                                                    model_path=VLLM_MODEL_PATH[model_name],
                                                    max_length=max_new_tokens,
                                                    **{k: v for k, v in original_func.keywords.items() if k != 'model_path'})
            else:
                supported_VLM[model_name] = partial(original_func.func, 
                                                    model_path=VLLM_MODEL_PATH[model_name],
                                                    max_new_tokens=max_new_tokens,
                                                    **{k: v for k, v in original_func.keywords.items() if k != 'model_path'})
        try:
            for model_name in transformers_version_UrbanLLaVA:
                supported_VLM[model_name]=partial(VILA, model_path=VLLM_MODEL_PATH[model_name], max_new_tokens=max_new_tokens)
        except Exception as e:
            print(e)
            print("UrbanLLaVA is not supported")

        self.enable_proxy()
        self.model = supported_VLM[self.model_name]()

    def get_vlm_model(self):
        return self.model

    def enable_proxy(self):
        # set proxy for OpenAI models
        if self.model_name in ["GPT4o", "GPT4o_MINI"]:
            os.environ["http_proxy"] = 'http://127.0.0.1:10190'
            os.environ["https_proxy"] = 'http://127.0.0.1:10190'


    def clean_proxy(self):
        try:
            del os.environ["http_proxy"]
            del os.environ["https_proxy"]
        except Exception as e:
            print("Failed to delete proxy environment")
