source /usr/local/anaconda3/bin/activate vllm
export CUDA_VISIBLE_DEVICES=4

USER=""
API_KEY=""
SERVER_IP=""
SERVER_PORT=23199
MODEL_NAME=llama3-8B
MODEL_PATH=/path/Meta-Llama-3-8B-Instruct/

exec -a "vllm-$MODEL_NAME@$USER" python -m vllm.entrypoints.openai.api_server \
  --served-model-name $MODEL_NAME \
  --api-key $API_KEY \
  --model $MODEL_PATH \
  --trust-remote-code \
  --host $SERVER_IP \
  --port $SERVER_PORT \
  --max-model-len 4096 \
  --disable-log-stats \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95

# more settings please refer to the following docs
# vllm installation https://docs.vllm.ai/en/latest/getting_started/installation.html
# autoAWQ https://docs.vllm.ai/en/latest/quantization/auto_awq.html
# vllm engine parameters: https://docs.vllm.ai/en/latest/models/engine_args.html
# vllm openai server parameters: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html