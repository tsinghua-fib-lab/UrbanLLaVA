# source /usr/local/miniconda3/bin/activate vila-vlmeval
export CUDA_VISIBLE_DEVICES=5
export DeepInfra_API_KEY=""
export SiliconFlow_API_KEY=""                   
export OpenAI_API_KEY=""
export OPENAI_API_KEY="$OpenAI_API_KEY"
export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"
export DASHSCOPE_API_KEY=""
MODELS=("GPT4o_MINI" "Llama-3-VILA1.5-8b")

DATA_VERSION='all'

echo "Start running evaluation on general tasks"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
        python -m evaluate.general.general_inference --model_name $MODEL --data_name $DATA_VERSION
        python -m evaluate.general.general_stats --model_name $MODEL
done