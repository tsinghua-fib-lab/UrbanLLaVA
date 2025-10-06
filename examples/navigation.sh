# source /usr/local/miniconda3/bin/activate vila-vlmeval
export CUDA_VISIBLE_DEVICES=2
export DeepInfra_API_KEY=""
export SiliconFlow_API_KEY=""                   
export OpenAI_API_KEY=""
export OPENAI_API_KEY="$OpenAI_API_KEY"
export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"
export DASHSCOPE_API_KEY=""


CITIES=('Beijing' 'NewYork' 'London')
MODELS=("Llama-3-VILA1.5-8b" "GPT4o_MINI")
DATA_VERSION='all'


DATA_VERSION='all'

echo "Start running navigation"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m evaluate.outdoor_navigation.eval --model_name $MODEL --data_name $DATA_VERSION --city_name $CITY 
    done
done
echo "Finish running navigation"
