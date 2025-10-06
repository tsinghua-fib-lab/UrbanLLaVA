# source /usr/local/miniconda3/bin/activate vila-vlmeval
export CUDA_VISIBLE_DEVICES=3
export DeepInfra_API_KEY=""
export SiliconFlow_API_KEY=""                   
export OpenAI_API_KEY=""
export OPENAI_API_KEY="$OpenAI_API_KEY"
export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"
export DASHSCOPE_API_KEY=""
CITIES=('Beijing' 'NewYork' 'London')
MODELS=("Llama-3-VILA1.5-8b" "GPT4o_MINI")
DATA_VERSION='all'

echo "Start running evaluation on SAT_count_buildings task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m evaluate.cross_view.SAT_count_buildings.SAT_count_buildings_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION
        python -m evaluate.cross_view.SAT_count_buildings.SAT_count_buildings_stats --city_name $CITY --model_name $MODEL
    done
done

echo "Start running evaluation on SAT_count_pois task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m evaluate.cross_view.SAT_count_pois.SAT_count_pois_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION
        python -m evaluate.cross_view.SAT_count_pois.SAT_count_pois_stats --city_name $CITY --model_name $MODEL
    done
done

echo "Start running evaluation on STV_SAT_location task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m evaluate.cross_view.STV_SAT_location.STV_SAT_location_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION
        python -m evaluate.cross_view.STV_SAT_location.STV_SAT_location_stats --city_name $CITY --model_name $MODEL
    done
done

echo "Start running evaluation on STV_SAT_mapping task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m evaluate.cross_view.STV_SAT_mapping.STV_SAT_mapping_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION
        python -m evaluate.cross_view.STV_SAT_mapping.STV_SAT_mapping_stats --city_name $CITY --model_name $MODEL
    done
done