# source /usr/local/miniconda3/bin/activate vila-vlmeval
export CUDA_VISIBLE_DEVICES=4
export DeepInfra_API_KEY=""
export SiliconFlow_API_KEY=""                   
export OpenAI_API_KEY=""
export OPENAI_API_KEY="$OpenAI_API_KEY"
export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"
export DASHSCOPE_API_KEY=""
CITIES=('Beijing' 'London' 'NewYork')
MODELS=("Llama-3-VILA1.5-8b" "GPT4o_MINI")
DATA_VERSION='all'

echo "Start running evaluation on street view address task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        # python -m evaluate.uniimage.stv_address.stv_address_convert --city_name $CITY --task_name stv_address_mc
        python -m evaluate.uniimage.stv_address.stv_address_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION --task_name stv_address_mc
        python -m evaluate.uniimage.stv_address.stv_address_stats --city_name $CITY --model_name $MODEL --task_name stv_address_mc
    done
done

echo "Start running evaluation on street view landmark task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        # python -m evaluate.uniimage.stv_landmark.stv_landmark_convert --city_name $CITY --task_name stv_landmark_mc
        python -m evaluate.uniimage.stv_landmark.stv_landmark_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION --task_name stv_landmark_mc
        python -m evaluate.uniimage.stv_landmark.stv_landmark_stats --city_name $CITY --model_name $MODEL --task_name stv_landmark_mc
    done
done

echo "Start running evaluation on satellite address task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        # python -m evaluate.uniimage.sat_address.sat_address_convert --city_name $CITY --task_name sat_address_mc
        python -m evaluate.uniimage.sat_address.sat_address_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION --task_name sat_address_mc
        python -m evaluate.uniimage.sat_address.sat_address_stats --city_name $CITY --model_name $MODEL --task_name sat_address_mc
    done
done

echo "Start running evaluation on satellite landuse task"
for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        # python -m evaluate.uniimage.sat_landuse.sat_landuse_convert --city_name $CITY --task_name sat_landuse_mc
        python -m evaluate.uniimage.sat_landuse.sat_landuse_inference --city_name $CITY --model_name $MODEL --data_name $DATA_VERSION --task_name sat_landuse_mc
        python -m evaluate.uniimage.sat_landuse.sat_landuse_stats --city_name $CITY --model_name $MODEL --task_name sat_landuse_mc
    done
done