cities=("Beijing" "NewYork" "London")
work_dir="../../data/"
model="gpt-4o-mini-2024-07-18"

# Depend on SAT_interpolate_address_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting SAT combined address for $city"
    python ./annotate/sat_combine_address.py --city $city --work_dir $work_dir --model_name $model
done
wait
echo "Finish getting SAT combined address"
# Get sat_address_combined_{city}_{zl}.csv

# Depend on SAT_interpolate_address_{city}_{zl}.csv
# Depend on short_clipped_results_{zl}/driving_{img_name}.txt
# Depend on short_clipped_results_{zl}/pois_{img_name}.txt
for city in "${cities[@]}"; do
    echo "Getting SAT scene description for $city"
    python ./annotate/sat_scene_description.py --city $city --work_dir $work_dir --model_name $model
done
wait
echo "Finish getting SAT scene description"
# Get rs_osm_description_{city}_{zl}.csv

# Depend on SAT_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting SAT grounding description for $city"
    python ./annotate/sat_generate_grounding_template.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting SAT grounding description"
# Get rs_grounding_selfmade_{zl}.csv

# Depend on SAT_{city}_{zl}.csv
# Depend on short_clipped_results_{zl}/landuse_{img_name}.txt
for city in "${cities[@]}"; do
    echo "Getting SAT landuse for $city"
    python ./annotate/sat_landuse_template.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting SAT landuse"
# Get rs_landuse_description_{zl}.csv

# Depend on stv_in_sat_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting STV scene description for $city"
    python ./annotate/stv_description_gpt.py --city $city --work_dir $work_dir --model_name $model
done
wait
echo "Finish getting STV scene description"
# Get stv_description.csv

# Depend on stv_in_sat_nearest_features_update_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting STV landmark for $city"
    python ./annotate/stv_landmark_gpt.py --city $city --work_dir $work_dir --model_name $model
done
wait
echo "Finish getting STV landmark"
# Get stv_poi_landmark_update.jsonl