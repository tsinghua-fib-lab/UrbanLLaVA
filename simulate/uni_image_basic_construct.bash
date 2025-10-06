cities=("Beijing" "NewYork" "London")
work_dir="../../data/"


# Dependency: rs_osm_description_{city}_{zl}.csv
#             sat_address_combined_{city}_{zl}.csv
#             rs_grounding_selfmade_{zl}.csv
#             rs_landuse_description_{zl}.jsonl
#             stv_in_sat_address_deploy_{zl}.csv
#             stv_description.jsonl
#             stv_poi_landmark_update.jsonl

for city in "${cities[@]}"; do
    echo "Formatting data to VILA format for $city"
    python ./format/uni_basic_llava.py --city $city --work_dir $work_dir
done
wait
echo "Finish formatting data to VILA format"

# Get llava/format/{city}_basic_all_data_llava.json, etc.