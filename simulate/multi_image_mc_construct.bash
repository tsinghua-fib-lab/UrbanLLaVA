cities=("Beijing" "NewYork" "London")
work_dir="../../../data/"

for city in "${cities[@]}"; do
    echo  "Getting POIs and buildings number for $city"
    python ./advance/cross-view/generate_poi_building_count.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting POIs and buildings number"

for city in "${cities[@]}"; do
    echo  "Generating SAT-Count data for $city"
    python ./format/multi_SAT_count_llava.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating SAT-Count data"

for city in "${cities[@]}"; do
    echo  "Getting street view images and corresponding satellite images for $city"
    python ./advance/cross-view/SAT_stv_corres.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting street view images and corresponding satellite images"

for city in "${cities[@]}"; do
    echo  "Getting partition information between street view images and satellite images for $city"
    python ./advance/cross-view/stv_in_sat_partition.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting partition information between street view images and satellite images"

for city in "${cities[@]}"; do
    echo  "Generating cross SAT-STV data for $city"
    python ./format/multi_SAT_cross_STV_llava.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating cross SAT-STV data"

for city in "${cities[@]}"; do
    echo  "GeneratinG STV_compare data for $city"
    python ./format/multi_STV_compare_llava.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating STV_compare data"