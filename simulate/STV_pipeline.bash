cities=("Beijing" "NewYork" "London")
work_dir="../../data/"

# Depend on /ThreeCityImage/{city}/StreetView/
# Depend on SAT_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Finding corresponding streetview images for $city"
    python ./streetview/spatial_join.py --city $city --work_dir $work_dir
done
wait
echo "Finish finding corresponding streetview images"
# Get stv_in_sat_{city}_{zl}.csv
# Get sampled_stv_images/

# Depend on stv_in_sat_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Querying address for streetview images for $city"
    python ./streetview/osm_address_web_stv_my.py --city $city --work_dir $work_dir
done
wait
echo "Finish querying address for streetview images"
# Get stv_in_sat_address_deploy_{zl}.csv

# Depend on stv_in_sat_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting nearest POI for streetview images for $city"
    python ./streetview/stv_nearest_pois.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting nearest POI for streetview images"
# Get stv_in_sat_nearest_features_{city}_{zl}.csv

# Depend on stv_in_sat_nearest_features_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting 10 POI for streetview images for $city"
    python ./streetview/process_stv_near.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting 10 POI for streetview images"
# Get stv_in_sat_nearest_features_updated_{city}_{zl}.csv