cities=("Beijing" "NewYork" "London")
work_dir="../../data/"

# Depend on SAT_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Interpolating SAT, 25 for zl15, 9 for zl17, query address for $city"
    python ./address/interpolate_sat_coord.py --city $city --work_dir $work_dir
done
wait
echo "Finish interpolating SAT, 25 for zl15, 9 for zl17, query address"
# Get SAT_interpolate_{city}_{zl}.csv

# Depend on SAT_interpolate_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Getting Interpolated SAT's address for $city"
    python ./address/osm_address_web_my.py --city $city --work_dir $work_dir
done
wait
echo "Finish getting Interpolated SAT's address"
# Get SAT_interpolate_address_{city}_{zl}.csv