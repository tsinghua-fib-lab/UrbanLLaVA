cities=("Beijing" "NewYork" "London")
work_dir="../../data/"


for city in "${cities[@]}"; do
    echo "Making sat_addr task data for $city"
    python ./format/uni_mc_SAT_addr.py --city $city --work_dir $work_dir
done
wait
echo "Finish making sat_addr task data"

for city in "${cities[@]}"; do
    echo "Making sat_landuse task data for $city"
    python ./format/uni_mc_SAT_landuse.py --city $city --work_dir $work_dir
done
wait
echo "Finish making sat_landuse task data"

for city in "${cities[@]}"; do
    echo "Making stv_addr task data for $city"
    python ./format/uni_mc_STV_addr.py --city $city --work_dir $work_dir
done
wait
echo "Finish making stv_addr task data"

for city in "${cities[@]}"; do
    echo "Making stv_landmark task data for $city"
    python ./format/uni_mc_STV_landmark.py --city $city --work_dir $work_dir
done
wait
echo "Finish making stv_landmark task data"