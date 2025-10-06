cities=("Beijing" "NewYork" "London")
work_dir="../../../data/"

for city in "${cities[@]}"; do
    echo "Generating sat_count CoT with template for $city"
    python ./advance/CoT/sat_address_cot/gen_CoT_template.py --city $city --work_dir $work_dir
    echo "Using GPT to polish CoT for $city"
    python ./advance/CoT/sat_address_cot/gpt_polish.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating CoT for sat_address"

for city in "${cities[@]}"; do
    echo "Generating sat_count CoT with template for $city"
    python ./advance/CoT/sat_count_cot/gen_CoT_template.py --city $city --work_dir $work_dir
    echo "Using GPT to polish CoT for $city"
    python ./advance/CoT/sat_count_cot/gpt_polish.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating CoT for sat_count"

for city in "${cities[@]}"; do
    echo "Generating stv_address CoT with template for $city"
    python ./advance/CoT/stv_address_cot/gen_CoT_template.py --city $city --work_dir $work_dir
    echo "Using GPT to polish CoT for $city"
    python ./advance/CoT/stv_address_cot/gpt_polish.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating CoT for stv_address"

for city in "${cities[@]}"; do
    echo "Generating sat_cross_stv CoT with template for $city"
    python ./advance/CoT/sat_cross_stv_cot/gen_CoT_template.py --city $city --work_dir $work_dir
    echo "Using GPT to polish CoT for $city"
    python ./advance/CoT/sat_cross_stv_cot/gpt_polish.py --city $city --work_dir $work_dir
done
wait
echo "Finish generating CoT for sat_cross_stv"