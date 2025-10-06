cities=("Beijing" "NewYork" "London")
work_dir="../../data/"

# Remove the work_dir if it exists
# rm -rf $work_dir

# Depend on /ThreeCityImage/{city}/Sat_{zl}/
for city in "${cities[@]}"; do
    echo "Filtering images for $city"
    python ./satelite/make_image_list_sat.py --city $city --work_dir $work_dir
done
wait
echo "Finish filtering images"
# Get /sample_sat_image_{zl}/
# Get SAT_{city}_{zl}.csv

# Depend on SAT_{city}_{zl}.csv
for city in "${cities[@]}"; do
    echo "Creating shp file for $city"
    python ./satelite/make_sat_shp.py --city $city --work_dir $work_dir
done
wait
echo "Finish creating shp file"
# Get SAT_{city}_{zl}.shp, SAT_{city}_{zl}.dbf, SAT_{city}_{zl}.shx, SAT_{city}_{zl}.prj

# Depend on SAT_{city}_{zl}.shp, /ThreeCityImage/city_geojson_three_cities/{city}_{typ}.geojson
for city in "${cities[@]}"; do
    echo "Clip shp point data for $city"
    python ./satelite/clip_shp_point.py --city $city --work_dir $work_dir
done
wait
echo "Finish clipping shp point data"
# Get clipped_results_{zl}/clipped_{typ}_{polygon['region_nam'].split('.')[0]}.geojson

# Depend on SAT_{city}_{zl}.csv
# Depend on clipped_results_{zl}/clipped_{typ}_{img_name}.geojson
for city in "${cities[@]}"; do
    echo "Change OSM's lat and lon to SAT's pixel for $city"
    python ./satelite/coord_to_pixel.py --city $city --work_dir $work_dir
done
wait
echo "Finish changing OSM's lat and lon to SAT's pixel"
# Get clipped_results_{zl}_updated/clipped_{typ}_{img_name}_updated.geojson
# Get clipped_results_pixel_{zl}/clipped_{typ}_{img_name}_pixel.geojson

# Depend on clipped_results_pixel_{zl}/clipped_{typ}_{img_name}_pixel.geojson
for city in "${cities[@]}"; do
    echo "Filter out none valid data for $city"
    python ./satelite/extract_non_null_values.py --city $city --work_dir $work_dir
done
wait
echo "Finish filtering out none valid data"
# Get clipped_results_pixel_non_null_{zl}/{typ}_{img_name}.txt

# Depend on SAT_{city}_{zl}.csv
# Depend on clipped_results_pixel_non_null_{zl}/driving_{img_name}.txt
for city in "${cities[@]}"; do
    echo "Process driving data for $city"
    python ./satelite/process_driving.py --city $city --work_dir $work_dir
done
wait
echo "Finish processing driving data"
# Get "short_clipped_results_{zl}/driving_{img_name}.txt"

# Depend on SAT_{city}_{zl}.csv
# Depend on clipped_results_pixel_non_null_{zl}/landuse_{img_name}.txt
for city in "${cities[@]}"; do
    echo "Process landuse data for $city"
    python ./satelite/process_landuse.py --city $city --work_dir $work_dir
done
wait
echo "Finish processing landuse data"
# Get "short_clipped_results_{zl}/landuse_{img_name}.txt"

# Depend on SAT_{city}_{zl}.csv
# Depend on clipped_results_pixel_non_null_{zl}/poi_{img_name}.txt
for city in "${cities[@]}"; do
    echo "Process POI data for $city"
    python ./satelite/process_poi.py --city $city --work_dir $work_dir
done
wait
echo "Finish processing POI data"
# Get "short_clipped_results_{zl}/poi_{img_name}.txt"