echo "Run data curation pipeline"

bash ./SAT_pipeline.bash
echo "Finish SAT pipeline"

bash ./STV_pipeline.bash
echo "Finish STV pipeline"

bash ./address.bash
echo "Finish address querying"

bash ./annotate.bash
echo "Finish annotation"

echo "Finish data preparation, start uni_image_basic_construct"

bash ./uni_image_basic_construct.bash
echo "Finish uni_image_basic_construct"

echo "Finish uni_image_basic_construct, start uni_image_mc_construct"

bash ./uni_image_mc_construct.bash
echo "Finish uni_image_mc_construct"

echo "Finish uni_image_mc_construct, start multi_image_mc_construct"

bash ./multi_image_mc_construct.bash
echo "Finish multi_image_mc_construct"

echo "Finish multi_image_mc_construct, start CoT_construct"

bash ./CoT_construct.bash
echo "Finish CoT_construct"

echo "Finish CoT_construct"
echo "Finish data curation pipeline"