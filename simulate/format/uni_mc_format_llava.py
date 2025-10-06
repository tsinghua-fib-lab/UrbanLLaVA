import os
import json
import random
import argparse
random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--work_dir', type=str, default='../../data/')

    args = parser.parse_args()
    city_name = args.city_name
    work_dir = args.work_dir

    cur_dir = os.path.join(work_dir, f"dev-{city_name}/")

    output_dir = os.path.join(cur_dir, "llava_uniimage_mc_train")
    os.makedirs(output_dir, exist_ok=True)

    task_list = ['sat_address_mc', 'sat_landuse_mc', 'stv_address_mc', 'stv_landmark_mc']
    summary_info = {}

    for task_name in task_list:
        unformatted_file_path = os.path.join(cur_dir, "uni_image_data", task_name, city_name, f"{city_name}_{task_name}_train.json")

        with open(unformatted_file_path, 'r') as f:
            unformatted_data = json.load(f)

        if len(unformatted_data) > 20000:
            print(f"Length of {task_name} is {len(unformatted_data)}, truncated to 20000")
            unformatted_data = random.sample(unformatted_data, 20000)

        formatted_data = []
        for item in unformatted_data:
            formatted_item = {}
            image_name = item['image'].split('/')[-1]
            formatted_item['id'] = image_name
            formatted_item['image'] = item['image']
            formatted_item['conversations'] = [
                {
                    "from": "human",
                    'value': item['prompt']
                },
                {
                    "from": "gpt",
                    "value": item['reference']
                }
            ]

            formatted_data.append(formatted_item)

        output_file_path = os.path.join(output_dir, f"{city_name}_{task_name}_train_llava.json")
        with open(output_file_path, 'w') as f:
            json.dump(formatted_data, f, indent=4, ensure_ascii=False)
            print(f"Formatted data for {task_name} saved to {output_file_path}")
        summary_info[task_name] = len(formatted_data)

    summary_file_path = os.path.join(output_dir, "summary.json")
    with open(summary_file_path, 'w') as f:
        json.dump(summary_info, f, indent=4, ensure_ascii=False)
        print(f"Summary info saved to {summary_file_path}")