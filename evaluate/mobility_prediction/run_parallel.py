import argparse
from multiprocessing import Pool

from .llm_mob import main


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--user_cnt', type=int, default=50)
    parser.add_argument('--traj_cnt', type=int, default=10)

    args = parser.parse_args()
    user_cnt = args.user_cnt            # users 
    sample_single_user = args.traj_cnt  # trajectory for each user
    data_version="mini"
    split_path="citydata/mobility/checkin_split/"
    test_path="citydata/mobility/checkin_test_pk/"
    
    models = ["gpt4omini"]
    # models = [
    #     "gpt-3.5", "gpt-4", "meta-llama/Meta-Llama-3-70B-Instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2",
    #     "meta-llama/Meta-Llama-3-8B-Instruct", "deepseek-chat"
    # ]
    cities = [
            "Beijing", "Cape", "London", "Moscow", "Mumbai", "Nairobi", "NewYork" ,"Paris" ,"San", "Sao", "Shanghai", "Sydney","Tokyo"
        ]
    
    # main(city, model, user_cnt=50, sample_single_user=10, num_historical_stay=40, num_context_stay=5, split_path="./checkin_split/", test_path="./checkin_test_pk/", data_version="all")
    para_group = []
    for c in cities:
        for m in models:
            para_group.append([c, m, user_cnt, sample_single_user, 40, 5, split_path, test_path, data_version])

    with Pool(6) as pool:
        results = pool.starmap(main, para_group)