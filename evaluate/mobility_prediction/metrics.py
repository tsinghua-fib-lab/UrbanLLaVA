import os
import pandas as pd
from sklearn.metrics import f1_score
import ast
import csv
import numpy as np



def get_acc1_f1(df):
    acc1 = (df['prediction'] == df['ground_truth']).sum() / len(df)
    f1 = f1_score(df['ground_truth'], df['prediction'], average='weighted')
    return acc1, f1

def get_is_correct(row):
    pred_list = row['prediction']
    if row['ground_truth'] in pred_list:
        row['is_correct'] = True
    else:
        row['is_correct'] = False
    
    return row


def get_is_correct10(row):
    pred_list = row['top10']
    if row['ground_truth'] in pred_list:
        row['is_correct10'] = True
    else:
        row['is_correct10'] = False
        
    pred_list = row['top5']
    if row['ground_truth'] in pred_list:
        row['is_correct5'] = True
    else:
        row['is_correct5'] = False

    pred = row['top1']
    if pred == row['ground_truth']:
        row['is_correct1'] = True
    else:
        row['is_correct1'] = False
    
    return row


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_ndcg(prediction, targets, k=10):
    """
    Calculates the NDCG score for the given predictions and targets.

    Args:
        prediction (Nxk): list of lists. the softmax output of the model.
        targets (N): torch.LongTensor. actual target place id.

    Returns:
        the sum ndcg score
    """
    for _, xi in enumerate(prediction):
        if len(xi) < k:
            #print(f"the {i}th length: {len(xi)}")
            xi += [-5 for _ in range(k-len(xi))]
        elif len(xi) > k:
            xi = xi[:k]
        else:
            pass
    
    n_sample = len(prediction)
    prediction = np.array(prediction)
    targets = np.broadcast_to(targets.reshape(-1, 1), prediction.shape)
    hits = first_nonzero(prediction == targets, axis=1, invalid_val=-1)
    hits = hits[hits>=0]
    ranks = hits + 1
    ndcg = 1 / np.log2(ranks + 1)
    return np.sum(ndcg) / n_sample


def cal_metrics(output_dir):
     # Calculate the metric for all user
    # output_dir = 'output/Mixtral-8x22B-Instruct-v0.1_Paris_top1_wot'
    file_list = [file for file in os.listdir(output_dir) if file.endswith('.csv')]
    print(file_list)
    file_path_list = [os.path.join(output_dir, file) for file in file_list]

    df = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    for file_path in file_path_list:
        iter_df = pd.read_csv(file_path)
        df = pd.concat([df, iter_df], ignore_index=True)
        
    df_cleaned = df.dropna(subset=['prediction', 'ground_truth'])
    df_cleaned['prediction'] = df_cleaned['prediction'].apply(
        # lambda x: int(x) if isinstance(x, int) else int(x.split('or')[0]) if 'or' in x else int(x)
        lambda x: int(x) if isinstance(x, int) else -100
    )
    df_cleaned['ground_truth'] = df_cleaned['ground_truth'].apply(lambda x: int(x))

    acc1, f1 = get_acc1_f1(df_cleaned)
    return acc1, f1


if __name__ == "__main__":
    # Calculate the metric for all user
    # TODO: Fill in the model names that you want to evaluate
    models = ["Llama-3-VILA1.5-8b",]  
    cities = ["Beijing"]

    csv_file = 'results/prediction_results/metrics.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'City', 'Acc@1', 'F1'])
        for model in models:
            for city in cities:
                output_dir = f'results/prediction_results/{model}_{city}_top1_wot'
                acc1, f1 = cal_metrics(output_dir=output_dir)
                writer.writerow([model, city, acc1, f1])

    print(f"Results have been saved to {csv_file}")
