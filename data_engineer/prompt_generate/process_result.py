from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

index_list = ['227899050', '232215894', '2078102203']

csv_file_list = {
    '227899050': '/share/home/wusiyuan/imagereward_work/prompt_generate/result/metrics_227899050.csv',
    '232215894': '/share/home/wusiyuan/imagereward_work/prompt_generate/result/metrics_232215894.csv',
    '2078102203': '/share/home/wusiyuan/imagereward_work/prompt_generate/result/metrics_2078102203.csv',   
}


map_score = {
    "No": 3,
    "Uncertain": 2,
    "Yes": 1,
}

PROMPT_CSV = '/share/home/wusiyuan/imagereward_work/prompt_generate/Stable-Diffusion-Prompts/data/train.csv'


def check_better(scores: Dict[str, List[int]], i: int, j: int) -> Tuple[int, int]:
    score_i = scores[index_list[i]]
    score_j = scores[index_list[j]]
    
    if score_i == score_j:
        return -1, -1
    elif all([score_i[k] >= score_j[k] for k in range(3)]):
        priority = sum([score_i[k] - score_j[k] for k in range(3)])
        return i, priority
    elif all([score_j[k] >= score_i[k] for k in range(3)]):
        priority = sum([score_j[k] - score_i[k] for k in range(3)])
        return j, priority
    else:
        return -1, -1


def process_csv():
    # read csv from csv_file_list
    dfs = {k: pd.read_csv(v) for k, v in csv_file_list.items()}
    
    # assert all df in dfs have the same length
    assert len(set([len(df) for df in dfs.values()])) == 1, "All df should have the same length"
    len_df = [len(df) for df in dfs.values()][0]
    
    # result_df = pd.DataFrame(columns=['image_path_winner', 'image_path_loser', 'priority'])
    result_dict = {"image_path_winner": [], "image_path_loser": [], "priority": []}
    
    # iterate over each row
    for i in tqdm(range(len_df)):
        # get the row from each df
        rows = {k: df.iloc[i] for k, df in dfs.items()}
        
        # each row refers to image_path,a,b,c
        # a,b,c are ['No', 'Uncertain', 'Yes']
        # convert a,b,c to score using map_score
        try:
            scores = {k: [map_score[row[k]] for k in ['a', 'b', 'c']] for k, row in rows.items()}
        except KeyError:
            print(f"KeyError at row {i}")
            continue
        
        # for each pair in scores, check which one is better
        pair = None
        highest_priority = -1
        for i in range(3):
            for j in range(i+1, 3):
                winner_id, priority = check_better(scores, i, j)
                assert winner_id in [-1, i, j], "winner_id should be -1, i, or j"
                
                if winner_id != -1 and priority > highest_priority:
                    highest_priority = priority
                    pair = (index_list[winner_id], index_list[i + j - winner_id])

        # append to result_dict if a pos-neg pair is found
        if pair is not None:
            result_dict["image_path_winner"].append(rows[pair[0]]['image_path'])
            result_dict["image_path_loser"].append(rows[pair[1]]['image_path'])
            result_dict["priority"].append(highest_priority)
    
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv('/share/home/wusiyuan/imagereward_work/prompt_generate/result/priority_pair.csv', index=False)


def analysis():
    df = pd.read_csv('/share/home/wusiyuan/imagereward_work/prompt_generate/result/priority_pair.csv')
    # print value counts of priority and sort by index
    print(df['priority'].value_counts().sort_index())
    print(df['priority'].mean())
    print(df['priority'].median())
    
    
def keep_only(priority_threshold: int):
    df = pd.read_csv('/share/home/wusiyuan/imagereward_work/prompt_generate/result/priority_pair.csv')
    df = df[df['priority'] >= priority_threshold]
    df.to_csv(f'/share/home/wusiyuan/imagereward_work/prompt_generate/result/priority_pair_over_{priority_threshold}.csv', index=False)


def append_prompt(priority_csv: str, prompt_csv: str, result_csv: str):
    priority_df = pd.read_csv(priority_csv)
    prompt_df = pd.read_csv(prompt_csv)
    # get index from priority_df
    # priority_df has three columns: image_path_winner,image_path_loser,priority
    # image_path_winner is like "/share/home/wusiyuan/imagereward_work/prompt_generate/Stable-Diffusion-Prompts/generated_images_sdxl-2078102203/000003.png"abs
    # get index from file name, and turn into int
    priority_df['index'] = priority_df['image_path_winner'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))
    prompt_df['index'] = range(len(prompt_df))
    
    # inner merge prompt_df and priority_df on index
    merged_df = pd.merge(prompt_df, priority_df, on='index')
    
    # save onlly image_path_winner, image_path_loser, prompt, and priority
    merged_df = merged_df[['image_path_winner', 'image_path_loser', 'Prompt', 'priority']]
    
    # save to csv
    merged_df.to_csv(result_csv, index=False)


if __name__ == "__main__":
    pass
    # process_csv()
    # analysis()
    # keep_only(4)
    
    # priority_csv = '/share/home/wusiyuan/imagereward_work/prompt_generate/result/priority_pair_over_4.csv'
    # result_csv = '/share/home/wusiyuan/imagereward_work/prompt_generate/result/priority_pair_over_4_prompt.csv'
    # append_prompt(priority_csv, PROMPT_CSV, result_csv)
    