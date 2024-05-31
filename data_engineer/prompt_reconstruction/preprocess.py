from config import *
from typing import List, Dict

import json
import pandas as pd
import os
import tqdm


def read_dalle3_dataset(dataset_path=DALLE3_DATASET_PATH) -> pd.DataFrame:
    with open(dataset_path, 'r') as f:
        data = f.readlines()
    data = [json.loads(x) for x in data]
    data = pd.DataFrame(data)
    return data

def read_midjourney_dataset(dataset_root=MIDJOURNEY_DATASET_ROOT, file_index=None) -> pd.DataFrame:
    # Find all files under the 'dataset_root' directory and keep only the .jsonl files
    files = [os.path.join(dp, f) for dp, _, filenames in os.walk(dataset_root) for f in filenames if f.endswith('.jsonl')]
    files.sort()

    # If file_index is not None, keep only the file whose filename ends with '0..0{file_index}.meta.jsonl', 
    # where '0..0' is a sequence of 0s which makes the file index length 5
    if file_index is not None:
        file_index = str(file_index).zfill(5)
        files = [f for f in files if f.endswith(f'{file_index}.meta.jsonl')]
    
    # Read all the .jsonl files into a single dataframe
    data = []
    for file in tqdm.tqdm(files):
        with open(file, 'r') as f:
            data.extend(f.readlines())
    data = [json.loads(x) for x in tqdm.tqdm(data)]
    data = pd.DataFrame(data)
    return data


def get_dalle3_revised_prompts(path=DALLE3_REVISED_PROMPTS_ROOT) -> List[str]:    
    revised_prompts = {'basic_test': [], 'basic_train': [], 'basic_val': []}
    for root, dirs, files in tqdm.tqdm(os.walk(path)):
        # check if the directory name starts with basic_test, basic_train or basic_val
        if root.split('/')[-1].startswith('basic_test'):
            category = 'basic_test'
        elif root.split('/')[-1].startswith('basic_train'):
            category = 'basic_train'
        elif root.split('/')[-1].startswith('basic_val'):
            category = 'basic_val'
        else:
            continue
        # append the revised prompt to the corresponding list
        for file in files:
            if file == 'revised_prompt.txt':
                with open(os.path.join(root, file), 'r') as f:
                    revised_prompts[category].append(f.read())
        
    return revised_prompts


def load_and_get_prompts_dalle3():
    print("\n\nLoading DALLE3 dataset...")
    data = read_dalle3_dataset()
    print("Size of the DALLE3 dataset: ", data.shape[0])
    prompts = data['caption'].unique()
    print("Number of distinct prompts in the DALLE3 dataset: ", len(prompts))
    print("\nSaving prompts to files...")
    with open(DALLE3_PROMPTS_TARGET_FILE, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    print(f"Prompts from DALLE3 dataset are saved to {DALLE3_PROMPTS_TARGET_FILE}\n")
    return prompts


def load_and_get_prompts_midjourney(file_index=None):
    print("\n\nLoading MidJourney dataset...")
    data = read_midjourney_dataset(file_index=file_index)
    print("Size of the MidJourney dataset: ", data.shape[0])
    prompts = data['image_prompt'].unique()
    print("Number of distinct prompts in the MidJourney dataset: ", len(prompts))
    print("\nSaving prompts to files...")
    with open(MIDJOURNEY_PROMPTS_TARGET_FILE, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    print(f"Prompts from MidJourney dataset are saved to {MIDJOURNEY_PROMPTS_TARGET_FILE}\n")    
    return prompts


def load_and_get_prompts_dalle3_revised():
    print("\n\nLoading DALLE3 revised prompts...")
    revised_prompts : Dict[str, List[str]] = get_dalle3_revised_prompts()
    print("The revised prompts are loaded successfully.\n")
    print("Number of revised prompts for basic_test: ", len(revised_prompts['basic_test']))
    print("Number of revised prompts for basic_train: ", len(revised_prompts['basic_train']))
    print("Number of revised prompts for basic_val: ", len(revised_prompts['basic_val']))
    print("\nSaving revised prompts to files...")
    
    if not os.path.exists(DALLE3_REVISED_PROMPTS_TARGET_DIR):
        os.makedirs(DALLE3_REVISED_PROMPTS_TARGET_DIR)
    # save to separate files for each category
    for category, prompts in revised_prompts.items():
        with open(os.path.join(DALLE3_REVISED_PROMPTS_TARGET_DIR, f'{category}_revised_prompts.txt'), 'w') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
    print(f"Revised prompts from DALLE3 dataset are saved to {DALLE3_REVISED_PROMPTS_TARGET_DIR}\n")
    return revised_prompts


if __name__ == "__main__":
    pass
