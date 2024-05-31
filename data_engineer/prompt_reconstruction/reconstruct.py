import os, sys
from typing import Literal, List, Tuple
import argparse

import tqdm
import pandas as pd

from config import *
from utils import load_chatgpt3_6b_model


# create an argparse object to parse the command line arguments
def parse():
    parser = argparse.ArgumentParser(description='Reconstruct prompts with ChatGLM3 6B.')
    parser.add_argument('--split_category', type=str, 
                        default='basic_test', help='The category of the split to reconstruct prompts.')
    parser.add_argument('--max_length', type=int, default=None, help='The maximum number of prompts to reconstruct.')
    parser.add_argument('--start_index', type=int, default=0, help='The start index of the prompts to reconstruct.')
    parser.add_argument('--prompt_id', type=int, default=0, help='The id of the initialization prompt that is used for the reconstruction.')

    # parse the command line arguments
    args = parser.parse_args()
    return args


def load_dalle3_revised_prompts(
    path=DALLE3_REVISED_PROMPTS_ROOT, 
    split_category: Literal['basic_test', 'basic_train', 'basic_val'] = 'basic_test',
    max_length=None,
) -> Tuple[List[str], List[int]]:
    file_paths = []

    # find all txt file paths whose name is 'revised_prompt.txt' under the 'path' directory, where 'path' starts with split_category
    for root, dirs, files in tqdm.tqdm(os.walk(path)):
        if root.split('/')[-1].startswith(split_category):
            for file in files:
                if file == 'revised_prompt.txt':
                    file_paths.append(os.path.join(root, file))
    # sort the file paths
    file_paths.sort()

    if max_length:
        file_paths = file_paths[:max_length]

    # read the revised prompts from the file paths
    revised_prompts = []
    prompt_ids = []
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as f:
            revised_prompts.append(f.read())
        prompt_ids.append(file_path.split('/')[-2].split('_')[2])

    # print length of the revised prompts
    print(f"Length of the revised prompts: {len(revised_prompts)}\n", file=sys.stderr)

    return revised_prompts, prompt_ids


def reconstruction_with_chatglm3_6b(
    model=None, tokenizer=None, input_prompts=None, prompt_id=0, 
    split_category: Literal['basic_test', 'basic_train', 'basic_val'] = 'basic_test',
    max_length=None, output_file=None, 
):
    # initialization prompt
    initialization_prompt = PROMPT_MAPPING[prompt_id]
    prompt_pairs = {'id': [], 'original': [], 'reconstructed': []}
    
    if input_prompts is None:
        print(f"Loading the dalle3 revised prompts for {split_category}...", file=sys.stderr)
        prompt_pairs['original'], prompt_pairs['id'] = load_dalle3_revised_prompts(split_category=split_category, max_length=max_length)
        print(f"The revised prompts are loaded successfully.\n", file=sys.stderr)

    if model is None or tokenizer is None:
        print("Loading the ChatGLM3 6B model and tokenizer...", file=sys.stderr)
        model, tokenizer = load_chatgpt3_6b_model()
        print("The ChatGLM3 6B model and tokenizer are loaded successfully.\n", file=sys.stderr)

    # reconstruct the prompts with revised prompts and the initialization as the history
    print("\nReconstructing the prompts...\n", file=sys.stderr)

    error_count = 0
    error_log = []
    for i, prompt in enumerate(tqdm.tqdm(prompt_pairs['original'])):
        try:
            response, _ = model.chat(tokenizer, initialization_prompt + prompt, history=[])
            prompt_pairs['reconstructed'].append(response)
            print(f"Original prompt: \n --- {initialization_prompt + prompt}")
            print(f"Reconstructed prompt: \n >>> {response}\n")
            # flush the output buffer
            sys.stdout.flush()
        except Exception as e:
            print(f"An error occured. Skipping this prompt.", file=sys.stderr)
            prompt_pairs['reconstructed'].append("")
            error_count += 1
            error_log.append((prompt_pairs['id'][i], prompt, e))
        if max_length and i > max_length:
            break

    # save error log to a file
    with open(f'./output/{prompt_id}/dalle3_reconstruction_error_log-{split_category}.txt', 'w') as f:
        f.write(f"Error count: {error_count}\n")
        f.write(f"Error log:")
        for item in error_log:
            f.write(f"{item}\n")

    # save the reconstructed prompts to a csv file
    if not os.path.exists(f'./data/{prompt_id}'):
        os.makedirs(f'./data/{prompt_id}')
    output_file = f'./data/{prompt_id}/dalle3_revised_prompts_reconstructed_{split_category}.csv'

    if max_length:
        prompt_pairs = {key: value[:max_length] for key, value in prompt_pairs.items()}
    df = pd.DataFrame(prompt_pairs)
    df.to_csv(output_file, index=False)
    
    return prompt_pairs, error_count
    

if __name__ == "__main__":
    args = parse()

    if not os.path.exists(f'./output/{args.prompt_id}'):
        os.makedirs(f'./output/{args.prompt_id}')
    output_redirect = f'./output/{args.prompt_id}/dalle3_reconstruction_process_log-{args.split_category}.txt'
    sys.stdout = open(output_redirect, 'w')
    print(f"Stdout is redirected to the file: {output_redirect}\n", file=sys.stderr)
    print(f"Command line arguments: {args}\n", file=sys.stderr)

    reconstruction_with_chatglm3_6b(max_length=args.max_length, split_category=args.split_category, prompt_id=args.prompt_id)
    sys.stdout.close()
