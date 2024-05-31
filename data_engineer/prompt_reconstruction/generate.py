import argparse
import shutil
import os
import sys
import re

from diffusers import StableDiffusionXLPipeline
import torch
import tqdm
import pandas as pd

from utils import get_image_size
from config import *


def generate(prompt: str, model: StableDiffusionXLPipeline, generator: torch.Generator, gs = 5):
    return model(prompt=prompt, generator=generator, guidance_scale=gs).images[0]


def load_sdxl_base_model(pretrained_model_name: str = SDXL_BASE_MODEL_PATH, seed=None):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.safety_checker = None

    generator = torch.Generator(device='cuda')
    if seed is not None:
        generator.manual_seed(seed)

    return pipe, generator


def parse():
    parser = argparse.ArgumentParser(description="Generate images from prompts")
    parser.add_argument("--dir_path", type=str, default=GENERATED_PROMPTS_DIR,
                        help="Path to the directory where the prompts are located.")
    parser.add_argument("--dir_id", type=int, required=True, help="Directory id where the prompts are located.")
    parser.add_argument("--split", type=str, required=True, default="train", help="The split of the prompts.")
    parser.add_argument("--start_id", type=int, default=0, 
                        help="The start prompt id to generate images for.")
    parser.add_argument("--max_prompts", type=int, default=100000, 
                        help="The maximum number of prompts to generate images for. Start from 'start_id'")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="Debug info. If not None, output would be redirected to this file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the generator.")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    if args.output_path is not None:
        # make sure the path exists and redirect the output to the file
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        sys.stdout = open(args.output_path, 'w')
    
    # create the directory to store the generated images
    os.makedirs(os.path.join(GENERATED_IMAGES_DIR, str(args.dir_id)), exist_ok=True)

    # load the sdxl model
    sdxl_model, generator = load_sdxl_base_model()

    # search under prompt_csv_dir for the csv file with 'split' and 'filter' in the name
    prompt_csv_dir = os.path.join(args.dir_path, str(args.dir_id))
    prompt_csv_file = None
    for file in os.listdir(prompt_csv_dir):
        if args.split in file and "filter" in file:
            prompt_csv_file = file
            break
    assert prompt_csv_file is not None, "No filtered prompt csv file found for the given split."

    # load data from the csv file, only load the prompts from start_id to start_id + max_prompts
    prompts = pd.read_csv(os.path.join(prompt_csv_dir, prompt_csv_file))
    prompts = prompts.iloc[args.start_id:args.start_id + args.max_prompts]

    print(f"Loaded {len(prompts)} prompts from {prompt_csv_file} for split {args.split} under dir {args.dir_id}.")
    print(f"Generating images for prompts from {args.start_id} to {args.start_id + args.max_prompts}.")

    # iterate through the prompts and generate images, I want to use tqdm to track the progress
    for i in tqdm.tqdm(range(len(prompts))):
        # if image already exists, skip
        prompt = prompts.iloc[i]
        prompt_id = prompt["id"]
        path = os.path.join(GENERATED_IMAGES_DIR, str(args.dir_id), f"{args.split}-{prompt_id}.png")
        if os.path.exists(path):
            print(f"Image for prompt {prompt_id} already exists, skip.")
            continue
        
        prompt_text = prompt["reconstructed"]
        # if prompt is empty, skip
        if pd.isna(prompt_text) or len(prompt_text) == 0:
            continue 
        image = generate(prompt_text, sdxl_model, generator)
        image.save(path)


def grab_original_dalle3_images(src_dir=DALLE3_ORIGINAL_IMAGES_ROOT, tgt_dir=ORIGINAL_IMAGES_DIR, remove_existing=True):
    # remove png files in the target directory
    if remove_existing:
        for f in os.listdir(tgt_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(tgt_dir, f))
        print(f"Removed existing images in {tgt_dir}.")
    
    # create the target directory if it does not exist
    os.makedirs(tgt_dir, exist_ok=True)

    # copy the images from the source directory to the target directory
    for root, d, file in tqdm.tqdm(os.walk(src_dir)):
        for f in file:
            if f.endswith(".png"):
                prompt_id = root.split('_')[3]
                split = root.split('_')[2]
                assert re.match(r'^\d+$', prompt_id), f"Prompt id '{prompt_id}' is not a number."
                assert split in ['train', 'val', 'test'], f"Split '{split}' is not valid."
                target_file_path = os.path.join(tgt_dir, f'{split}-{prompt_id}.png')
                shutil.copy(os.path.join(root, f), target_file_path)


def create_dataset(
    original_prompt_dir=DALLE3_REVISED_PROMPTS_ROOT,
    original_image_dir=ORIGINAL_IMAGES_DIR,
    generated_image_dir=GENERATED_IMAGES_DIR,
    target_dataset_file=DALLE3_ALIGNMENT_DATASET_DIR
):
    pass


if __name__ == "__main__":
    # torch.set_grad_enabled(False)
    # main()

    # grab_original_dalle3_images()
    pass