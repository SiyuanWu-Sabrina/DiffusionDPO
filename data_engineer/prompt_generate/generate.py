import argparse
import shutil
import os
import sys
import re

from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import torch
import tqdm
import pandas as pd

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


def load_sd15_model(pretrained_model_name: str = SD15_MODEL_PATH, seed=None):
    pipe = DiffusionPipeline.from_pretrained(
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
    parser.add_argument("--start_id", type=int, default=0, 
                        help="The start prompt id to generate images for.")
    parser.add_argument("--max_prompts", type=int, default=10000, 
                        help="The maximum number of prompts to generate images for. Start from 'start_id'")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="Debug info. If not None, output would be redirected to this file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the generator.")
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "sd15"],
                        help="The model to use for image generation. Options: sdxl, sd15.")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    if args.output_path is not None:
        # make sure the path exists and redirect the output to the file
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        sys.stdout = open(args.output_path, 'w')

    # load the sdxl model
    sd_model, generator = load_sdxl_base_model(seed=args.seed) if args.model == "sdxl" else load_sd15_model(seed=args.seed)

    # load data from the csv file, only load the prompts from start_id to start_id + max_prompts
    prompts = pd.read_csv(Stable_Diffusion_CSV_Train_Path)
    prompts = prompts.iloc[args.start_id:args.start_id + args.max_prompts]
    prompts = prompts['Prompt'].tolist()

    print(f"Loaded {len(prompts)} prompts from {Stable_Diffusion_CSV_Train_Path}.")
    print(f"Generating images for prompts from {args.start_id} to {args.start_id + args.max_prompts - 1}.")

    # create the directory to store the generated images
    generated_img_dir = Generated_SDXL_Image_Dir if args.model == "sdxl" else Generated_SD15_Image_Dir
    generated_img_dir += "-" + str(args.seed) if args.seed is not None else ""
    os.makedirs(os.path.join(generated_img_dir), exist_ok=True)

    # iterate through the prompts and generate images, I want to use tqdm to track the progress
    for i in tqdm.tqdm(range(len(prompts))):
        # if image already exists, skip
        prompt_text = prompts[i]
        prompt_id = i + args.start_id

        prompt_id_fill = str(prompt_id).zfill(6)
        path = os.path.join(generated_img_dir, f"{prompt_id_fill}.png")
        if os.path.exists(path):
            print(f"Image for prompt {prompt_id} already exists, skip.")
            continue

        # if prompt is empty, skip
        if pd.isna(prompt_text) or len(prompt_text) == 0:
            print(f"Prompt {prompt_id} is empty, skip.")
            continue

        image = generate(prompt_text, sd_model, generator)
        image.save(path)
        print(f"Generated image for prompt {prompt_id}.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

    pass