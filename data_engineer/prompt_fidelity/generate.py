import os, sys

from diffusers import StableDiffusionXLPipeline
import torch
import tqdm
import pandas as pd

GENERATED_IMAGES_DIR = '/share/home/wusiyuan/image/generated_laion_images/'
SDXL_BASE_MODEL_PATH = "/share/Stable-Diffusion/stable-diffusion-xl-base-1.0"


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


def main(dataset_name="tmp", m=None):
    # create the directory to store the generated images
    dir_path = os.path.join(GENERATED_IMAGES_DIR, dataset_name)
    os.makedirs(dir_path, exist_ok=True)

    print(f"Generating images for dataset: {dataset_name}")
    print(f"Output directory: {dir_path}")

    # load prompts
    print("Loading prompts...")
    if dataset_name == 'laion_high_res':
        caption_parquet_path = f"/share/home/wusiyuan/imagereward_work/prompt_fidelity/caption/laion_high_res/output.parquet"
        pass
    elif dataset_name == 'laion115m':
        caption_parquet_path = f"/share/home/wusiyuan/imagereward_work/prompt_fidelity/caption/laion115m/output.parquet"
        pass
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    prompt_dict = {"prompt": [], "generated_image_path": [], "original_image_path": []}
    df = pd.read_parquet(caption_parquet_path)
    # print(df.head())
    # exit()
    if m is None:
        prompts = df["long_caption"].tolist()
        img_names = df["image_name"].tolist()
    else:
        prompts = df["long_caption"].tolist()[:m]
        img_names = df["image_name"].tolist()[:m]

    print(f"Number of prompts: {len(prompts)}"
            f"\nFirst prompt: {prompts[0]}"
            f"\nLast prompt: {prompts[-1]}")

    # load the sdxl model
    print("Loading SDXL model...")
    sdxl_model, generator = load_sdxl_base_model()

    print("Generating images...")
    # iterate through the prompts and generate images, I want to use tqdm to track the progress
    for i, prompt in tqdm.tqdm(enumerate(prompts)):
        path = os.path.join(dir_path, f"{i}.png")
        # # if image already exists, skip
        # if os.path.exists(path):
        #     continue

        prompt_dict["prompt"].append(prompt)
        prompt_dict["generated_image_path"].append(path)
        prompt_dict["original_image_path"].append(img_names[i])

        if pd.isna(prompt) or len(prompt) == 0:
            continue
        image = generate(prompt, sdxl_model, generator)
        image.save(path)

    # save prompt dict
    prompt_df = pd.DataFrame(prompt_dict)
    prompt_df.to_csv(os.path.join(dir_path, "prompts.csv"), index=False)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    dataset_name = sys.argv[1]
    main(dataset_name=dataset_name)
