# Original dataset paths which has the raw data
DALLE3_DATASET_PATH = '/share/img_datasets/dalle3_datas/dalle3_imgs.jsonl'
MIDJOURNEY_DATASET_ROOT = '/share/img_datasets/midjourney_4choose1_new_1778w/'
DALLE3_REVISED_PROMPTS_ROOT = '/share/imagereward_work/dalle3/'
DALLE3_ORIGINAL_IMAGES_ROOT = '/share/imagereward_work/dalle3/'

# Target file paths to store the original prompts
DALLE3_PROMPTS_TARGET_FILE = '/share/imagereward_work/prompt_reconstruction/data/dalle3_prompts_orig.txt'
MIDJOURNEY_PROMPTS_TARGET_FILE = '/share/imagereward_work/prompt_reconstruction/data/midjourney_prompts_orig.txt'
DALLE3_REVISED_PROMPTS_TARGET_DIR = '/share/imagereward_work/prompt_reconstruction/data/dalle3_revised_prompts_orig/'

# Directory to store the generated prompts
GENERATED_PROMPTS_DIR = '/share/imagereward_work/prompt_reconstruction/data/generated_dalle3_prompts/'
GENERATED_IMAGES_DIR = '/share/imagereward_work/prompt_reconstruction/data/generated_dalle3_images/'
ORIGINAL_IMAGES_DIR = '/share/imagereward_work/prompt_reconstruction/data/original_dalle3_images/'

# recaptioned prompts
DALLE3_PROMPT_BLIP_LARGE = '/share/imagereward_work/prompt_reconstruction/data/blip_large.csv'
DALLE3_PROMPT_BLIP2_OPT = '/share/imagereward_work/prompt_reconstruction/data/blip2_opt.csv'
DALLE3_PROMPT_BLIP2_FLAN = '/share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv'

# Directory to store created datasets for DPO training
DALLE3_ALIGNMENT_DATASET_DIR = '/share/imagereward_work/prompt_reconstruction/data/dalle3_alignment_dataset/'

# Model paths
SDXL_BASE_MODEL_PATH = "/share/Stable-Diffusion/stable-diffusion-xl-base-1.0"
CHATGLM3_6B_PRETRAINED_MODEL_PATH = '/share/official_pretrains/hf_home/chatglm3-6b/'

# Prompt mapping, the key is the prompt id and shows the directory under which the generated prompts are stored
# /share/imagereward_work/prompt_reconstruction/data/generated_dalle3_prompts/{id}/
PROMPT_MAPPING = {
    0: "Please rewrite the following sentence: ",
    1: "Remove at least two important elements or features of elements in the following sentence, and then output the modified sentence: ",
    2: "Insert at least new two elements or features of elements in the following sentence. Do not change other part of the sentence. Output the modified sentence only. \n",
    3: "Change at least two elements or features of elements in the following sentence. Make sure the meaning of the sentence changes. Output the modified sentence. \n",
    4: "Change at least two and at most four elements or features of elements in the following sentence. Make sure the meaning of the sentence changes and output the modified sentence. \n",
    5: "Shorten the following sentence with minimal modification in meaning, and then output the modified sentence: ",
    6: "Extend the following sentence to a longer sentence: ",

    #!!! Add new mappings only after previous mappings to avoid changing the existing prompts and their ids
}