import os, sys
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
import torch
# Can do clip_utils, aes_utils, hps_utils
from utils.pickscore_utils import Selector
torch.set_grad_enabled(False)

# NAME = "tmp-sdxl-selr-noseed-2"
# INDEX = 128

NAME = sys.argv[1]
INDEX = int(sys.argv[2])
SAVE_IMG = sys.argv[3] == 'True'

print(f"Running {NAME}-{INDEX} with save_img={SAVE_IMG}")

sys.stdout = open(f"result/{NAME}-{INDEX}.txt", "w")

print("Loading models...")

dpo_unet = UNet2DConditionModel.from_pretrained(
                            # 'mhdang/dpo-sd1.5-text2image-v1',
                            # 'mhdang/dpo-sdxl-text2image-v1',
                            f'/share/imagereward_work/DiffusionDPO-v2/ckp/{NAME}/checkpoint-{INDEX}',
                            # alternatively use local ckptdir (*/checkpoint-n/)
                            subfolder='unet',
                            torch_dtype=torch.float16).to('cuda')

# pretrained_model_name = "CompVis/stable-diffusion-v1-4"
# pretrained_model_name = "runwayml/stable-diffusion-v1-5"
# pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
pretrained_model_name = "/share/Stable-Diffusion/stable-diffusion-xl-base-1.0"
gs = (5 if 'stable-diffusion-xl' in pretrained_model_name else 7.5)

if 'stable-diffusion-xl' in pretrained_model_name:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                   torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.safety_checker = None # Trigger-happy, blacks out >50% of "robot tiger"

# Score generations automatically w/ reward model
ps_selector = Selector('cuda')

print("Models loaded successfully!")

unets = [pipe.unet, dpo_unet]
names = ["Orig. SDXL", "DPO SDXL"]

def gen(prompt, seed=0, run_baseline=True):
    ims = []
    generator = torch.Generator(device='cuda')
    for unet_i in ([0, 1] if run_baseline else [1]):
        # print(f"Prompt: {prompt}\nSeed: {seed}\n{names[unet_i]}")
        pipe.unet = unets[unet_i]
        generator = generator.manual_seed(seed)
        
        im = pipe(prompt=prompt, generator=generator, guidance_scale=gs).images[0]
        ims.append(im)
    return ims

default_prompts = [
    "A pile of sand swirling in the wind forming the shape of a dancer",
    "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    "a smiling beautiful sorceress with long dark hair and closed eyes wearing a dark top surrounded by glowing fire sparks at night, magical light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "A purple raven flying over big sur, light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "a smiling beautiful sorceress wearing a modest high necked blue suit surrounded by swirling rainbow aurora, hyper-realistic, cinematic, post-production",
    "Anthro humanoid turtle skydiving wearing goggles, gopro footage",
    "A man in a suit surfing in a river",
    "photo of a zebra dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography",
    "A typhoon in a tea cup, digital render",
    "A cute puppy leading a session of the United Nations, newspaper photography",
    "Worm eye view of rocketship",
    "Glass spheres in the desert, refraction render",
    "anthropmorphic coffee bean drinking coffee",
    "A baby kangaroo in a trenchcoat",
    "A towering hurricane of rainbow colors towering over a city, cinematic digital art",
    "A redwood tree rising up out of the ocean",
]

def get_example_prompts(dataset_name='default'):
    if dataset_name == 'default':
        return default_prompts
    elif dataset_name.lower() == "drawbench":
        # load all prompts from csv file "prompt/data.csv"
        import pandas as pd
        df = pd.read_csv("prompt/data.csv")
        return df['Prompts'].tolist()
    return []

dataset_name = "drawbench"
img_saving_path = f"img/{dataset_name}/{NAME}-{INDEX}"
if not os.path.exists(img_saving_path):
    os.makedirs(img_saving_path)
example_prompts = get_example_prompts(dataset_name)

print("\nGenerating images and scoring them:")

orig_win = 0
dpo_win = 0
orig_score_sum = 0.0
dpo_score_sum = 0.0
for id, p in enumerate(example_prompts):
    ims = gen(p) # could save these if desired
    if SAVE_IMG:
        for i, im in enumerate(ims):
            im.save(os.path.join(img_saving_path, f"prompt_{id}_{names[i]}.png"))
    scores = ps_selector.score(ims, p)
    orig_win += scores[0] > scores[1]
    dpo_win += scores[1] > scores[0]
    orig_score_sum += scores[0]
    dpo_score_sum += scores[1]
    print(f"{id}: dpo win = {scores[1] > scores[0]}, ", "ori. win/ dpo win: ", orig_win, '/', dpo_win)

print("Orig. wins:", orig_win)
print("DPO wins:", dpo_win)
print("Orig. score mean:", orig_score_sum / len(example_prompts))
print("DPO score mean:", dpo_score_sum / len(example_prompts))

# print("\nPrompts:")

# # to get partiprompts captions
# from datasets import load_dataset
# dataset = load_dataset("./parti-prompts")
# print(dataset['train']['Prompt'])