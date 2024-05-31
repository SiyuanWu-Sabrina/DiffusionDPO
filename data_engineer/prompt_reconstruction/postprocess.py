import sys, os
import re
import pandas as pd
import tqdm
from PIL import Image

from utils import load_chatgpt3_6b_model, text_similarity
from config import *

# get data from the csv file, the file has three columns: id,original,reconstructed
# store the data into a dataframe
# then for each row, get the reconstructed prompt and see if it contains various lines separated by '\n'
# if it does, then split the prompt and the print splitted prompts
# for each such output, print the id, original and the splitted prompt


def filter_reconstructed_prompts(file_path, model, tokenizer):
    # check if file_path exists
    if not os.path.exists(file_path):
        return
    
    df = pd.read_csv(file_path)
    for i, row in tqdm.tqdm(df.iterrows()):
        reconstructed = row['reconstructed']
        list_of_reconstructed_prompts = []
        if '\n' in reconstructed:
            list_of_reconstructed_prompts = reconstructed.split('\n')
        else:
            continue

        # use the model and the tokenizer to check which sentence in the list_of_reconstructed_prompts
        # is the most similar to the original prompt
        original = row['original']
        max_similarity = 0
        most_similar_prompt = ""
        for prompt in list_of_reconstructed_prompts:
            similarity = text_similarity(original, prompt, model, tokenizer)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_prompt = prompt
        df.iloc[i, df.columns.get_loc('reconstructed')] = most_similar_prompt

    return df


def rename_files():
    for subdir in range(0, 7):
        root_dir = os.path.join(GENERATED_IMAGES_DIR, str(subdir))
        for i, filename in tqdm.tqdm(enumerate(os.listdir(root_dir))):
            number = re.search(r'\d+', filename).group()
            index = str(int(number)).zfill(4)
            split = filename.split("-")[0]
            os.rename(os.path.join(root_dir, filename), os.path.join(root_dir, f"{split}-{index}.png"))


def png2jpg():
    for subdir in range(0, 7):
        root_dir = os.path.join(GENERATED_IMAGES_DIR, str(subdir))
        for i, filename in tqdm.tqdm(enumerate(os.listdir(root_dir))):
            if filename.endswith(".png"):
                image = Image.open(os.path.join(root_dir, filename))
                image = image.convert("RGB")
                image.save(os.path.join(root_dir, filename.replace(".png", ".jpg")))
    
    root_dir = ORIGINAL_IMAGES_DIR
    for i, filename in tqdm.tqdm(enumerate(os.listdir(root_dir))):
        if filename.endswith(".png"):
            image = Image.open(os.path.join(root_dir, filename))
            image = image.convert("RGB")
            image.save(os.path.join(root_dir, filename.replace(".png", ".jpg")))


if __name__ == "__main__":
    # # load the model and tokenizer
    # model, tokenizer = load_chatgpt3_6b_model()

    # id = 6
    # for split in ['basic_val', 'basic_train']:
    #     # output_file_path = f"/share/imagereward_work/prompt_reconstruction/output/{id}/postprocess_output_{split}.txt"
    #     # sys.stdout = open(output_file_path, "w")

    #     print(f"Filtering the reconstructed prompts for {split} - {id}...", file=sys.stderr)

    #     file_path = f"/share/imagereward_work/prompt_reconstruction/data/{id}/dalle3_revised_prompts_reconstructed_{split}.csv"
    #     df = filter_reconstructed_prompts(file_path, model, tokenizer)
    #     df.to_csv(f"/share/imagereward_work/prompt_reconstruction/data/{id}/dalle3_revised_prompts_reconstructed_filtered_{split}.csv", index=False)
    
    # print("Renaming files...")
    # rename_files()

    # print("Converting png to jpg...")
    # png2jpg()

    # get the original dalle3 prompts under the directory /share/imagereward_work/dalle3/***/revised_prompts.txt
    # and store the dir name and corresponding prompt in a csv file

    png_names = []
    prompts = []
    for root, dirs, files in os.walk("/share/imagereward_work/dalle3"):
        for d in dirs:
            # find file ends with .txt
            for file in tqdm.tqdm(os.listdir(os.path.join(root, d))):
                if file.endswith(".txt"):
                    file_path = os.path.join(root, d, file)
                    with open(file_path, "r") as f:
                        prompt = f.read().strip()
                        split = d.split("_")[1]
                        index = d.split("_")[2]
                        assert split in ['train', 'val', 'test']
                        assert len(index) == 4
                        png_name = f"{split}-{index}.png"
                        
                        png_names.append(png_name)
                        prompts.append(prompt)
    
    df = pd.DataFrame({"image": png_names, "prompt": prompts})
    # store to /share/imagereward_work/prompt_reconstruction/data/dalle3_prompts.csv
    df.to_csv("/share/imagereward_work/prompt_reconstruction/data/dalle3_prompts.csv", index=False)

    pass