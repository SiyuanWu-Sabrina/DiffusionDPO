import os, sys, io
from typing import List, Dict, Tuple, Union, Any, Optional

import pandas as pd
from datasets import Dataset, load_dataset, Image
from PIL import Image as PILImage

GENERATED_DALLE3_IMAGES_DIR = '/share/imagereward_work/prompt_reconstruction/data/generated_dalle3_images/'
ORIGINAL_DALLE3_IMAGES_DIR = '/share/imagereward_work/prompt_reconstruction/data/original_dalle3_images/'
DALLE3_PROMPT_BLIP2_FLAN = '/share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv'

LAION115M_INFO = '/share/imagereward_work/prompt_fidelity/data/generated_laion_images/laion115m.csv'
LAION_HIGH_RES_INFO = '/share/imagereward_work/prompt_fidelity/data/generated_laion_images/laion_high_res.csv'


def align_file(file_list_a: List[str], file_list_b: List[str]) -> Tuple[List[str], List[str]]:
    """
    Align file list a and file list b.
    """
    root_a = os.path.dirname(file_list_a[0])
    root_b = os.path.dirname(file_list_b[0])

    file_list_a = sorted(file_list_a)
    file_list_b = sorted(file_list_b)

    file_list_a = [os.path.basename(file) for file in file_list_a]
    file_list_b = [os.path.basename(file) for file in file_list_b]

    file_list_a = [file for file in file_list_a if file in file_list_b]
    file_list_b = [file for file in file_list_b if file in file_list_a]

    file_list_a = [os.path.join(root_a, file) for file in file_list_a]
    file_list_b = [os.path.join(root_b, file) for file in file_list_b]

    return file_list_a, file_list_b


def load_my_dataset(dataset_loader_args: Dict[str, Any], seed):
    dataset_all = {'train': {"jpg_0": [], "jpg_1": [], "label_0": [], "caption": []},
                   'validation': {"jpg_0": [], "jpg_1": [], "label_0": [], "caption": []},
                   'test': {"jpg_0": [], "jpg_1": [], "label_0": [], "caption": []}}

    for dataset_name in dataset_loader_args.keys():
        print(f"Loading dataset {dataset_name}...")
        if dataset_name == "dalle3":
            dataset = load_my_dataset_dalle3(**dataset_loader_args[dataset_name])
        elif dataset_name == "laion115m":
            dataset = load_my_dataset_laion("laion115m", **dataset_loader_args[dataset_name])
        elif dataset_name == "laion_high_res":
            dataset = load_my_dataset_laion("laion_high_res", **dataset_loader_args[dataset_name])
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
        # merge datasets
        for split in dataset.keys():
            for key in dataset[split].keys():
                dataset_all[split][key].extend(dataset[split][key])

        # print dataset info
        print(f"Dataset {dataset_name} loaded.")
        for split in dataset.keys():
            print(f"{split}: {len(dataset[split]['jpg_0'])} samples")
        print()

    print("Turning dataset dict into Dataset object...")
    dataset_train = Dataset.from_dict(dataset_all['train'])
    dataset_train = dataset_train.shuffle(seed=seed)
    dataset_train = dataset_train.cast_column("jpg_0", Image()).cast_column("jpg_1", Image())

    dataset_val = Dataset.from_dict(dataset_all['validation'])
    dataset_val = dataset_val.cast_column("jpg_0", Image()).cast_column("jpg_1", Image())

    dataset_test = Dataset.from_dict(dataset_all['test'])
    dataset_test = dataset_test.cast_column("jpg_0", Image()).cast_column("jpg_1", Image())

    return {"train": dataset_train, "validation": dataset_val, "test": dataset_test}


def load_my_dataset_laion(
    name: str,
    splits: List[str] = ["train", "validation", "test"],
    default_label: float = 1.0,
    seed: int = 42,
):
    if name == "laion115m":
        info_csv_file = LAION115M_INFO
    elif name == "laion_high_res":
        info_csv_file = LAION_HIGH_RES_INFO
    else:
        raise ValueError(f"Unknown laion dataset name: {name}")
    
    print("Loading my dataset...")
    info = pd.read_csv(info_csv_file)
    # info has columns [prompt,generated_image_path,original_image_path]

    # separate the dataset into train and val and test using random split
    train_info = info.sample(frac=0.8, random_state=seed)
    val_test_info = info.drop(train_info.index)
    val_info = val_test_info.sample(frac=0.5, random_state=seed)
    test_info = val_test_info.drop(val_info.index)

    # load dataset
    orig_imgs_train = train_info["original_image_path"].tolist()
    mod_imgs_train = train_info["generated_image_path"].tolist()
    prompt_train = train_info["prompt"].tolist()
    orig_imgs_val = val_info["original_image_path"].tolist()
    mod_imgs_val = val_info["generated_image_path"].tolist()
    prompt_val = val_info["prompt"].tolist()
    orig_imgs_test = test_info["original_image_path"].tolist()
    mod_imgs_test = test_info["generated_image_path"].tolist()
    prompt_test = test_info["prompt"].tolist()

    dataset_train = None
    dataset_val = None
    dataset_test = None

    if 'train' in splits:
        assert len(orig_imgs_train) == len(mod_imgs_train) == len(prompt_train)
        dataset_train = {
            "jpg_0": orig_imgs_train,
            "jpg_1": mod_imgs_train,
            "label_0": [default_label]*len(orig_imgs_train),
            "caption": prompt_train,
        }
    if 'validation' in splits:
        assert len(orig_imgs_val) == len(mod_imgs_val) == len(prompt_val)
        dataset_val = {
            "jpg_0": orig_imgs_val,
            "jpg_1": mod_imgs_val,
            "label_0": [default_label]*len(orig_imgs_val),
            "caption": prompt_val,
        }
    if 'test' in splits:
        assert len(orig_imgs_test) == len(mod_imgs_test) == len(prompt_test)
        dataset_test = {
            "jpg_0": orig_imgs_test,
            "jpg_1": mod_imgs_test,
            "label_0": [default_label]*len(orig_imgs_test),
            "caption": prompt_test,
        }
    
    dataset = {"train": dataset_train, "validation": dataset_val, "test": dataset_test}
    return dataset


def load_my_dataset_dalle3(
    original_images_dir: str = ORIGINAL_DALLE3_IMAGES_DIR,
    modified_images_root_dir: str = GENERATED_DALLE3_IMAGES_DIR,
    caption_csv_file: str = DALLE3_PROMPT_BLIP2_FLAN,
    modified_images_subdir: List[str] = [0,1,2,3,4,5,6],
    splits: List[str] = ["train", "validation", "test"],
    default_label: float = 1.0,
):
    print("Loading my dataset...")
    orig_imgs = os.listdir(original_images_dir)
    orig_imgs = [os.path.join(original_images_dir, img) for img in orig_imgs]
    orig_imgs = [img for img in orig_imgs if img.endswith(".jpg")]

    mod_imgs_dict = {}
    for subdir in modified_images_subdir:
        mod_imgs = os.listdir(os.path.join(modified_images_root_dir, str(subdir)))
        mod_imgs = [os.path.join(modified_images_root_dir, str(subdir), img) for img in mod_imgs]
        mod_imgs_dict[subdir] = mod_imgs
        mod_imgs_dict[subdir] = [img for img in mod_imgs_dict[subdir] if img.endswith(".jpg")]

    # read the caption csv file, 'image' col as key, 'prompt' col as value
    prompts = pd.read_csv(caption_csv_file)
    caption_dict = dict(zip(prompts["image"], prompts["prompt"]))
    caption_dict = {key.replace(".png", ".jpg"): value for key, value in caption_dict.items()}

    orig_imgs_train = []
    mod_imgs_train = []
    prompt_train = []
    orig_imgs_val = []
    mod_imgs_val = []
    prompt_val = []
    orig_imgs_test = []
    mod_imgs_test = []
    prompt_test = []

    for subdir in mod_imgs_dict:
        orig_imgs_now, mod_imgs_dict[subdir] = align_file(orig_imgs, mod_imgs_dict[subdir])
        assert len(orig_imgs_now) == len(mod_imgs_dict[subdir])

        # separate the dataset into train and val and test using the file name
        orig_imgs_now.sort()
        mod_imgs_dict[subdir].sort()
        for orig_img, mod_img in zip(orig_imgs_now, mod_imgs_dict[subdir]):
            assert os.path.basename(orig_img) == os.path.basename(mod_img)
            assert os.path.basename(orig_img) in caption_dict
            if "val" in orig_img:
                orig_imgs_val.append(orig_img)
                mod_imgs_val.append(mod_img)
                prompt_val.append(caption_dict[os.path.basename(orig_img)])
            elif "test" in orig_img:
                orig_imgs_test.append(orig_img)
                mod_imgs_test.append(mod_img)
                prompt_test.append(caption_dict[os.path.basename(orig_img)])
            else:
                orig_imgs_train.append(orig_img)
                mod_imgs_train.append(mod_img)
                prompt_train.append(caption_dict[os.path.basename(orig_img)])
        
    # load dataset from the file list
    # the dataset has keys ['train', 'validation', 'test']
    # and for each key, it has columns ['jpg_0', 'jpg_1', 'label_0', 'caption']
    dataset_train = None
    dataset_val = None
    dataset_test = None
    
    if 'train' in splits:
        assert len(orig_imgs_train) == len(mod_imgs_train) == len(prompt_train)
        dataset_train = {
            "jpg_0": orig_imgs_train,
            "jpg_1": mod_imgs_train,
            "label_0": [default_label]*len(orig_imgs_train),
            "caption": prompt_train,
        }
    
    if 'validation' in splits:
        assert len(orig_imgs_val) == len(mod_imgs_val) == len(prompt_val)
        dataset_val = {
            "jpg_0": orig_imgs_val,
            "jpg_1": mod_imgs_val,
            "label_0": [default_label]*len(orig_imgs_val),
            "caption": prompt_val,
        }

    if 'test' in splits:
        assert len(orig_imgs_test) == len(mod_imgs_test) == len(prompt_test)
        dataset_test = {
            "jpg_0": orig_imgs_test,
            "jpg_1": mod_imgs_test,
            "label_0": [default_label]*len(orig_imgs_test),
            "caption": prompt_test,
        }
        
    dataset = {"train": dataset_train, "validation": dataset_val, "test": dataset_test}
    return dataset


def show(dataset):
    print(dataset.keys())
    for key1 in dataset.keys():
        print(key1)
        print(dataset[key1].column_names)
        for data in dataset[key1]:
            print('jpg_0', data['jpg_0'].convert("RGB"))
            print('jpg_1', data['jpg_1'].convert("RGB"))
            print('label_0', data['label_0'])
            print('caption', data['caption'])
            break
        print()

def show_orig(dataset):
    print(dataset.keys())
    for key1 in dataset.keys():
        print(key1)
        print(dataset[key1].column_names)
        for data in dataset[key1]:
            print('jpg_0', PILImage.open(io.BytesIO(data['jpg_0'])).convert("RGB"))
            print('jpg_1', PILImage.open(io.BytesIO(data['jpg_1'])).convert("RGB"))
            print('label_0', data['label_0'])
            print('caption', data['caption'])
            break
        print()


if __name__ == "__main__":
    dataset = load_my_dataset()
    show(dataset)
    print(dataset)
    print("##################################################")

    print("Original load_dataset...")
    dataset = load_dataset("/share/img_datasets/pickapic_v2")
    show_orig(dataset)
    print(dataset)
