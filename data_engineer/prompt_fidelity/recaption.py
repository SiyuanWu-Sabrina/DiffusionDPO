import os, sys

from PIL import Image

from config import LAION_HIGH_RES_PATH, LAION115M_PATH

def read_dataset(dataset_name='laion_high_res'):
    if dataset_name == 'laion_high_res':
        # read all jpg files in LAION_HIGH_RES_PATH
        # return the list of PIL.Image objects
        images = []
        for filename in os.listdir(LAION_HIGH_RES_PATH):
            if filename.endswith('.jpg'):
                images.append(Image.open(os.path.join(LAION_HIGH_RES_PATH, filename)))
        return images
    elif dataset_name == 'laion115m':
        images = []
        for filename in os.listdir(LAION115M_PATH):
            if filename.endswith('.jpg'):
                images.append(Image.open(os.path.join(LAION115M_PATH, filename)))
        return images
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_cogvlm_model():
    # Load the COGVL model
    pass

def recaption(images, model):
    pass