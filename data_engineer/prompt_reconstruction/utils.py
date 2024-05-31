from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import DiffusionPipeline
from PIL import Image

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import *


def load_chatgpt3_6b_model(path=CHATGLM3_6B_PRETRAINED_MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()
    model = model.eval()
    return model, tokenizer


def text_similarity(text1, text2, model, tokenizer) -> int:
    question = "What is the similarity between text a and text b? Answer with a number between 0 and 100: "
    # calculate the similarity between two texts
    text_pair = "\nText a: " + text1 + "\n Text b: " + text2
    response, _ = model.chat(tokenizer, question + text_pair, history=[])

    # get the similarity score in the output text, the score is the last number in the output text
    # use re module to do this
    try:
        similarity_score = int(re.findall(r'\d+', response)[-1])
    except:
        similarity_score = 0
    return similarity_score


def get_image_size(image_file_path: str) -> Tuple[int, int]:
    with Image.open(image_file_path) as img:
        width, height = img.size
    return width, height


if __name__ == "__main__":
    print("size of the sdxl image: ", get_image_size("/share/home/wusiyuan/imagereward_work/prompt_reconstruction/cat.png"))
    print("size of the dalle image: ", get_image_size("/share/home/wusiyuan/imagereward_work/dalle3/basic_test_0000_Two_exquisitely_wrought_silver_p/dalle3_4a767f62-ec1f-4efb-a3c8-a302b26fd2ef.png"))
