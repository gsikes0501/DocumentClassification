from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from tqdm import tqdm

import numpy as np

import sys
from pathlib import Path

images_folder_path = "C:\\Users\\gsikes\\GitRepositories\\DataExtraction\\data\\inputs\\"
# for dir in Path(images_folder_path).glob("*"):
#     dir.rename(str(dir).lower().replace(" ", "_"))
    
lst_image_paths = list(Path(images_folder_path).glob("*"))
print(lst_image_paths)

lst_converted_img_pgs = []
for image_path in lst_image_paths:
    lst_converted_images = convert_from_path(image_path, poppler_path= r"C:\Users\gsikes\AdditionalSoftware\Release-23.11.0-0\poppler-23.11.0\Library\bin")
    for converted_img in lst_converted_images:
        lst_converted_img_pgs.append(converted_img)

for image in lst_converted_img_pgs:
    image.convert("RGB")

feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True, ocr_lang='eng')
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = layoutlmv3_processor = LayoutLMv3Processor(feature_extractor, tokenizer)

encoding = processor(
    image,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

print(encoding.keys())