from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-6]))
sys.path.append('/'.join(pwd.split('/')[:-5]))
sys.path.append('/'.join(pwd.split('/')[:-4]))
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

import argparse
import copy
import numpy as np

from PIL import Image
from PIL import ImageFile

import re
from tqdm import tqdm
from typing import List
import zipfile

import torch
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration



from src.utils.data_utils import get_file_name, read_json, read_jsonl, json_to_jsonl, save_jsonl, make_directory, format_trans_mmqa_to_ranking
from src.utils.args import prepare_logger, str2bool

import logging
logger = logging.getLogger()

np.set_printoptions(precision=4)
ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_IMAGE_TOKEN = "<image>"

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)
    

def image_description_generation(img_context_list: List[dict],
                                 question: str='',
                                 image_zip_file_path: str='',
                                 llava_processor = None,
                                 llava_model = None,
                                 max_new_tokens: int=100,
                                ) -> List[dict]:

    prompts = []
    images = []
    for ctx in img_context_list:
        ctx_id = ctx['id']
        ctx_title = clean_text(ctx['title'])
        image_path = ctx['path']

        llava_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n Create a detailed description for the image with the caption: '{ctx_title}'. Your description should provide enough detail to help answer the question: '{question}'. However, if the subject mentioned in the caption is not relevant to the question, you can disregard the question when creating your description. \nASSISTANT:"
        prompts.append(llava_prompt)
        
        with zipfile.ZipFile(image_zip_file_path, 'r') as zip_file:
            file_name = 'final_dataset_images/' + image_path
            with zip_file.open(file_name) as image_file:
                try:
                    # Load the image into memory
                    image = Image.open(image_file)
#                     display(image)
                except Exception as e:
                    logger.info(f"WARNING: {e}")
                    logger.info(f"ctx_id: {ctx_id}")
                    logger.info(f"ctx_title: {ctx_title}")
                    image = None
                images.append(copy.deepcopy(image))
                
    # Perform batch inference
    try:
        inputs = llava_processor(prompts, images=images, return_tensors="pt", padding=True).to(llava_model.device)
        output = llava_model.generate(**inputs, max_new_tokens=max_new_tokens)
        answers = llava_processor.batch_decode(output, skip_special_tokens=True)
    except (TypeError, ValueError) as error:
        logger.info(f"WARNING: {error}")
        answers = []
        for prompt, image in zip(prompts, images):
            try:
                inputs = llava_processor([prompt], images=[image], return_tensors="pt", padding=True).to(llava_model.device)
                output = llava_model.generate(**inputs, max_new_tokens=max_new_tokens)
                answer = llava_processor.batch_decode(output, skip_special_tokens=True)
                answers.append(answer[0])
            except:
                answers.append('ASSISTANT: This image is broken. Please refer to the title and use your own knowledge.')
        
    output = []
    for i, ctx in enumerate(img_context_list):
        ctx['text'] = answers[i].split('ASSISTANT:')[-1].strip()
        logger.debug(f"ctx_title: {ctx_title}")
        logger.debug(f"ctx['text']: {ctx['text']}")
        logger.debug("\n************************************************************")   
        output.append(ctx)
    return output


def get_image_descriptions(data_list: List[dict],
                           image_zip_file_path: str = '',
                           llava_processor = None,
                           llava_model = None,
                           max_new_tokens: int=100,
                           cache_path: str='',
                          ) -> List[dict]:

    if os.path.exists(cache_path):
        logger.info(f"Loading cached examples from: {cache_path}")
        output = read_jsonl(cache_path)
        logger.info("Done.")
    else:
        output = []

    cached_steps = len(output)

    for i, item in tqdm(enumerate(data_list), 
                        desc='Generating image descriptions for MMQA dataset...', 
                        total=len(data_list),
                        ):
        if i < cached_steps:
            continue
        
        question = item['question']
       
        image_context = item['img_neg_context'] + item['img_pos_context']

        if len(image_context) == 0:
            output.append(item)
            continue

        image_context = image_description_generation(image_context,
                                                    question=question,
                                                    image_zip_file_path=image_zip_file_path,
                                                    llava_processor=llava_processor,
                                                    llava_model=llava_model,
                                                    max_new_tokens=max_new_tokens,
                                                    )
        
        item['img_neg_context'] = image_context[: len(item['img_neg_context'])]
        item['img_pos_context'] = image_context[len(item['img_neg_context']) : ]
            
        output.append(item)

        if args.debug:
            caching_interval = 1
        else:
            caching_interval = 30

        if i % caching_interval == 0:
            logger.info(f"Saving {len(output)} examples to: {cache_path}...")
            save_jsonl(output, cache_path)
            logger.info("Done.")

    return output 


def main(args):
    if args.data_path.endswith('.json'):
        mmqa_data = read_json(args.data_path)
        mmqa_data = json_to_jsonl(mmqa_data)
    elif args.data_path.endswith('.jsonl'):
        mmqa_data = read_jsonl(args.data_path)
    else:
        raise Exception(f"Unsupported file type: {args.data_path}")

    text_corpus_dict = read_json(args.text_corpus_path)
    image_corpus_dict = read_json(args.image_corpus_path)

    mmqa_ranking_train_data = format_trans_mmqa_to_ranking(mmqa_data,
                                                           use_mixed_txt_img=args.use_mixed_txt_img,
                                                           text_corpus_dict=text_corpus_dict,
                                                           image_corpus_dict=image_corpus_dict,
                                                          )
    
    if args.debug:
        mmqa_ranking_train_data = mmqa_ranking_train_data[:5]

    logger.info(f"len(mmqa_ranking_train_data): {len(mmqa_ranking_train_data)}")
    
    llava_processor = AutoProcessor.from_pretrained(args.llava_model_name)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        args.llava_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    output_path = os.path.join(args.output_dir, 'MMQA_with_ImageDescription_' + get_file_name(args.data_path) + '.jsonl')

    dataset = get_image_descriptions(mmqa_ranking_train_data,
                                    image_zip_file_path=args.image_zip_file_path,
                                    llava_processor=llava_processor,
                                    llava_model=llava_model,
                                    max_new_tokens=args.max_new_tokens,
                                    cache_path=output_path,
                                    )
    
    save_jsonl(dataset, output_path)

    logger.info(f"len(dataset): {len(dataset)}")
    logger.info(f"len(mmqa_ranking_train_data): {len(mmqa_ranking_train_data)}")
    assert len(dataset) == len(mmqa_ranking_train_data)
    
    logger.info(f"Saving dataset to: {output_path}")
    
    

def define_args(parser):
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        )
    parser.add_argument('--debug',
                        type=str2bool, 
                        nargs='?',
                        const=True, 
                        default=False,
                        help="Activate debug mode."
                        )
    parser.add_argument('--use_mixed_txt_img',
                        type=str2bool, 
                        nargs='?',
                        const=True, 
                        default=False,
                        help="Whether to include questions that require mixed text and image evidence to answer."
                        )
    parser.add_argument('--llava_model_name',
                        type=str,
                        required=True,
                        )
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        )
    parser.add_argument('--image_zip_file_path',
                        type=str,
                        required=False,
                        default=""
                        )
    parser.add_argument('--text_corpus_path',
                        type=str,
                        required=True,
                        default=""
                        )
    parser.add_argument('--image_corpus_path',
                        type=str,
                        required=True,
                        default=""
                        )
    parser.add_argument('--max_new_tokens',
                        type=int,
                        required=True,
                        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating description for the images in the MMQA dataset using LLaVa model.')
    define_args(parser)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        make_directory(args.output_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.output_dir, "data_construction.log"))
    main(args)
    logger.info("All Done!")
