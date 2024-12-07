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
import base64
from io import BytesIO
import numpy as np

from PIL import Image
from PIL import ImageFile

import re
from tqdm import tqdm
from typing import List

import torch
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration



from src.utils.data_utils import get_file_name, read_json, read_jsonl, json_to_jsonl, save_jsonl, make_directory, format_trans_webqa_to_ranking
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
                                 img_tsv_path: str='',
                                 lineidx=None,
                                 llava_processor = None,
                                 llava_model = None,
                                 max_new_tokens: int=100,
                                ) -> List[dict]:

    prompts = []
    images = []
    for ctx in img_context_list:
        ctx_id = ctx['id']
        ctx_title = clean_text(ctx['title'])

        llava_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n Create a detailed description for the image with the caption: '{ctx_title}'. Your description should provide enough detail to help answer the question: '{question}'. However, if the subject mentioned in the caption is not relevant to the question, you can disregard the question when creating your description. \nASSISTANT:"
        prompts.append(llava_prompt)
        
        with open(img_tsv_path, "r") as fp:
            fp.seek(lineidx[int(ctx_id)%10000000])
            _, img_base64 = fp.readline().strip().split('\t')
        try:
            image = Image.open(BytesIO(base64.b64decode(img_base64)))
        except Exception as e:
            logger.info(f"WARNING: {e}")
            logger.info(f"ctx_id: {ctx_id}")
            logger.info(f"ctx_title: {ctx_title}")
            image = None            
        images.append(image)
        
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
                           imgs_lineidx_path: str = '',
                           img_tsv_path: str = '',
                           llava_processor = None,
                           llava_model = None,
                           max_new_tokens: int=100,
                           is_test: bool=False,
                           cache_path: str='',
                          ) -> List[dict]:
    
    with open(imgs_lineidx_path, "r") as fp_lineidx:
        lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

    if os.path.exists(cache_path):
        logger.info(f"Loading cached examples from: {cache_path}")
        output = read_jsonl(cache_path)
        logger.info("Done.")
    else:
        output = []

    cached_steps = len(output)

    for i, item in tqdm(enumerate(data_list), 
                        desc='Generating image descriptions for WebQA dataset...', 
                        total=len(data_list),
                        ):
        if i < cached_steps:
            continue
        
        question = item['question']
        if is_test:
            if len(item['img_context']) == 0:
                output.append(item)
                continue

            item['img_context'] = image_description_generation(item['img_context'],
                                                               question=question,
                                                               img_tsv_path=img_tsv_path,
                                                               lineidx=lineidx,
                                                               llava_processor=llava_processor,
                                                               llava_model=llava_model,
                                                               max_new_tokens=max_new_tokens,
                                                              )
            
        else:
            image_context = item['img_neg_context'] + item['img_pos_context']

            if len(image_context) == 0:
                output.append(item)
                continue

            image_context = image_description_generation(image_context,
                                                           question=question,
                                                           img_tsv_path=img_tsv_path,
                                                           lineidx=lineidx,
                                                           llava_processor=llava_processor,
                                                           llava_model=llava_model,
                                                           max_new_tokens=max_new_tokens,
                                                          )
            
            item['img_neg_context'] = image_context[: len(item['img_neg_context'])]
            item['img_pos_context'] = image_context[len(item['img_neg_context']) : ]
            
        output.append(item)

        if i % 30 == 0:
            logger.info(f"Saving {len(output)} examples to: {cache_path}...")
            save_jsonl(output, cache_path)
            logger.info("Done.")

    return output 


def main(args):
    if args.data_path.endswith('.json'):
        webqa_data = read_json(args.data_path)
        webqa_data = json_to_jsonl(webqa_data)
    elif args.data_path.endswith('.jsonl'):
        webqa_data = read_jsonl(args.data_path)
    else:
        raise Exception(f"Unsupported file type: {args.data_path}")
    
    if args.debug:
        webqa_data = webqa_data[:5]

    webqa_ranking_train_data = format_trans_webqa_to_ranking(webqa_data,
                                                            is_test=args.is_test,
                                                            )
    
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

    output_path = os.path.join(args.output_dir, 'WebQA_with_ImageDescription_' + get_file_name(args.data_path) + '.jsonl')

    dataset = get_image_descriptions(webqa_ranking_train_data,
                                    imgs_lineidx_path=args.imgs_lineidx_path,
                                    img_tsv_path=args.img_tsv_path,
                                    llava_processor=llava_processor,
                                    llava_model=llava_model,
                                    max_new_tokens=args.max_new_tokens,
                                    is_test=args.is_test,
                                    cache_path=output_path,
                                    )
    
    save_jsonl(dataset, output_path)

    logger.info(f"len(dataset): {len(dataset)}")
    assert len(dataset) == len(webqa_data)
    
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
    parser.add_argument('--is_test',
                        type=str2bool, 
                        nargs='?',
                        const=True, 
                        default=False,
                        help="Process for testing data."
                        )
    parser.add_argument('--llava_model_name',
                        type=str,
                        required=True,
                        )
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        )
    parser.add_argument('--imgs_lineidx_path',
                        type=str,
                        required=False,
                        default=""
                        )
    parser.add_argument('--img_tsv_path',
                        type=str,
                        required=True,
                        )
    
    parser.add_argument('--max_new_tokens',
                        type=int,
                        required=True,
                        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating description for the images in the WebQA dataset using LLaVa model.')
    define_args(parser)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        make_directory(args.output_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.output_dir, "data_construction.log"))
    main(args)
    logger.info("All Done!")
