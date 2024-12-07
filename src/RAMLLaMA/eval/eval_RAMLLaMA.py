from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-6]))
sys.path.append('/'.join(pwd.split('/')[:-5]))
sys.path.append('/'.join(pwd.split('/')[:-4]))
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import logging
from PIL import ImageFile
import random
from tqdm import tqdm
from typing import List

import torch
import transformers

from unsloth import FastLanguageModel

from RAMQA.src.utils.config import parser
from RAMQA.src.utils.args import prepare_logger
from RAMQA.src.utils.data_utils import read_jsonl, make_directory, save_jsonl, get_file_name

np.set_printoptions(precision=4)
ImageFile.LOAD_TRUNCATED_IMAGES = True

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

torch.set_float32_matmul_precision('medium')

def eval(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.init_model:
        # Reload model in FP16 and merge it with LoRA weights
        # 2. Load Llama3 model
        model, tokenizer = FastLanguageModel.from_pretrained(
                                                                model_name = args.model_path,
                                                                max_seq_length = args.max_seq_len,
                                                                dtype = None,
                                                                load_in_4bit = args.bits == 4,
                                                            )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
                                                                model_name = args.llama_model_name,
                                                                max_seq_length = args.max_seq_len,
                                                                dtype = None,
                                                                load_in_4bit = args.bits == 4,
                                                            )
        

    def generate_text_in_batch(text_list: List[str]):
        inputs = tokenizer(text_list, 
                           return_tensors="pt", 
                           padding=True,
                           ).to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    

    testing_data = read_jsonl(args.test_file)
    if args.debug:
        testing_data = testing_data[:args.predict_batch_size * 2 + 1]

    cache_path = os.path.join(args.output_dir, 'llama_qa_' + get_file_name(args.test_file) + '.jsonl')
    if os.path.exists(cache_path):
        logger.info(f"Loading cached examples from: {cache_path}")
        output = read_jsonl(cache_path)
        logger.info("Done.")
    else:
        output = []

    assert len(output) % args.predict_batch_size == 0

    total_steps = len(testing_data) // args.predict_batch_size if len(testing_data) % args.predict_batch_size == 0 else len(testing_data) // args.predict_batch_size + 1
    
    cached_steps = len(output) // args.predict_batch_size

    logger.info(f"total_steps: {total_steps}")
    logger.info(f"cached_steps: {cached_steps}")

    for i in tqdm(range(0, len(testing_data), args.predict_batch_size), desc='Predicting answers based on context...', total=total_steps):
        if i < cached_steps * args.predict_batch_size:
            continue

        prompts = []
        for item in testing_data[i: i + args.predict_batch_size]:
            prompt = item['text'].split('### Response:')[0] + '### Response:\n        '
            prompts.append(prompt)

        generated_text_list = generate_text_in_batch(prompts)
        logger.debug(f"len(generated_text_list): {len(generated_text_list)}")
        
        for item, generated_text in zip(testing_data[i: i + args.predict_batch_size], generated_text_list):
            prediction = generated_text.split('### Response:')[1].strip()

            item['prediction'] = prediction
            output.append(item)

        cache_interval = 20
        if args.debug:
            cache_interval = 1
        logger.debug(f"cache_interval: {cache_interval}")
        if i % cache_interval == 0:
            logger.info(f"Saving {len(output)} examples to: {cache_path}...")
            save_jsonl(output, cache_path)
            logger.info("Done.")

    logger.info(f"Saving {len(output)} examples to: {cache_path}...")
    save_jsonl(output, cache_path)
    logger.info(f"len(output): {len(output)}")
    logger.info(f"len(testing_data): {len(testing_data)}")
    assert len(output) == len(testing_data)
    logger.info("Done.")


if __name__=='__main__':
    make_directory(args.output_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.output_dir, "llama3_webqa_eval.log"))
    logger.info(args)
    eval(args)
    logger.info("ALL DONE!")
    