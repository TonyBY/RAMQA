from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import List
from tqdm import tqdm
from transformers import AutoProcessor

from io import BytesIO
from PIL import Image
import base64
import numpy as np
import pandas as pd
import random

import logging
logger = logging.getLogger()


class RankingDatasetWebQA(Dataset):
    def __init__(self, 
                 data: List[dict] = None, 
                 target_label_distribution: List[int]=[1.0, 1.0],
                 weighted_sampling: bool=False,
                 seed: int = None,
                 is_debug: bool=False, 
                 is_train: bool=True,
                 is_test: bool=False,
                 processor: AutoProcessor = None,
                 max_length: int = None,
                 lineidx_path: str=None,
                 img_tsv_path: str=None,

                ):
        super().__init__()

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.is_debug = is_debug
        self.is_train = is_train

        if self.is_train:
            random.shuffle(data)

        self.processor = processor
        self.max_length = max_length

        self.img_tsv_path = img_tsv_path

        self.sample_num = processor.image_processor.resample
        self.h, self.w = processor.image_processor.crop_size.values()

        with open(lineidx_path, "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

        self.data = pd.DataFrame(data)
        if is_debug:
            self.data = self.data[:256]

        self.target_label_distribution = target_label_distribution

        if not is_test:
            self.class_weights = self.get_class_weights()
            logger.info(f"self.class_weights: {self.class_weights}")

        if weighted_sampling:
            self.sampler = self.get_weighted_data_sampler()
        else:
            self.sampler = None
        
    def get_weighted_data_sampler(self):
        if self.class_weights != None:
            class_weights = self.class_weights
        
        sample_weights = [0] * len(self.data)
        for idx in tqdm(range(len(self.data)), desc='Getting weighted sampler based on distribution of classes...'):
            class_id = self.data.iloc[idx].label
            class_weight = class_weights[class_id]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, 
                                        num_samples=len(sample_weights), 
                                        replacement=True) 
        # If we set replacement=False, we will only see that example once.
        # But, since we are doing oversampling, we want it to be True.
        return sampler

    def get_class_weights(self):
        ep = 1e-5
        logger.info(f"self.target_label_distribution: {self.target_label_distribution}")
        pos_cnt, neg_cnt = self.get_class_cnt()
        class_weights = [self.target_label_distribution[0]/(neg_cnt+ep), self.target_label_distribution[1]/(pos_cnt+ep)]
        return class_weights
    
    def get_class_cnt(self):
        pos_cnt = len(self.data.iloc[self.data.index[self.data.label==1]])
        neg_cnt = len(self.data.iloc[self.data.index[self.data.label==0]])
        logger.info(f"pos_cnt: {pos_cnt}")
        logger.info(f"neg_cnt: {neg_cnt}")

        assert pos_cnt + neg_cnt == len(self.data)
        return pos_cnt, neg_cnt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, image_id, text, label = self.data.iloc[idx]

        if image_id == "nan" or image_id == '' or image_id == None or np.isnan(image_id):
            inputs =  (str(question), None, str(text), int(label))
        else:
            try:
                inputs =  (str(question), int(image_id), str(text), int(label))
            except Exception as e:
                logger.debug(f"type(question): {type(question)}")
                logger.debug(f"type(image_id): {type(image_id)}")
                logger.debug(f"image_id: {image_id}")
                logger.debug(f"text: {text}")
                logger.debug(f"type(label): {type(label)}")
                logger.debug(f"label: {label}")
                raise e
            
        question, image_id, text, label = inputs
        prompt = f"Question: {question}\nEvidence: <image> {text}</s>"

        input_ids, attention_mask = self.processor.tokenizer(prompt,
                                                            padding=True,
                                                            truncation=True,
                                                            return_tensors="pt",
                                                            max_length=self.max_length,
                                                            ).values()
        
        if image_id:
            with open(self.img_tsv_path, "r") as fp:
                fp.seek(self.lineidx[int(image_id)%10000000])
                _, img_base64 = fp.readline().strip().split('\t')
            try:
                img = Image.open(BytesIO(base64.b64decode(img_base64)))
                try:
                    pixel_values = self.processor.image_processor(img, return_tensors='pt')["pixel_values"]
                except Exception as e:
                    logger.info(f"img: {img}")
                    logger.info(f"image_id: {image_id}")
                    logger.info(f"question: {question}")
                    logger.info(f"text: {text}")
                    logger.info(f"label: {label}")
                    logger.info(f"WARNING: {e}")
                    pixel_values = torch.zeros((1, self.sample_num, self.h, self.w))
            except Exception as e:
                logger.info(f"image_id: {image_id}")
                logger.info(f"question: {question}")
                logger.info(f"text: {text}")
                logger.info(f"label: {label}")
                logger.info(f"WARNING: {e}")
                pixel_values = torch.zeros((1, self.sample_num, self.h, self.w))
        else:
            pixel_values = torch.zeros((1, self.sample_num, self.h, self.w))

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'label': label,
                }
