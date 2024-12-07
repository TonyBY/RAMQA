from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-4]))
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from typing import List
from tqdm import tqdm
from transformers import AutoProcessor

from PIL import Image
import pandas as pd
import random

import zipfile
from RAMQA.src.utils.data_utils import read_json

import logging
logger = logging.getLogger()


class RankingDatasetMMQA(Dataset):
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
                 image_zip_file_path: str = None,
                 image_corpus_path: str=None,
                 text_corpus_path: str=None,

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

        self.text_corpus_dict = read_json(text_corpus_path)
        self.image_corpus_dict = read_json(image_corpus_path)
        self.image_zip_file_path = image_zip_file_path

        self.sample_num = processor.image_processor.resample
        self.h, self.w = processor.image_processor.crop_size.values()

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
    
    def get_image_by_path(self, image_path):
        # Open the zipped folder
        with zipfile.ZipFile(self.image_zip_file_path, 'r') as zip_file:  

            file_name = 'final_dataset_images/' + image_path
            with zip_file.open(file_name) as image_file:
                # Load the image into memory
                image = Image.open(image_file)
                return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, question, image_doc_id, text_doc_id, label = self.data.iloc[idx]

        if image_doc_id == "nan" or image_doc_id == '' or image_doc_id == None or pd.isnull(image_doc_id):
            text = self.text_corpus_dict[text_doc_id]['title'] +  ' . ' +  self.text_corpus_dict[text_doc_id]['text']
            inputs =  (str(question), None, str(text), int(label))
        else:
            try:
                text = self.image_corpus_dict[image_doc_id]['title']
                inputs =  (str(question), str(image_doc_id), str(text), int(label))
            except Exception as e:
                logger.debug(f"type(question): {type(question)}")
                logger.debug(f"type(image_id): {type(image_doc_id)}")
                logger.debug(f"image_doc_id: {image_doc_id}")
                logger.debug(f"type(label): {type(label)}")
                logger.debug(f"label: {label}")
                raise e
            
        question, image_doc_id, text, label = inputs
        prompt = f"Question: {question}\nEvidence: <image> {text}</s>"

        input_ids, attention_mask = self.processor.tokenizer(prompt,
                                                            padding=True,
                                                            truncation=True,
                                                            return_tensors="pt",
                                                            max_length=self.max_length,
                                                            ).values()
        
        if image_doc_id:
            image_path = self.image_corpus_dict[image_doc_id]['path']
            try:
                # Open the zipped folder
                with zipfile.ZipFile(self.image_zip_file_path, 'r') as zip_file:  

                    file_name = 'final_dataset_images/' + image_path
                    with zip_file.open(file_name) as image_file:
                        # Load the image into memory
                        img = Image.open(image_file)
                        try:
                            pixel_values = self.processor.image_processor(img, return_tensors='pt')["pixel_values"]
                        except Exception as e:
                            logger.info(f"img: {img}")
                            logger.info(f"image_doc_id: {image_doc_id}")
                            logger.info(f"question: {question}")
                            logger.info(f"text: {text}")
                            logger.info(f"label: {label}")
                            logger.info(f"WARNING: {e}")
                            pixel_values = torch.zeros((1, self.sample_num, self.h, self.w))
            except Exception as e:
                logger.info(f"image_doc_id: {image_doc_id}")
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
