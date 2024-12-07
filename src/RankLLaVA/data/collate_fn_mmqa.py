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
from torch.nn.utils.rnn import pad_sequence

import logging
logger = logging.getLogger()

def collate_fn_leftPad(batch_inputs, 
                        pad_token_id: Tensor = None,
                        ):
    flipped_batch_input_ids = []
    flipped_batch_attention_masks = []
    batch_pixel_values = []
    batch_labels = []
    for inputs in batch_inputs:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        pixel_values = inputs['pixel_values']
        label = inputs['label']
        
        flipped_batch_input_ids.append(input_ids.view(-1).flip(dims=[0]))
        flipped_batch_attention_masks.append(attention_mask.view(-1).flip(dims=[0]))
        batch_pixel_values.append(pixel_values)
        batch_labels.append(label)

        logger.debug(f"input_ids.shape: {input_ids.shape}")
        logger.debug(f"attention_mask.shape: {attention_mask.shape}")
        logger.debug(f"pixel_values.shape: {pixel_values.shape}")

    logger.debug(f"len(flipped_batch_input_ids): {len(flipped_batch_input_ids)}")
    logger.debug(f"len(flipped_batch_attention_masks): {len(flipped_batch_attention_masks)}")
    logger.debug(f"len(batch_pixel_values): {len(batch_pixel_values)}")
    logger.debug(f"len(batch_labels): {len(batch_labels)}")

    batch_input_ids = pad_sequence(flipped_batch_input_ids, batch_first=True, padding_value=pad_token_id).flip(dims=[1])
    batch_attention_masks = pad_sequence(flipped_batch_attention_masks, batch_first=True, padding_value=0).flip(dims=[1])
    batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    output = {'input_ids':  batch_input_ids,
            'attention_mask': batch_attention_masks,
            'pixel_values': batch_pixel_values,
            'labels': batch_labels,
            }
    logger.debug(f"output: {output}")
    return output


def collate_fn_rightPad(batch_inputs, 
               pad_token_id: Tensor = None,
               ):
    batch_input_ids = []
    batch_attention_masks = []
    batch_pixel_values = []
    batch_labels = []
    for inputs in batch_inputs:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        pixel_values = inputs['pixel_values']
        label = inputs['label']
        
        batch_input_ids.append(input_ids.view(-1))
        batch_attention_masks.append(attention_mask.view(-1))
        batch_pixel_values.append(pixel_values)
        batch_labels.append(label)

        logger.debug(f"input_ids.shape: {input_ids.shape}")
        logger.debug(f"attention_mask.shape: {attention_mask.shape}")
        logger.debug(f"pixel_values.shape: {pixel_values.shape}")

    logger.debug(f"len(batch_input_ids): {len(batch_input_ids)}")
    logger.debug(f"len(batch_attention_masks): {len(batch_attention_masks)}")
    logger.debug(f"len(batch_pixel_values): {len(batch_pixel_values)}")
    logger.debug(f"len(batch_labels): {len(batch_labels)}")

    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
    batch_attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=0)
    batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return {'input_ids':  batch_input_ids,
            'attention_mask': batch_attention_masks,
            'pixel_values': batch_pixel_values,
            'labels': batch_labels,
            }