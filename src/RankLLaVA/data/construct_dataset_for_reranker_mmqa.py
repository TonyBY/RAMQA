from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from PIL import ImageFile
import numpy as np
from tqdm import tqdm


from RAMQA.src.utils.data_utils import read_jsonl, save_jsonl

np.set_printoptions(precision=4)
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import logging

logger = logging.getLogger()


def define_args(parser):
    parser.add_argument('--mmqa_data_path',
                        type=str,
                        required=True,
                        default="RAMQA/data/multimodalqa/dataset/MMQA_train.jsonl"
                        )
    
    parser.add_argument('--output_path',
                        type=str,
                        required=False,
                        default="RAMQA/data/multimodalqa/ranking_data/ranking_train_data_mmqa.jsonl"
                        )
    

def main(args):
    mmqa_data = read_jsonl(args.mmqa_data_path)

    """
    {
        q_id:,
        question: ,
        label:
        image_doc_id: ,
        text_doc_id: ,
    }
    """
    balanced_dev = False

    output = []
    for item in tqdm(mmqa_data):
        question_id = item['qid']
        question = item['question'].strip('"')
        
        num_pos = len(item['supporting_context'])
        num_neg = 0
        
        if item['metadata']['modalities'] == ['image'] or \
            item['metadata']['modalities'] == ['text'] or \
            item['metadata']['modalities'] == ['text', 'image'] or \
            item['metadata']['modalities'] == ['image', 'text']:
                
            supporting_img_doc_ids = []
            supporting_txt_doc_ids = []
            for sc in item['supporting_context']:
                if sc['doc_part'] == 'image':
                    supporting_img_doc_ids.append(sc['doc_id'])
                elif sc['doc_part'] == 'text':
                    supporting_txt_doc_ids.append(sc['doc_id'])
                else:
                    logger.debug(f"item: {sc['item']}")
                    raise Exception(f"Unexpectied data type: {sc['doc_part']}")
                    
            for image_doc_id in set(supporting_img_doc_ids):              
                output.append({'qid': question_id,
                            'question': question, 
                            'image_doc_id': image_doc_id,
                            'text_doc_id': None,
                            'label': 1
                                }
                            )
                            
            for text_doc_id in set(supporting_txt_doc_ids):
                output.append({'qid': question_id,
                            'question': question, 
                            'image_doc_id': None,
                            'text_doc_id': text_doc_id,
                            'label': 1,
                                }
                            )
                
            neg_img_ids = set(item['metadata']['image_doc_ids']) - set(supporting_img_doc_ids)
            neg_txt_ids = set(item['metadata']['text_doc_ids']) - set(supporting_txt_doc_ids)
                
            for image_doc_id in neg_img_ids:
                if balanced_dev and num_neg == num_pos:
                    break
                num_neg += 1
                                    
                output.append({'qid': question_id,
                            'question': question, 
                            'image_doc_id': image_doc_id,
                            'text_doc_id': None,
                            'label': 0
                                }
                            )
                            
            for text_doc_id in neg_txt_ids:
                if balanced_dev and num_neg == num_pos:
                    break
                    
                num_neg += 1
                output.append({'qid': question_id,
                            'question': question, 
                            'image_doc_id': None,
                            'text_doc_id': text_doc_id,
                            'label': 0,
                                }
                            )
        else:
            continue
    
    save_jsonl(output, args.output_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args() 
    main(args)
    logger.info("All Done.")
