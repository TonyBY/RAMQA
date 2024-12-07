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
import argparse
import logging


from src.utils.data_utils import read_json, json_to_jsonl, save_jsonl

np.set_printoptions(precision=4)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger()


def define_args(parser):
    parser.add_argument('--webqa_data_path',
                        type=str,
                        required=True,
                        default="RAMQA/data/WebQA/main_data/WebQA_train.json"
                        )
    
    parser.add_argument('--img_tsv_path',
                        type=str,
                        required=False,
                        default="RAMQA/data/WebQA/imgs/imgs.tsv"
                        )
    
    parser.add_argument('--imgs_lineidx_path',
                        type=str,
                        required=False,
                        default="RAMQA/data/WebQA/imgs/imgs.lineidx"
                        )
    
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        default="RAMQA/data/WebQA/ranking_data/ranking_train_data_webqa.jsonl"
                        )
    
    parser.add_argument('--is_test',
                        type=str,
                        required=True,
                        default="False"
                        )
    

def main(args):
    webqa_data = read_json(args.webqa_data_path)
    webqa_data_lines = json_to_jsonl(webqa_data)

    """
    {
        question: ,
        label:
        image: ,
        text: ,
    }
    """
    output = []

    with open(args.imgs_lineidx_path, "r") as fp_lineidx:
        lineidx = [int(i.strip()) for i in tqdm(fp_lineidx.readlines())]

    for item in tqdm(webqa_data_lines):
        question = item['Q'].strip('"')

        if args.is_test.lower() == 'true':
            for img in item['img_Facts']:
                image_id = img['image_id']
                caption = img['caption']
                output.append({'question': question, 
                            'image_id': image_id,
                            'text': caption,
                            'label': 1
                                }
                            )
                
            for txt in item['txt_Facts']:
                text = txt['title'] + ' . ' + txt['fact']
                output.append({'question': question, 
                            'image_id': None,
                            'text': text,
                            'label': 1
                                }
                            )
                txt_num += 1
        
        else:
            for img in item['img_posFacts']:
                image_id = img['image_id']
                with open(args.img_tsv_path, "r") as fp:
                    fp.seek(lineidx[int(image_id)%10000000])
                    _, img_base64 = fp.readline().strip().split('\t')
                caption = img['caption']
                output.append({'question': question, 
                            'image_base64': img_base64,
                            'text': caption,
                            'label': 1
                                }
                            )
            for img in item['img_negFacts']:
                image_id = img['image_id']
                with open(args.img_tsv_path, "r") as fp:
                    fp.seek(lineidx[int(image_id)%10000000])
                    _, img_base64 = fp.readline().strip().split('\t')
                caption = img['caption']
                output.append({'question': question, 
                            'image_base64': img_base64,
                            'text': caption,
                            'label': 0
                                }
                            )
                            
            for txt in item['txt_posFacts']:
                text = txt['title'] + ' . ' + txt['fact']
                output.append({'question': question, 
                            'image_base64': '',
                            'text': text,
                            'label': 1
                                }
                            )
                            
            for txt in item['txt_negFacts']:
                text = txt['title'] + ' . ' + txt['fact']
                output.append({'question': question, 
                            'image_base64': '',
                            'text': text,
                            'label': 0
                                }
                            )
    
    save_jsonl(output, args.output_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args() 
    main(args)
    logger.info("All Done.")
