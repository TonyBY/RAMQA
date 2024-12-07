from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-4]))
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

import copy
import numpy as np
import random
import re

from PIL import ImageFile

from tqdm import tqdm
from typing import List


from src.utils.data_utils import read_jsonl, save_jsonl, read_json, json_to_jsonl, save_json

import argparse
import logging
logger = logging.getLogger()

np.set_printoptions(precision=4)
ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_SYSTEM_PROMPT_FOR_RETRIEVAL = """
Please identify and return all relevant Evidence IDs that can help answer the question, filtering out any distracting background information provided.
""".strip()

DEFAULT_SYSTEM_PROMPT_FOR_QA = """
First, try to answer the following question based on your own knowledge. If you're unsure of the answer, use the provided background information for guidance. However, keep in mind that this context may not always be accurate.
""".strip()


DEFAULT_SYSTEM_PROMPT_FOR_RETRIEVAL_AND_QA = """
First, identify and return all relevant Evidence IDs that can assist in answering the question, while excluding any unrelated background information provided. Next, attempt to answer the question using your own knowledge. If uncertain, refer to those identified relevant evidence for guidance. However, be aware that these evidence may not always be accurate.
""".strip()


DEFAULT_IMAGE_TOKEN = "<image>"


prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""


def define_args(parser):
    parser.add_argument('--rank_result_data_path',
                        type=str,
                        required=True,
                        default=""
                        )
    
    parser.add_argument('--is_test',
                        type=str,
                        required=True,
                        default="False"
                        )

    parser.add_argument('--train_data_path',
                        type=str,
                        required=True,
                        default=""
                        )
     
    parser.add_argument('--dev_data_path',
                        type=str,
                        required=True,
                        default=""
                        )
    
    parser.add_argument('--test_data_path',
                        type=str,
                        required=True,
                        default=""
                        )
    
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        default=""
                        )
    
    parser.add_argument('--global_evi_dict_path',
                        type=str,
                        required=False,
                        default=""
                        )
    
    parser.add_argument('--prompt_style',
                        type=str,
                        required=False,
                        default="llama3"
                        )
    
    parser.add_argument('--prompt_task',
                        type=str,
                        required=False,
                        default="JOINT"
                        )
    
    parser.add_argument('--topk',
                        type=int,
                        required=False,
                        default=15
                        )
    
    parser.add_argument('--permute_times',
                        type=int,
                        required=False,
                        default=5
                        )


def get_global_img_desciption_dic(all_data: List[dict]) -> dict:
    global_evidence_dict = {}
    for item in tqdm(all_data):
        if 'txt_neg_context' in item:
            evidence_list = item['txt_neg_context'] + item['txt_pos_context'] + item['img_neg_context'] + item['img_pos_context']
        else:
            evidence_list = item['txt_context'] + item['img_context']
            
        for evi in evidence_list:
            evi_id = str(evi['id'])
            if evi_id not in global_evidence_dict:
                global_evidence_dict[evi_id] = evi


def unify_evi_format(input_data_list, 
                     global_evidence_dict={},
                    ):
    return [global_evidence_dict[str(item['id'])] for item in input_data_list]


def permute_evidence_and_increase_examples_from_rankingLLaVa_results(input_data_list: List[dict]=[],
                                                                     topk: int=20,
                                                                     permute_times: int=5,
                                                                     global_evidence_dict: dict={},
                                                                     is_test: bool = False,
                                                                      ):
    output = []
    expected_output_length = 0
    for i, item in tqdm(enumerate(input_data_list), 
                        desc='Permuting the evidence list...', 
                        total=len(input_data_list),
                       ):
        # NOTE: We do not need to keep the golden evidence sequence the same in the input and the output. Which will make the model more robust.
        # However, we also do not want to permute/change the golden evidence sequce, becasue it may cause the model not learning with the Casual LM objective.
        context = item['context'][:topk]
        if is_test:
            expected_output_length += 1
            item['context'] = unify_evi_format(context, global_evidence_dict=global_evidence_dict)
            output.append(copy.deepcopy(item))
            continue 
        
        golden_evidence = item['positive_ctxs']
        
        context = unify_evi_format(context, global_evidence_dict=global_evidence_dict)
        golden_evidence = unify_evi_format(golden_evidence, global_evidence_dict=global_evidence_dict)
        
        del item['positive_ctxs']
        del item['negative_ctxs']
        del item['hard_negative_ctxs']
        
        if len(context) < permute_times:
            print(f"i: {i}, len(context): {len(context)} < permute_times: {permute_times}")
            
        if permute_times > 0:
            expected_output_length += min(permute_times, len(context))
            for _ in range(min(permute_times, len(context))):
                random.shuffle(context)
                item['context'] = context
                item['golden_evidence'] = golden_evidence
                output.append(copy.deepcopy(item))
        else:
            expected_output_length += 1
            item['context'] = context
            item['golden_evidence'] = golden_evidence
            output.append(copy.deepcopy(item))
            
    assert len(output) == expected_output_length
        
    return output  


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)


def generate_text(
                    question_context: str="",
                    response: str="", 
                    prompt_style: str='llama2',
                    prompt_task: str='QA', # QA, Retrieval, JOINT
                    is_test: bool=False,
                ) -> str:
    
    if prompt_task == 'QA':
        DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_FOR_QA
    elif prompt_task == 'Retrieval':
        DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_FOR_RETRIEVAL
    elif prompt_task == 'JOINT':
        DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_FOR_RETRIEVAL_AND_QA
    else:
        raise Exception(f"Unknown prompt_task: {prompt_task}")
    
    if prompt_style == 'llama2':
        if is_test:
            return f"<s>[INST] <<SYS>> {DEFAULT_SYSTEM_PROMPT} <</SYS>> {question_context} [/INST]"
        else:
            return f"<s>[INST] <<SYS>> {DEFAULT_SYSTEM_PROMPT} <</SYS>> {question_context} [/INST] {response} </s>"
    
    elif prompt_style == 'llama3':
        if is_test:
            return prompt_template.format(DEFAULT_SYSTEM_PROMPT, question_context, response)
        else:
            return prompt_template.format(DEFAULT_SYSTEM_PROMPT, question_context, response) + '<|end_of_text|>'
    else:
        raise Exception(f"Unrecongnized prompt styple name: {prompt_style}")
        
        
def get_response_prompt(data_point, 
                        prompt_task: str='QA', # QA, Retrieval, JOINT
                        ):
    
    answer = data_point['answer_text'].strip()
    
    if prompt_task == "QA":
        return answer        
    
    golden_evidence = data_point['golden_evidence'] # Make sure the golden_evidence is suffled in the preprocessing step.
    
    text = "*** RETRIEVL RESULT:\n                "    
    for i, item in enumerate(golden_evidence):
        ctx_id = item['id']
        text += f"{ctx_id};"
        
    text = text[:-1]
    
    if prompt_task == "Retrieval":
        return text
    
    if prompt_task == "JOINT":
        text += f'\n        *** ANSWER:\n                {answer}'
        return text
    
    raise Exception(f"Unknown prompt_task: {prompt_task}")
    
    

def get_user_input_prompt(data_point,
                          is_test: bool=False,
                          topk: int=100,
                         ):
    text = ""
    question = data_point['question'].strip()
    
    if is_test:
        if 'context' in data_point:
            context = data_point['context']
        else:
            context = data_point['img_context'] + data_point['txt_context']
    else:
        context = data_point['context'] # Make sure the context is suffled if the prompt is for retrieval purpos.
    
    context = context[:topk]
    
    text += f"question: {question.strip()}\n"
    
    for i, item in enumerate(context):
        ctx_id = item['id']
        ctx_txt = clean_text(item['text'])
        if 'title' in item:
            ctx_title = item['title']
        else:
            ctx_title = ''
                
        knowledge = 'Evidence ID: ' + str(ctx_id) + ' -- title: ' + ctx_title + ' -- Content: ' + ctx_txt
        text += f"        context{i}: {knowledge}\n\n"

    return text


def generate_example(data_point, 
                     is_test: bool=False,
                     prompt_style: str='llama2', # llama2, llama3
                     prompt_task: str='QA', # QA, Retrieval, JOINT
                     topk: int=100,
                 ): 
    
    question_context = get_user_input_prompt(data_point,
                                             is_test=is_test,
                                             topk=topk,
                                            )
    
    if is_test:
        response = ""
    else:
        response = get_response_prompt(data_point, 
                                       prompt_task=prompt_task,
                                      )
        
    return {
                "text": generate_text(question_context=question_context,
                                      response=response,
                                      prompt_style=prompt_style,
                                      prompt_task=prompt_task,
                                      is_test=is_test,
                                    )
                }


def process_dataset(data: List[dict]=None,
                          is_test: bool=False,
                          prompt_style: str='llama2', # llama2, llama3
                          prompt_task: str='QA', # QA, Retrieval, JOINT
                          topk: int=100,
                           ) -> List[dict]:
    output = []
    for i, item in tqdm(enumerate(data), 
                        desc=f'Constructing Generative {prompt_task} dataset...', 
                        total=len(data),
                       ):
        example = generate_example(item,
                                   is_test=is_test,
                                   prompt_style=prompt_style,
                                   prompt_task=prompt_task,
                                   topk=topk,
                                   )
        
        output.append(example)
          
    return output


def load_data(data_path: str) -> List[dict]:
    if os.path.exists(data_path):
        if data_path.endswith('.jsonl'):
            data = read_jsonl(data_path)
        elif data_path.endswith('.json'):
            data = json_to_jsonl(read_json(data_path))
        else:
            raise Exception('Unexpected data format.')
    else:
        logger.warning(f'File does not exist: {data_path}.')
        data = []
    return data
        

def main(args):
    # Builded a global image description dictionary
    if os.path.exists(args.global_evi_dict_path):
        global_evidence_dict = read_json(args.global_evi_dict_path)
    else:
        train_data = load_data(args.train_data_path)    
        dev_data = load_data(args.dev_data_path)
        test_data = load_data(args.test_data_path)

        global_evidence_dict = get_global_img_desciption_dic(train_data + dev_data + test_data)
        save_json(global_evidence_dict, args.global_evi_dict_path)

    # RankingLLaVa result format transformation
    rank_result_data = read_jsonl(args.rank_result_data_path)

    ## Permute Evidence
    permuted_data = permute_evidence_and_increase_examples_from_rankingLLaVa_results(
                                input_data_list=copy.deepcopy(rank_result_data),
                                topk = args.topk, 
                                permute_times = args.permute_times,
                                global_evidence_dict = global_evidence_dict,
                                is_test=args.is_test,
                                )
    
    ## Training data generation
    processed_data = process_dataset(permuted_data,
                                                 is_test=args.is_test,
                                                 prompt_style=args.prompt_style,
                                                 prompt_task=args.prompt_task,
                                                 topk=args.topk,
                                                )
    
    save_jsonl(processed_data, args.output_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args() 
    main(args)
    logger.info("All Done.")
