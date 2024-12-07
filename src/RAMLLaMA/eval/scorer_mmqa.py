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

import difflib
import json
import numpy as np
import re
import string
from tqdm import tqdm
from typing import List, Union, Tuple, Set
from scipy.optimize import linear_sum_assignment
from word2number.w2n import word_to_num

from RAMQA.src.utils.data_utils import get_file_dir, read_jsonl, save_json, read_json

np.set_printoptions(precision=4)

import argparse
import logging
from RAMQA.src.utils.args import prepare_logger
logger = logging.getLogger()


def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)

def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text
    
    
def _lower(text: str) -> str:
    return text.lower()


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    # TODO: this is not included in the original drop evaluation script, we need to have our own in the end anyways.
    elif _is_word_number(text):
        return str(float(word_to_num(text)))
    else:
        return text


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False
    
    
def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False
    

def _answer_to_bags(
        answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def list_em(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        return 1.0
    else:
        return 0.0
    

def list_f1(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, gold_answers):
    scores_for_ground_truths = []
    for gold_answer in gold_answers:
        score = metric_fn(prediction, gold_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_predictions_example(prediction, gold_answer, example_types=None):
    eval_funcs = {
        "list_em": list_em,
        "list_f1": list_f1
    }
    instance_eval_results = {
        metric: metric_max_over_ground_truths(
            func, prediction, gold_answer
        ) for metric, func in eval_funcs.items()
    }

    return instance_eval_results

def evidence_filter(evidence_list: List[dict],
                    filter_type: str='PERQA',
                    threshold: float = 0.1,
                   ) -> List[dict]:
    
    if filter_type == 'PERQA':
        output = evidence_list[:2]
        if float(evidence_list[1]['score']) - float(evidence_list[2]['score']) < threshold:
            output.append(evidence_list[2])
        
    elif filter_type == 'Normalize':
        scores = [float(evi['score']) for evi in evidence_list]
        min_score = min(scores)
        max_score = max(scores)

        output = []
        for evi in evidence_list:
            score = (float(evi['score']) - (min_score + max_score) / 2) / (max_score - min_score)
            if score >= threshold:
                output.append(evi)
    else:
        raise Exception(f"Unknown filter_type: {filter_type}")
            
    if output:
        return output
    else:
        logger.info(f"WARNING: No predicted evidence left, use the top ranked one.")
        return evidence_list[0]
    

def levenshtein_distance(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def get_most_similar_string(target: str, candidates: List[str]):
    output = target
    max_sim=float('-inf')
    
    for c in candidates:
        distance = levenshtein_distance(target, c)
        if distance > max_sim:
            max_sim = distance
            output = c
    return output, max_sim


def get_ori_mmqa_data(mmqa_data):
    output = []
    for item in tqdm(mmqa_data):
        question = item['question']
        if item['metadata']['modalities'] == ['image'] or \
            item['metadata']['modalities'] == ['text'] or \
            item['metadata']['modalities'] == ['text', 'image'] or \
            item['metadata']['modalities'] == ['image', 'text']:
            if question == "In the television series in which Tyler Alvarez acts with a poster featuring the exterior of a house, what is the name of jude's boyfriend?":
                logger.info(f"item: {item['supporting_context']}")
            output.append(item)
            
    return output


def mmqa_eval(mmqa_RAMLLaMA_data: List[dict]=[], 
               mmqa_rank_data: List[dict]=[], 
               mmqa_ori_data: List[dict]=[],
               mmqa_global_evi_dict: dict={},
               is_test: bool=False,
               topk: int=5,
               switch: bool=False,
               ep: float=10**-5,
               output_dir: str='',
               batch_size: int=4,
               IS_GENERATIVE_RETRIEVAL: bool=False,
               filter_type: str='PERQA',
               threshold: float=0.1,
               evi_sim_th: float=0.9,
               save_results: bool=False,
               question_type: str='',
              ):
    
    assert len(mmqa_RAMLLaMA_data) == len(mmqa_rank_data) == len(mmqa_ori_data), f"len(mmqa_RAMLLaMA_data): {len(mmqa_RAMLLaMA_data)}, len(mmqa_rank_data): {len(mmqa_rank_data)}, len(mmqa_ori_data): {len(mmqa_ori_data)}"
    
    output = {}
    result_dict = {}
    rec_scores = []
    f1_scores = []
    golden_res_list = []
    gen_res_list = []
    
    mmqa_em_scores = []
    mmqa_f1_scores = []
    
    pred_evi_number_list = []
    
    questoin_cnt = 0
    for i, item in tqdm(enumerate(zip(mmqa_RAMLLaMA_data, mmqa_rank_data, mmqa_ori_data)), total=len(mmqa_RAMLLaMA_data)):
        qa_item, rank_item, org_item = item
        
        if question_type and org_item['metadata']['type'] != question_type:
            continue
            
        questoin_cnt += 1

        queston_id = org_item['qid']
        question = rank_item['question'].strip()
        
        if question[0] == '"':
            question = question[1:-1]

        if IS_GENERATIVE_RETRIEVAL and "*** ANSWER:" in qa_item['prediction'] and "*** RETRIEVL RESULT:" in qa_item['prediction']:
            pre_evi_ids, pred_answer = qa_item['prediction'].split('*** ANSWER:')
            pre_evi_ids = pre_evi_ids.split('*** RETRIEVL RESULT:')[-1].strip().split(';')
            pred_answer = pred_answer.strip()
            pred_answer = '. '.join(pred_answer.split('. ')[:1])
            pred_answer = pred_answer.strip()
            if not pred_answer.endswith('.'):
                pred_answer = pred_answer + '.'

            gen_res_list.append(pred_answer)
            pre_evi = []
            for idx in pre_evi_ids:   
                if idx in mmqa_global_evi_dict:
                    evi = mmqa_global_evi_dict[idx]
                    max_sim = 1.0
                else:
                    all_ctx = [ctx.split('Evidence ID:')[-1].split('### Response:')[0].split('-- title:')[0].strip() for ctx in qa_item['text'].split('### Input:')[-1].split('context0:')[-1].split("\n\n")]
                    logger.info("###########################")
                    logger.info(f"idx: {idx}")
                    evi, max_sim = get_most_similar_string(idx, all_ctx)

                    logger.info(f"evi: {evi}")
                    logger.info(f"max_sim: {max_sim}")
                    logger.info(f"all_ctx: {all_ctx}")
                    logger.info(f"question: {question}")
                    logger.info(f"qa_item['prediction']: {qa_item['prediction']}")
                    logger.info(f"pred_answer: {pred_answer}")
                    logger.info(f"pre_evi_ids: {pre_evi_ids}")

                if max_sim > evi_sim_th:
                    pre_evi.append(evi)
                else:
                    logger.info(f"max_sim: {max_sim} < evi_sim_th: {evi_sim_th}, dropping evi: {evi}")
                    
                if pre_evi == []:
                    pre_evi = [ctx for ctx in rank_item['context']]
                    pre_evi = evidence_filter(pre_evi,
                                               filter_type=filter_type,
                                               threshold=threshold,
                                              )
                    pre_evi_ids = set([str(ctx['id']) for ctx in pre_evi[:topk]])
                    pre_evi = [ctx for ctx in pre_evi[:topk]]
                    logger.info(f"Using filtered evidence: {[str(ctx['id']) for ctx in pre_evi[:topk]]}")
                    
        elif IS_GENERATIVE_RETRIEVAL and "*** RETRIEVL RESULT:" in qa_item['prediction']:
            pred_answer = ''
            pre_evi_ids = qa_item['prediction'].split('*** RETRIEVL RESULT:')[-1].strip().split(';')
            pre_evi = []
            for idx in pre_evi_ids:   
                if idx in mmqa_global_evi_dict:
                    evi = mmqa_global_evi_dict[idx]
                    max_sim = 1.0
                else:
                    all_ctx = [ctx.split('Evidence ID:')[-1].split('### Response:')[0].split('-- title:')[0].strip() for ctx in qa_item['text'].split('### Input:')[-1].split('context0:')[-1].split("\n\n")]
                    logger.info("###########################")
                    logger.info(f"idx: {idx}")
                    evi, max_sim = get_most_similar_string(idx, all_ctx)

                    logger.info(f"evi: {evi}")
                    logger.info(f"max_sim: {max_sim}")
                    logger.info(f"all_ctx: {all_ctx}")
                    logger.info(f"question: {question}")
                    logger.info(f"qa_item['prediction']: {qa_item['prediction']}")
                    logger.info(f"pred_answer: {pred_answer}")
                    logger.info(f"pre_evi_ids: {pre_evi_ids}")

                if max_sim > evi_sim_th:
                    pre_evi.append(evi)
                else:
                    logger.info(f"max_sim: {max_sim} < evi_sim_th: {evi_sim_th}, dropping evi: {evi}")
                    
                if pre_evi == []:
                    pre_evi = [ctx for ctx in rank_item['context']]
                    pre_evi = evidence_filter(pre_evi,
                                               filter_type=filter_type,
                                               threshold=threshold,
                                              )
                    pre_evi_ids = set([str(ctx['id']) for ctx in pre_evi[:topk]])
                    pre_evi = [ctx for ctx in pre_evi[:topk]]
                    logger.info(f"Using filtered evidence: {[str(ctx['id']) for ctx in pre_evi[:topk]]}")
        else:
            pred_answer = qa_item['prediction']
            pred_answer = '. '.join(pred_answer.split('. ')[:1])
            pred_answer = pred_answer.strip()
            if not pred_answer.endswith('.'):
                pred_answer = pred_answer + '.'

            gen_res_list.append(pred_answer)

            pre_evi = [ctx for ctx in rank_item['context']]

            pre_evi = evidence_filter(pre_evi,
                                       filter_type=filter_type,
                                       threshold=threshold,
                                      )

            pre_evi_ids = set([str(ctx['id']) for ctx in pre_evi[:topk]])
            pre_evi = [ctx for ctx in pre_evi[:topk]]

            logger.info("*************************************")
            logger.info(f"question: {question}")
            logger.info(f"qa_item['prediction']: {qa_item['prediction']}")
            logger.info(f"pred_answer: {pred_answer}")
            logger.info(f"pre_evi_ids: {pre_evi_ids}")
        
        pred_evi_number_list.append(len(pre_evi))

        if not is_test:
            golden_answers = rank_item['answers']
            golden_res_list.append(golden_answers)
            
            eval_scores = evaluate_predictions_example(pred_answer, golden_answers)
            
            mmqa_em = eval_scores['list_em'] * 100
            mmqa_f1 = eval_scores['list_f1'] * 100
            
            mmqa_em_scores.append(mmqa_em)
            mmqa_f1_scores.append(mmqa_f1)            

            golden_evi_ids = set([str(ctx['id']) for ctx in rank_item['positive_ctxs']])

            evi_hits = golden_evi_ids.intersection(pre_evi_ids)
            recall = len(evi_hits) / (len(golden_evi_ids) + ep)
            rec_scores.append(recall)
            precision = len(evi_hits) / (len(pre_evi_ids) + ep)
            f1 = (2 * recall * precision) / (recall + precision + ep)
            
            f1_scores.append(f1)
        
            result_dict[queston_id] = {'question': question, 
                                        'golden_ans': golden_answers, 
                                        'gen_ans': pred_answer,
                                        'pre_evi': pre_evi,
                                        'golden_id': list(golden_evi_ids),
                                        'evi_recall': recall,
                                        'evi_f1': f1,
                                        'mmqa_f1': mmqa_f1,
                                        'mmqa_em': mmqa_em,
                                        }
            
        pre_evi_ids =  [int(eid) if eid.isdigit() else eid for eid in list(pre_evi_ids)]
        output[queston_id] = {'answer': pred_answer, 'sources': list(pre_evi_ids)}
    
    logger.info(f"question_type: {question_type}")
    logger.info(f"questoin_cnt: {questoin_cnt}")
    
    if os.path.exists(output_dir) and save_results:
        result_path = os.path.join(output_dir, f'analysis_results_topk-{topk}.json')
        logger.info(f"Saving analysis_results to: {result_path}")
        save_json(result_dict, result_path)
        logger.info("Done.")
    
    rec_scores = np.array(rec_scores)
    f1_scores = np.array(f1_scores)
    
    mmqa_f1_scores = np.array(mmqa_f1_scores)
    mmqa_em_scores = np.array(mmqa_em_scores)
        
    pred_evi_number_list = np.array(pred_evi_number_list)
    avg_pred_evi_number = pred_evi_number_list.mean()
    
    
    rec_score = rec_scores.mean()
    f1_score = f1_scores.mean()
    mmqa_f1_score = mmqa_f1_scores.mean()
    mmqa_em_score = mmqa_em_scores.mean()
    
    result_str = json.dumps(result_dict, indent=4)
    
    logger.info(f"avg_pred_evi_number : {avg_pred_evi_number}")
    
    logger.info(f"rec_score@{topk}: {rec_score}")
    logger.info(f"f1_score@{topk}: {f1_score}")
    logger.info(f"mmqa_f1_score: {mmqa_f1_score}")
    logger.info(f"mmqa_em_score: {mmqa_em_score}")
    
    return output, mmqa_f1_scores, mmqa_em_scores, rec_score, f1_score, result_str


def define_args(parser):
    parser.add_argument('--mmqa_global_evi_dict_path',
                        type=str,
                        required=False,
                        default=""
                        )
    
    parser.add_argument('--mmqa_dev_data_path',
                        type=str,
                        required=False,
                        default="False"
                        )
    
    parser.add_argument('--mmqa_dev_rank_data_path',
                        type=str,
                        required=False,
                        default=""
                        )

    parser.add_argument('--mmqa_dev_RAMLLaMA_data_path',
                        type=str,
                        required=False,
                        default=""
                        )
    

def main(args):
    mmqa_global_evi_dict = read_json(args.mmqa_global_evi_dict_path)
    mmqa_dev_data = read_jsonl(args.mmqa_dev_data_path)
    mmqa_dev_rank_data = read_jsonl(args.mmqa_dev_rank_data_path)

    mmqa_dev_RAMLLaMA_data = read_jsonl(args.mmqa_dev_RAMLLaMA_data_path)
    output_dir = get_file_dir(args.mmqa_dev_RAMLLaMA_data_path)

    topks = [1,2,3,4,5]
    filter_types = ['PERQA', 'Normalize']
    best_f1 = 0.0
    for topk in topks:
        logger.info(f"====================== topk: {topk} ======================")
        for filter_type in filter_types:
            logger.info(f"################### FilterType: {filter_type} ################################")
            if filter_type == 'PERQA':
                thresholds = [0.025, 0.05, 0.075, 0.1, 1.5]
            elif filter_type == 'Normalize':
                thresholds = [0.025, 0.05, 0.075, 0.1]

            for threshold in thresholds:
                logger.info(f"========= th: {threshold} =========")
                _, _, mmqa_em_scores, _, f1_score, _ = mmqa_eval(
                                                                mmqa_RAMLLaMA_data = mmqa_dev_RAMLLaMA_data, 
                                                                mmqa_rank_data = mmqa_dev_rank_data, 
                                                                mmqa_ori_data = get_ori_mmqa_data(mmqa_dev_data),
                                                                mmqa_global_evi_dict = mmqa_global_evi_dict,
                                                                is_test = False,
                                                                topk=topk,
                                                                output_dir=output_dir,
                                                                IS_GENERATIVE_RETRIEVAL=True,
                                                                save_results=False,
                                                            )

                if f1_score > best_f1:
                    best_f1 = f1_score

    logger.info(f"best_f1: {best_f1}")
    logger.info(f"mmqa_em_scores: {mmqa_em_scores}")
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()
    logger = prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(get_file_dir(args.mmqa_dev_RAMLLaMA_data_path), "score_mmqa.log"))
    main(args)
    logger.info("All Done.")
