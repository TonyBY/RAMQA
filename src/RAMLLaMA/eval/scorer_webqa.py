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

import string
from tqdm import tqdm
from typing import List

from RAMQA.src.utils.data_utils import get_file_dir, read_jsonl, jsonl_to_json, read_json, save_json, json_to_jsonl

from RAMQA.src.RAMLLaMA.eval.BARTScore.bart_score import BARTScorer
from RAMQA.src.utils.webqa_eval import webqa_metrics_approx

np.set_printoptions(precision=4)

import argparse
from RAMQA.src.utils.args import prepare_logger
import logging
logger = logging.getLogger()




TABLE = str.maketrans(dict.fromkeys(string.punctuation))

def normalize_text_for_bart(x):  # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())


def compute_bartscore_ParaBank(c, a, switch=False, batch_size=64):
    logger.info(f"batch_size: {batch_size}")
    bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')

    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch:
        score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc, batch_size=batch_size))
    else:
        score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc, batch_size=batch_size))
    return score


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


def webqa_eval(webqa_RAMLLaMA_data: List[dict]=[], 
               webqa_rank_data: List[dict]=[], 
               webqa_ori_data: List[dict]=[],
               webqa_global_evi_dict: dict={},
               is_test: bool=False,
               topk: int=5,
               switch: bool=False,
               ep: float=10**-5,
               calculate_BARTscore=True,
               output_dir: str='',
               batch_size: int=4,
               IS_GENERATIVE_RETRIEVAL: bool=False,
               filter_type: str='PERQA',
               threshold: float=0.1,
               evi_sim_th: float=0.9,
               save_results: bool=True,
              ):

    assert len(webqa_RAMLLaMA_data) == len(webqa_rank_data) == len(webqa_ori_data), f"len(webqa_RAMLLaMA_data): {len(webqa_RAMLLaMA_data)}, len(webqa_rank_data): {len(webqa_rank_data)}, len(webqa_ori_data): {len(webqa_ori_data)}"
    
    output = {}
    result_dict = {}
    acc_scores = []
    rec_scores = []
    f1_scores = []
    golden_res_list = []
    gen_res_list = []
    
    pred_evi_number_list = []
    for i, item in tqdm(enumerate(zip(webqa_RAMLLaMA_data, webqa_rank_data, webqa_ori_data)), total=len(webqa_RAMLLaMA_data)):
        qa_item, rank_item, org_item = item

        queston_id = org_item['Guid']
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
                if idx in webqa_global_evi_dict:
                    evi = webqa_global_evi_dict[idx]
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
                if idx in webqa_global_evi_dict:
                    evi = webqa_global_evi_dict[idx]
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
            Qcate = org_item['Qcate']

            golden_answer = rank_item['answer_text'].strip()
            golden_res_list.append(golden_answer)

            golden_evi_ids = set([str(ctx['id']) for ctx in rank_item['positive_ctxs']])

            evi_hits = golden_evi_ids.intersection(pre_evi_ids)
            recall = len(evi_hits) / (len(golden_evi_ids) + ep)
            rec_scores.append(recall)
            precision = len(evi_hits) / (len(pre_evi_ids) + ep)
            f1 = (2 * recall * precision) / (recall + precision + ep)
            
            f1_scores.append(f1)

            if calculate_BARTscore:
                res_dict = webqa_metrics_approx(pred_answer, golden_answer, Qcate)
                accuracy = res_dict['acc_approx']
            else:
                accuracy = 0.0
            acc_scores.append(accuracy)

            result_dict[queston_id] = {'question': question, 
                                        'golden_ans': golden_answer, 
                                        'gen_ans': pred_answer,
                                        'pre_evi': pre_evi,
                                        'golden_id': list(golden_evi_ids),
                                        'evi_recall': recall,
                                        'evi_f1': f1,
                                        'ans_acc': accuracy,
                                        }
        
        pre_evi_ids =  [int(eid) if eid.isdigit() else eid for eid in list(pre_evi_ids)]
        output[queston_id] = {'answer': pred_answer, 'sources': list(pre_evi_ids)}
        
    if os.path.exists(output_dir) and save_results:
        if IS_GENERATIVE_RETRIEVAL:
            output_path = os.path.join(output_dir, f'official_results_GIR_topk-{topk}.json')
        else:
            output_path = os.path.join(output_dir, f'official_results_FilterType-{filter_type}_Th-{threshold}_topk-{topk}.json')
        logger.info(f"Saving official_results to: {output_path}")
        save_json(output, output_path)
        logger.info("Done.")
        
        
        result_path = os.path.join(output_dir, f'analysis_results_topk-{topk}.json')
        logger.info(f"Saving analysis_results to: {result_path}")
        save_json(result_dict, result_path)
        logger.info("Done.")
    
    acc_scores = np.array(acc_scores)
    rec_scores = np.array(rec_scores)
    f1_scores = np.array(f1_scores)
    
    if calculate_BARTscore:
        logger.info("Calculating BARTscore...")
        normalizer = compute_bartscore_ParaBank(golden_res_list, 
                                                golden_res_list, 
                                                batch_size=batch_size)
        
        BARTscore = compute_bartscore_ParaBank(gen_res_list, 
                                               golden_res_list, 
                                               switch=switch, 
                                               batch_size=batch_size) / np.array(normalizer)
        
        bart_scores = np.where(BARTscore > 1, 1, BARTscore)
        logger.info("Done.")
        
        QA_scores = bart_scores * acc_scores
        
        result_dict_list = json_to_jsonl(result_dict)
        for i, item in tqdm(enumerate(zip(bart_scores, QA_scores, result_dict_list)), total=len(result_dict_list)):
            bs, qas, res = item
            result_dict_list[i]['bart_score'] = bs
            result_dict_list[i]['QA_score'] = qas
            result_dict_list[i]['qid'] = res['ori_key']
            
        if save_results:
            result_dict = jsonl_to_json(result_dict_list, key_name='ori_key')
            if IS_GENERATIVE_RETRIEVAL:
                result_dict_path = os.path.join(output_dir, f'analysis_results_GIR_topk-{topk}.json')
            else:
                result_dict_path = os.path.join(output_dir, f'analysis_results_FilterType-{filter_type}_Th-{threshold}_topk-{topk}.json')
            logger.info(f"Saving result_dict to: {result_dict_path}")
            save_json(result_dict, result_dict_path)
            logger.info("Done.")
        
        QA_score = QA_scores.mean()
        
        bart_score = bart_scores.mean()
    else:
        bart_score = 0.0
        QA_score = 0.0
        
    pred_evi_number_list = np.array(pred_evi_number_list)
    avg_pred_evi_number = pred_evi_number_list.mean()
    
    acc_score = acc_scores.mean()
    rec_score = rec_scores.mean()
    f1_score = f1_scores.mean()
    result_str = json.dumps(result_dict, indent=4)
    return output, QA_score, acc_score, rec_score, f1_score, bart_score, result_str


def define_args(parser):
    parser.add_argument('--webqa_global_evi_dict_path',
                        type=str,
                        required=False,
                        default=""
                        )
    
    parser.add_argument('--webqa_val_data_path',
                        type=str,
                        required=False,
                        default="False"
                        )
    
    parser.add_argument('--webqa_test_data_path',
                        type=str,
                        required=False,
                        default=""
                        )

    parser.add_argument('--webqa_val_rank_data_path',
                        type=str,
                        required=False,
                        default=""
                        )
    
    parser.add_argument('--webqa_test_rank_data_path',
                        type=str,
                        required=False,
                        default=""
                        )
    
    parser.add_argument('--webqa_val_RAMLLaMA_data_path',
                        type=str,
                        required=False,
                        default=""
                        )
    
    parser.add_argument('--webqa_test_RAMLLaMA_data_path',
                        type=str,
                        required=False,
                        default=""
                        )


def main(args):
    # Tune over validation set to get the highest retireval F1 score.
    webqa_global_evi_dict = read_json(args.webqa_global_evi_dict_path)

    webqa_val_data = read_json(args.webqa_val_data_path)
    webqa_val_data = json_to_jsonl(webqa_val_data)

    webqa_test_data = read_json(args.webqa_test_data_path)
    webqa_test_data = json_to_jsonl(webqa_test_data)

    webqa_val_rank_data = read_jsonl(args.webqa_val_rank_data_path)
    webqa_test_rank_data = read_jsonl(args.webqa_test_rank_data_path)

    webqa_val_RAMLLaMA_data = read_jsonl(args.webqa_val_RAMLLaMA_data_path)
    output_dir = get_file_dir(args.webqa_val_RAMLLaMA_data_path)

    filter_types = ['PERQA', 'Normalize']
    topks = [1,2,3,4,5]

    best_f1 = 0.0
    best_filter_type = ''
    best_threshold = 0.0
    best_topk = 0

    for topk in topks:
        logger.info(f"====================== topk: {topk} ======================")
        for filter_type in filter_types:
            logger.info(f"################### FilterType: {filter_type} ################################")
            if filter_type == 'PERQA':
                thresholds = [0, 0.0005, 0.001, 0.002]
            elif filter_type == 'Normalize':
                thresholds = [0, 0.0001, 0.0005, 0.001]

            for threshold in thresholds:
                logger.info(f"========= th: {threshold} =========")
                _, _, _, _, f1_score, _, _ = webqa_eval(
                                webqa_RAMLLaMA_data = webqa_val_RAMLLaMA_data, 
                                webqa_rank_data = webqa_val_rank_data[:len(webqa_val_RAMLLaMA_data)], 
                                webqa_ori_data = webqa_val_data[:len(webqa_val_RAMLLaMA_data)],
                                webqa_global_evi_dict=webqa_global_evi_dict,
                                is_test = False,
                                filter_type=filter_type,
                                threshold=threshold,
                                switch=False,
                                topk=topk,
                                output_dir=output_dir,
                                calculate_BARTscore=False,
                                IS_GENERATIVE_RETRIEVAL=True,
                                save_results=True,
                                )

                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_topk=topk
                    best_filter_type = filter_type
                    best_threshold = threshold

    logger.info(f"best_f1: {best_f1}")
    logger.info(f"best_filter_type: {best_filter_type}") 
    logger.info(f"best_threshold: {best_threshold}") 
    logger.info(f"bestbest_topk_f1: {best_topk}")

    # Get the QA score for the validation set.
    logger.info('################  Scores for the validation set  ##############################')
    _, QA_score, acc_score, rec_score, f1_score, bart_score, _ = webqa_eval(
        webqa_RAMLLaMA_data = webqa_val_RAMLLaMA_data, 
        webqa_rank_data = webqa_val_rank_data[:len(webqa_val_RAMLLaMA_data)], 
        webqa_ori_data = webqa_val_data[:len(webqa_val_RAMLLaMA_data)],
        webqa_global_evi_dict=webqa_global_evi_dict,
        is_test = False,
        switch=False,
        topk=best_topk,
        output_dir=output_dir,
        calculate_BARTscore=True,
        IS_GENERATIVE_RETRIEVAL=True,
        save_results=True,
        )
    logger.info(f"bart_score: {bart_score}")
    logger.info(f"QA_score: {QA_score}")
    logger.info(f"rec_score@{best_topk}: {rec_score}")
    logger.info(f"f1_score@{best_topk}: {f1_score}")
    logger.info(f"acc_score: {acc_score}")
    
    # Get the official submission format for the testing set.
    webqa_test_RAMLLaMA_data = read_jsonl(args.webqa_test_RAMLLaMA_data_path)

    _, QA_score, acc_score, rec_score, f1_score, bart_score, _ = webqa_eval(
        webqa_RAMLLaMA_data = webqa_test_RAMLLaMA_data, 
        webqa_rank_data = webqa_test_rank_data, 
        webqa_ori_data = webqa_test_data,
        webqa_global_evi_dict=webqa_global_evi_dict,
        calculate_BARTscore = False,
        is_test = True,
        IS_GENERATIVE_RETRIEVAL=True,
        topk=best_topk,
        filter_type=best_filter_type,
        threshold=best_threshold,
        evi_sim_th=0.9,
        output_dir=get_file_dir(args.webqa_test_RAMLLaMA_data_path),
        save_results=True,
        )
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args() 
    logger = prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(get_file_dir(args.webqa_test_RAMLLaMA_data_path), "score_webqa.log"))
    main(args)
    logger.info("All Done.")
