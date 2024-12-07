import json
from logging import Logger

import jsonlines
from typing import List, TextIO, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import functools
import time
import os
import math
import pickle5 as pickle
import torch
import re
import random
import unicodedata
from torch import Tensor as T

LOGGER_NAME = 'DSR-FC'
logger = logging.getLogger(LOGGER_NAME)


def init_logger(verbose: bool = False, log_file: str = ''):
    logging.getLogger().setLevel(logging.DEBUG)
    logger: Logger = logging.getLogger(LOGGER_NAME)

    if len(logger.handlers):
        pass
    else:
        # log.info will always be show in console
        # log.debug will also be shown when verbose flag is set
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        if log_file != '':
            log_file_dir = '/'.join(log_file.split('/')[:-1])
            make_directory(log_file_dir)
            f_handler = logging.FileHandler(log_file)
            logger.addHandler(f_handler)
            logger.info("file handler added.")
    return logger


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def load_data_to_list(data_path: str=None) -> List[dict]:
    if data_path.endswith('.json'):
        data = json_to_jsonl(read_json(data_path))
    elif data_path.endswith('.pkl'):
        data = load_pickle(data_path)
    elif data_path.endswith('.jsonl'):
        data = read_jsonl(data_path)
    else:
        raise Exception(f"Unknown data format: {data_path}")
    return data


def load_data_to_dict(data_path: str=None) -> List[dict]:
    if data_path.endswith('.json'):
        data = read_json(data_path)
    elif data_path.endswith('.pkl'):
        data = load_pickle(data_path)
    else:
        raise Exception(f"Unknown data format: {data_path}")
    return data


def save_json(json_obj: dict, output_path: str) -> None:
    jsonFile: TextIO = open(output_path, 'w')
    jsonFile.write(json.dumps(json_obj, indent=4, sort_keys=False))
    jsonFile.close


def read_json(json_path: str) -> object:
    with open(json_path, 'r') as f:
        return json.load(f)


@timer
def save_jsonl(items: List[dict], output_path: str = '') -> None:
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(items)


@timer
def read_jsonl(jsonl_path: str) -> List[dict]:
    with jsonlines.open(jsonl_path) as f:
        items = []
        for line in f.iter():
            items.append(line)
    return items

def json_to_jsonl(json_data: dict,
                  key_name: str='ori_key',):
    output = []
    for k in json_data:
        item = json_data[k]
        item[key_name] = k
        output.append(item)
    return output

def jsonl_to_json(item_list: List[dict],
                  key_name: str='ori_key',
                  ):
    new_dict = {}
    for item in item_list:
        key = item[key_name]
        new_dict[key] = item
    return new_dict

@timer
def read_lines_from_file(filename: str) -> List[str]:
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


@timer
def save_npy(embeddings: np.ndarray, vectors_file: str = '') -> None:
    np.save(vectors_file, embeddings)


@timer
def read_npy(npy_path: str) -> np.ndarray:
    items = np.load(npy_path).astype('float32')
    return items


def write_lines_to_file(lines: list, out_file: str) -> None:
    with open(out_file, 'w') as fp:
        for line in lines:
            fp.write("%s\n" % line)


def merge_dpr_data(batch_json_template: str, number_of_df_batches: int) -> List[dict]:
    output_dpr_data_list = []
    for i in range(0, number_of_df_batches):
        dpr_data_batch = read_json(batch_json_template.replace('$BATCH_NUMBER', str(i)))
        output_dpr_data_list.extend(dpr_data_batch)
    return output_dpr_data_list


def drop_empty_rows_by_column(df, column: str) -> pd.DataFrame:
    df = df.reset_index()
    index_to_drop = []
    drop_line_cnt = 0
    for index, row in tqdm(df.iterrows()):
        if row[column] == '' \
                or str(row[column]).lower() == 'none' \
                or pd.isna(row[column]) \
                or str(row[column]) == '[]' \
                or row[column] == []:
            index_to_drop.append(index)
            drop_line_cnt += 1

    df = df.drop(index=index_to_drop, inplace=False)
    return df


def zoom_in(df, column):
    logger.info(f"length: {len(df)}")
    return df.loc[df.index, [column]].iloc[:10].style.set_properties(subset=[column], **{'width-min': '300px'})


def recall(true_logs: List[str], pred_logs: List[str]) -> float:
    if len(true_logs) == 0 or len(pred_logs) == 0:
        return 0
    true_log_set = set(true_logs)
    score = 0
    for pre_log in pred_logs:
        if pre_log in true_log_set:
            score += 1
    return float(score / len(true_logs))


def precision(true_logs: List[str], pred_logs: List[str]) -> float:
    if len(true_logs) == 0 or len(pred_logs) == 0:
        return 0.0
    true_log_set = set(true_logs)
    score = 0
    for pre_log in pred_logs:
        if pre_log in true_log_set:
            score += 1
    return float(score / len(pred_logs))


def F1(recall_score, precision_score) -> float:
    if (recall_score + precision_score) != 0:
        return 2.0 * recall_score * precision_score / (recall_score + precision_score)
    else:
        return 0.0


def recall_precision_F1(true_logs: List[str], pred_logs: List[str]) -> float:
    recall_score = recall(true_logs, pred_logs)
    precision_score = precision(true_logs, pred_logs)
    f1_score = F1(recall_score, precision_score)
    return recall_score, precision_score, f1_score


def get_raw(docid: str, corpus_dir: str = './data/tweets/') -> str:
    path = corpus_dir + docid + '.json'
    json_obj = read_json(path)
    return json_obj['contents'].split('\n')[-1]


@timer
def load_file_as_df(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading df from: {file_path}")
    if file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path)
    elif file_path.split('.')[-1] == 'pkl':
        df = pd.read_pickle(file_path)
    else:
        raise Exception("Error, this function only loads .csv and .pkl files.")
    logger.info("Done.")
    return df


@timer
def save_df_to_file(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    logger.info(f"Saving df to: {file_path}")
    if file_path.split('.')[-1] == 'csv':
        df.to_csv(file_path, sep=',', index=False)
    elif file_path.split('.')[-1] == 'pkl':
        df.to_pickle(file_path)
    else:
        raise Exception("Error, this function only loads .csv and .pkl files.")
    logger.info("Done.")


def get_file_dir(file_path: str) -> str:
    return '/'.join(file_path.split('/')[:-1])


def get_file_name(file_path: str) -> str:
    return '.'.join(file_path.split('/')[-1].split('.')[:-1])


def make_directory(dir: str) -> str:
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def merge_mp_results(results: List) -> List[dict]:
    output_data = []
    for res in results:
        output_data.extend(res.get(timeout=1))
    return output_data


def split_list(data: List[object], chunk_num: int = 3) -> List[object]:
    chunk_size = math.ceil(len(data) / chunk_num)
    chunks = [data[x:x+chunk_size] for x in range(0, len(data), chunk_size)]

    assert sum([len(chunk) for chunk in chunks]) == len(data)
    return chunks


def create_bash_script_for_each_data_chunk(content_template:str, 
                                           bash_dir:str, 
                                           output_path_template:str, 
                                           num_chunks:int)->None:
    for i in range(0, num_chunks):
        content = content_template.replace('CHUNK_NUMBER_PLACE_HOLDER', str(i))
        output_file = f"{bash_dir}/{output_path_template.replace('CHUNK_NUMBER_PLACE_HOLDER', str(i))}"
        with open(output_file, 'w') as fp:
            fp.write(content)


def merge_data_chunks(data_chunk_path_template: str = None, 
                      number_of_chunks: int = None):
    
    all_data = []
    for i in tqdm(range(number_of_chunks), desc='Merging data chunks...'):
        cache_path = data_chunk_path_template.replace('CHUNK_NUMBER_PLACE_HOLDER', str(i))
        logger.info(f"Loading cached data_list from: {cache_path}")
        data_list = load_pickle(cache_path)
        logger.info('Done.')
        logger.info(f"len(data_list): {len(data_list)}")

        all_data.extend(data_list)
        logger.info(f"len(all_data): {len(all_data)}")
    
    output_path = data_chunk_path_template.replace("CHUNK_NUMBER_PLACE_HOLDER", f"all_{len(all_data)}")
    logger.info(f"Saving all_data of length {len(all_data)} to: {output_path}")
    dump_pickle(all_data, output_path)
    logger.info(f"Done.")


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def process_evid(sentence):
    sentence = convert_to_unicode(sentence)
    sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
    sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
    sentence = re.sub("-LRB-", "(", sentence)
    sentence = re.sub("-RRB-", ")", sentence)
    sentence = re.sub("-COLON-", ":", sentence)
    sentence = re.sub("_", " ", sentence)
    sentence = re.sub("\( *\,? *\)", "", sentence)
    sentence = re.sub("\( *[;,]", "(", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


@timer
def load_pickle(pkl_path: str) -> Any:
    with open(pkl_path, 'rb') as file:
        output_obj = pickle.load(file)
    return output_obj


@timer
def dump_pickle( data: Any, pkl_path: str) -> None:
    with open(pkl_path, 'wb') as file:
        pickle.dump(data, file)


@timer
def load_docids(docid_path: str) -> List[str]:
        id_f = open(docid_path, 'r')
        docids = [line.rstrip() for line in id_f.readlines()]
        id_f.close()
        return docids


def pred_format_webqa_to_fever(data: List[dict], 
                               singleHopNumbers: int=20,
                              ) -> List[dict]:
    
    if 'multihop_context' in data[0]:
        logger.info(f"singleHopNumbers: {singleHopNumbers}")
        
    fever_data = []
    for item in tqdm(data, desc='Transforming data format from pyserini to fever...'):
        fever_item = {"id": item['Guid'],
                      "label": item['A'],
                      "evidence": [img_pos['image_id'] for img_pos in item['img_posFacts']] + [txt_pos['snippet_id'] for txt_pos in item['txt_posFacts']],
                      "predicted_label": "",
                      "predicted_evidence": [],
                     }
        
        viewed_sentence_ids = set()

        reranked_evi = []
        reranked_evi = []

        if 'merged_retrieval' in item:
            context = item['merged_retrieval']
        else:
            if 'reranked_context' in item:
                reranked_evi = item['reranked_context']
            elif 'multihop_context' in item:
                reranked_evi = item['context'][:singleHopNumbers]
        
        if reranked_evi != []:
            reranked_evi = [evi['id'] for evi in reranked_evi]
            # logger.info(f"reranked_evi: {reranked_evi}")
            fever_item['predicted_evidence'].extend(reranked_evi)
            viewed_sentence_ids.update(reranked_evi)
                    
        if 'multihop_context' in item:
            context = item['multihop_context']
        elif 'context' in item:
            context = item['context']
        else:
            context=reranked_evi
        
        logger.debug(f"len(viewed_sentence_ids): {len(viewed_sentence_ids)}")
        logger.debug(f"len(reranked_evi): {len(reranked_evi)}")
        logger.debug(f"len(context): {len(context)}")
        for i in range(len(context) - len(reranked_evi)):
            evi = context[i]
            if evi['id'] in viewed_sentence_ids:
                continue

            fact_id = evi['id']
            fever_item['predicted_evidence'].append(fact_id)

        logger.debug(f"len(fever_item['predicted_evidence']): {len(fever_item['predicted_evidence'])}")
        fever_data.append(fever_item)

    return fever_data


def pred_format_webqa_to_reranking(data: List[dict],
                                   is_test: bool=False,
                                   )-> List[dict]:
    for idx in tqdm(range(len(data)), desc='Transforming data format from webqa to reranking...'):
        if is_test:
            txt_Facts = data[idx]['txt_Facts']
            img_Facts = data[idx]['img_Facts']
        else:
            txt_Facts = data[idx]['txt_posFacts'] + data[idx]['txt_negFacts']
            img_Facts = data[idx]['img_posFacts'] + data[idx]['img_negFacts']

        evidence_list = [str(img_fact['image_id']) for img_fact in img_Facts] + [str(txt_fact['snippet_id']) for txt_fact in txt_Facts]
        logger.info(f"len(evidence_list): {len(evidence_list)}")
        random.shuffle(evidence_list)
        
        data[idx]['context'] = evidence_list

        question_text = data[idx]['Q']
        if question_text[0] == '"':
            question_text = question_text[1:-1]
        data[idx]['claim'] = question_text
    return data


def get_sentence_by_id(sentId: str, wikiSentence: dict) -> str:
    try:
        sentene_text = wikiSentence[sentId]
    except:
        try:
            sentene_text = wikiSentence[unicodedata.normalize('NFC', sentId)]
        except Exception as e:
            logger.info(f"Warning: {e} Cannot get sentence_text from the wikiSentence dictionary with sentId: {sentId}.")
            return ''
    return sentene_text


def get_textural_fact_by_id(factId: str, evid_dict_all: dict) -> str:
    data_type = evid_dict_all[factId]['data_type']
    if data_type == 'txt':
        return evid_dict_all[factId]['title'] + " -- " + evid_dict_all[factId]['content'], data_type
    elif data_type == 'img':
        return evid_dict_all[factId]['content'], data_type
    else:
        raise Exception("Error: Unknown data type {data_type}.")


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_pad_id(tokenizer) -> int:
        return tokenizer.pad_token_id


def get_attn_mask(pad_token_id:int, tokens_tensor: T) -> T:
    return (tokens_tensor != pad_token_id).long()


def get_multihop_data(data_list: List[dict]=None, level: int=None) -> List[dict]:
    multi_hop_examples = []
    for item in tqdm(data_list):
        if item['label'] == "NOT ENOUGH INFO":
            continue
            
        if level == 0 and any([len(eg) > 1 for eg in item['evidence']]):
            multi_hop_examples.append(item)
            
        elif level == 1 and any([len(eg) > 1 for eg in item['evidence']]):
            for eg in item['evidence']:
                if len(eg) > 1:
                    diff_doc_set = set()
                    for e in eg:
                        diff_doc_set.add(e[2])
                    if len(diff_doc_set) > 1:
                        multi_hop_examples.append(item)
                        break
        elif level == 2 and all([len(eg) > 1 for eg in item['evidence']]):
            multi_hop_examples.append(item)
        
        elif level == 3 and all([len(eg) > 1 for eg in item['evidence']]):
            for eg in item['evidence']:
                diff_doc_num_list = []
                diff_doc_set = set()
                for e in eg:
                    diff_doc_set.add(e[2])
                diff_doc_num_list.append(len(diff_doc_set))
                if min(diff_doc_num_list) > 1:
                    multi_hop_examples.append(item)
                    break
    return multi_hop_examples

def get_multiHop_and_singleHop_data(data_all: List[dict]) -> Tuple[List[dict], 
                                                                   List[dict], 
                                                                   List[dict], 
                                                                   List[dict]]:
    data_multi_strict_level3 = []
    data_multi_strict_level2 = []
    data_multi_strict_level1 = []
    data_multi_strict_level0 = []
    data_strict_single = []
    data_loose_single = []
    
    for i in tqdm(range(len(data_all))):
        item = data_all[i]
     
        if item['label'] == 'NOT VERIFIABLE':
            continue
        if any([len(eg) == 1 for eg in item['evidence']]):
            data_loose_single.append(item)

        if all([len(eg) == 1 for eg in item['evidence']]):
            data_strict_single.append(item)

        if any([len(eg) > 1 for eg in item['evidence']]):
            data_multi_strict_level0.append(item)

        if any([len(eg) > 1 for eg in item['evidence']]):
            for eg in item['evidence']:
                if len(eg) > 1:
                    diff_doc_set = set()
                    for e in eg:
                        diff_doc_set.add(e[2])
                    if len(diff_doc_set) > 1:
                        data_multi_strict_level1.append(item)
                        break

        if all([len(eg) > 1 for eg in item['evidence']]):
            data_multi_strict_level2.append(item)
            
        if all([len(eg) > 1 for eg in item['evidence']]):
            for eg in item['evidence']:
                diff_doc_num_list = []
                diff_doc_set = set()
                for e in eg:
                    diff_doc_set.add(e[2])
                diff_doc_num_list.append(len(diff_doc_set))
                if min(diff_doc_num_list) > 1:
                    data_multi_strict_level3.append(item)
                    break
        
    return data_loose_single, data_strict_single, data_multi_strict_level0, data_multi_strict_level1, data_multi_strict_level2, data_multi_strict_level3


def get_label_num(label: str) -> int:
    if label.upper() in ["REFUTES", "CONTRADICT"]:
        return 0
    if label.upper() in ["SUPPORT", "SUPPORTS", "ENTAILMENT"]:
        return 2
    return 1


def format_trans_webqa_to_ranking(data_list: List[dict],
                               is_test: bool = False,
                              ) -> List[dict]:
    output = []
    for item in tqdm(data_list, desc='Transforming data format from WebQA to Ranking...'):
        qid = item['Guid']
        question = item['Q'].strip('"')
        
        
        if is_test:
            txt_context = [{'id': tf['snippet_id'], 
                            'title': tf['title'],
                            'text': tf['fact'],
                            'data_type': 'txt',
                            } for tf in item['txt_Facts']]
            
            img_context = [{'id': imf['image_id'], 
                            'title': imf['caption'],
                            'text': '',
                            'data_type': 'img',
                            } for imf in item['img_Facts']]
            
            output.append({'qid': qid,
                           'question': question,
                           'txt_context': txt_context,
                           'img_context': img_context,
                          }
                         )
            
        else:
            txt_neg_context = [{'id': nf['snippet_id'], 
                                'title': nf['title'],
                                'text': nf['fact'],
                                'data_type': 'txt',
                                } for nf in item['txt_negFacts']]

            txt_pos_context = [{'id': pf['snippet_id'], 
                                'title': pf['title'],
                                'text': pf['fact'],
                                'data_type': 'txt',
                                } for pf in item['txt_posFacts']]

            img_neg_context = [{'id': nf['image_id'], 
                                'title': nf['caption'],
                                'text': '',
                                'data_type': 'img',
                                } for nf in item['img_negFacts']]

            img_pos_context = [{'id': pf['image_id'], 
                                'title': pf['caption'],
                                'text': '',
                                'data_type': 'img',
                                } for pf in item['img_posFacts']]
            
            
            Qcate = item['Qcate']
            answer_text = item['A'][0].strip('"')
            
            output.append({'qid': qid,
                           'Qcate': Qcate,
                           'question': question,
                           'answer_text': answer_text,
                           'txt_neg_context': txt_neg_context,
                           'txt_pos_context': txt_pos_context,
                           'img_neg_context': img_neg_context,
                           'img_pos_context': img_pos_context,
                          }
                         )
    return output


def format_trans_mmqa_to_ranking(data_list: List[dict],
                                 use_mixed_txt_img: bool = False,
                                 text_corpus_dict: dict = {},
                                 image_corpus_dict: dict = {},
                              ) -> List[dict]:
    output = []
    for item in tqdm(data_list, desc='Transforming data format from MMQA to Ranking...'):
        if item['metadata']['modalities'] == ['image'] or item['metadata']['modalities'] == ['text']:
            pass
        elif use_mixed_txt_img and (item['metadata']['modalities'] == ['text', 'image'] or item['metadata']['modalities'] == ['image', 'text']):
            pass
        else:
            continue
    
        qid = item['qid']
        
        question = item['question']
        answer_text = item['answers'][0]['answer']
    
        supporting_img_doc_ids = []
        supporting_txt_doc_ids = []
        for sc in item['supporting_context']:
            if sc['doc_part'] == 'image':
                supporting_img_doc_ids.append(sc['doc_id'])
            elif sc['doc_part'] == 'text':
                supporting_txt_doc_ids.append(sc['doc_id'])
            else:
                print(f"item: {sc['item']}")
                raise Exception(f"Unexpectied data type: {sc['doc_part']}")
        
        img_pos_context = []
        for evi_id in set(supporting_img_doc_ids):
            img_pos_context.append({'id': evi_id, 
                                    'title': image_corpus_dict[evi_id]['title'],
                                    'text': "",
                                    'data_type': 'img',
                                    'path': image_corpus_dict[evi_id]['path'],
                                    }
                                  )
            
        txt_pos_context = []
        for evi_id in set(supporting_txt_doc_ids):
            txt_pos_context.append({'id': evi_id, 
                                    'title': text_corpus_dict[evi_id]['title'],
                                    'text': text_corpus_dict[evi_id]['text'],
                                    'data_type': 'txt',
                                    }
                                  )
            
        neg_img_ids = set(item['metadata']['image_doc_ids']) - set(supporting_img_doc_ids)
        neg_txt_ids = set(item['metadata']['text_doc_ids']) - set(supporting_txt_doc_ids)
        
        img_neg_context = []
        for evi_id in neg_img_ids:
            img_neg_context.append({'id': evi_id, 
                                    'title': image_corpus_dict[evi_id]['title'],
                                    'text': "",
                                    'data_type': 'img',
                                    'path': image_corpus_dict[evi_id]['path'],
                                    }
                                  )
            
        txt_neg_context = []
        for evi_id in neg_txt_ids:
            txt_neg_context.append({'id': evi_id, 
                                    'title': text_corpus_dict[evi_id]['title'],
                                    'text': text_corpus_dict[evi_id]['text'],
                                    'data_type': 'txt',
                                    }
                                  )

        output.append({'qid': qid,
                       'question': question,
                       'answer_text': answer_text,
                       'txt_neg_context': txt_neg_context,
                       'txt_pos_context': txt_pos_context,
                       'img_neg_context': img_neg_context,
                       'img_pos_context': img_pos_context,
                      }
                     )
        
    return output
