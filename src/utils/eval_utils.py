import numpy as np
from typing import List

import logging
logger = logging.getLogger()


def eval_for_each_question(
                            hits,
                            hits_1,
                            hits_relax,
                            all_count,
                            all_count_1,
                            topk_preds: List[int]=None,
                            top_doc_ids: set=None,
                            golden_evidence_set: List[str]=None,
                            debug: bool=False,                        
                        ):
    """
    Checkes if there is a hit among the topk envidence.
    """
    is_one = len(golden_evidence_set) == 1 # Check if one-hop evidence.
    if is_one:
        all_count_1 += 1

    evi_hits = golden_evidence_set.intersection(topk_preds)
    
    if len(evi_hits) > 0:
        hits_relax += 1
        if golden_evidence_set.issubset(topk_preds):
            hits += 1
            if is_one:
                hits_1 += 1

    recall = len(evi_hits) / len(golden_evidence_set)
    precision = len(evi_hits) / len(topk_preds)
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)
                
    all_count += 1

    return (
            hits,
            hits_1,
            hits_relax,
            recall,
            f1,
            all_count,
            all_count_1,
            )


def custom_eval(retrievl_results: List[dict]=None, 
                max_evidence: int=5,
                context_field: str='context',
                ) -> float:
    hits = 0
    hits_1 = 0
    hits_relax = 0
    recall_scores = []
    f1_scores = []
    all_count = 0
    all_count_1 = 0

    logger.debug(f"max_evidence: {max_evidence}")
    for item in retrievl_results:
        topk_preds = [pred['id'] for pred in item[context_field][:max_evidence]]
        
        golden_evidence_set = set([str(pos_evi['id']) for pos_evi in item['positive_ctxs']])

        logger.debug(f"len(topk_preds): {len(topk_preds)}")
        logger.debug(f"topk_preds: {topk_preds}")
        logger.debug(f"len(golden_evidence_set): {len(golden_evidence_set)}")
        logger.debug(f"golden_evidence_set: {golden_evidence_set}")
        
        hits, hits_1, hits_relax, recall, f1, all_count, all_count_1 = eval_for_each_question(
                                                                        hits,
                                                                        hits_1,
                                                                        hits_relax,
                                                                        all_count,
                                                                        all_count_1,
                                                                        topk_preds=topk_preds,
                                                                        golden_evidence_set=golden_evidence_set,                      
                                                                    )
        recall_scores.append(recall)
        f1_scores.append(f1)

    recall_scores = np.array(recall_scores)
    f1_scores = np.array(f1_scores)

    rec_score = recall_scores.mean()
    f1_score = f1_scores.mean()

    logger.info(
        "All examples evidence hit_strict ratio(recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits, all_count, hits / (all_count + 1e-6)
        )
    )

    logger.info(
        "One-hop evidence hit ratio(recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits_1, all_count_1, hits_1 / (all_count_1 + 1e-6)
        )
    )

    logger.info(
        "All examples evidence hit_relaxed ratio(recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits_relax, all_count, hits_relax / (all_count + 1e-6)
        )
    )

    logger.info(
        "F1 score = {:.3f}".format(
            f1_score
        )
    )

    logger.info(
        "Averaged Recall score = {:.3f}".format(
            rec_score
        )
    )

    return f1_score
