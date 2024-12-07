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
sys.path.append(pwd)
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import logging
import numpy as np
from PIL import Image
from peft import TaskType
import random
from tqdm import tqdm
from typing import List
import zipfile

import torch
from torch import autocast
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')

from transformers import (
                            AutoProcessor,
                            LlavaForConditionalGeneration,
                            logging as tf_logging,
                        )
from transformers.utils.peft_utils import ADAPTER_WEIGHTS_NAME
from transformers.trainer import _is_peft_model
tf_logging.set_verbosity_error()

from src.utils.config import parser
from src.utils.args import prepare_logger
from src.utils.data_utils import make_directory, read_json, read_jsonl, save_jsonl, move_to_device, get_file_name
from src.RankLLaVA.models.RankLLaVA_model import RankLLaVA
from src.utils.model_utils import find_all_linear_names
from src.utils.eval_utils import custom_eval

args = parser.parse_args()
logger = logging.getLogger()


def get_topk_reranked_evidence_right_pad(model=None,
                                        model_inputs: List[dict] = None, 
                                        batch_size: int=None,
                                        pad_token_id: int=None,
                                        device=None,
                                        ):
    """
    Output: 
        topk_preds: a list of index number of evidence in the candidate sentence set.
        score_criteria: a tensor of scores of the reranked sentences.
    """
    i = 0
    all_preds = []
    model.eval()
    logger.info("Doing Right Padding....")
    with autocast(dtype=torch.bfloat16, device_type="cuda"):
        with torch.no_grad():
            while i < len(model_inputs):                
                batch_inputs = model_inputs[i : i + batch_size]

                batch_input_ids = []
                batch_attention_masks = []
                batch_pixel_values = []
                for inputs in batch_inputs:
                    input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    pixel_values = inputs['pixel_values']
                    
                    batch_input_ids.append(input_ids.view(-1))
                    batch_attention_masks.append(attention_mask.view(-1))
                    batch_pixel_values.append(pixel_values)

                    logger.debug(f"input_ids.shape: {input_ids.shape}")
                    logger.debug(f"attention_mask.shape: {attention_mask.shape}")
                    logger.debug(f"pixel_values.shape: {pixel_values.shape}")

                logger.debug(f"len(batch_input_ids): {len(batch_input_ids)}")
                logger.debug(f"len(batch_attention_masks): {len(batch_attention_masks)}")
                logger.debug(f"len(batch_pixel_values): {len(batch_pixel_values)}")

                batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
                batch_attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=0)
                batch_pixel_values = torch.cat(batch_pixel_values, dim=0)

                collated_batch_inputs =  {'input_ids':  batch_input_ids,
                                          'attention_mask': batch_attention_masks,
                                          'pixel_values': batch_pixel_values,
                                          'is_test': True,
                                        }
              
                preds = model(**move_to_device(collated_batch_inputs, device)).logits.float().detach().cpu()
                logger.debug(f"preds.size(): {preds.size()}")
                logger.debug(f"preds: {preds}")
                
                del collated_batch_inputs
                
                all_preds += [preds]
                i = i + batch_size

    # sofrmaxt scores for each label of each evidence.           
    all_preds = torch.cat(all_preds, dim=0)
    logger.debug(f"all_preds.size(): {all_preds.size()}")
    logger.debug(f"all_preds: {all_preds}")
    softmax_scores = F.softmax(all_preds, dim=1)
    logger.debug(f"softmax_scores.size(): {softmax_scores.size()}")
    logger.debug(f"softmax_scores: {softmax_scores}")
    
    if args.num_labels == 3:
        score_criteria = 1 - softmax_scores[:, 1]
    else:
        score_criteria = softmax_scores[:, 1]
    logger.debug(f"score_criteria: {score_criteria}")

    topk_preds = torch.argsort(score_criteria, descending=True)
    logger.debug(f"topk_preds: {topk_preds}")

    return topk_preds, score_criteria


def get_topk_reranked_evidence_left_pad(model=None,
                                        model_inputs: List[dict] = None, 
                                        batch_size: int=None,
                                        pad_token_id: int=None,
                                        device=None,
                                        ):
    """
    Output: 
        topk_preds: a list of index number of evidence in the candidate sentence set.
        score_criteria: a tensor of scores of the reranked sentences.
    """
    i = 0
    all_preds = []
    model.eval()
    logger.info("Doing Left Padding....")
    with autocast(dtype=torch.bfloat16, device_type="cuda"):
        with torch.no_grad():
            while i < len(model_inputs):                
                batch_inputs = model_inputs[i : i + batch_size]

                flipped_batch_ctx_input_ids = []
                flipped_batch_ctx_attention_masks = []
                batch_pixel_values = []
                for inputs in batch_inputs:
                    input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    pixel_values = inputs['pixel_values']
                    
                    flipped_batch_ctx_input_ids.append(input_ids.view(-1).flip(dims=[0]))
                    flipped_batch_ctx_attention_masks.append(attention_mask.view(-1).flip(dims=[0]))
                    batch_pixel_values.append(pixel_values)

                    logger.debug(f"input_ids.shape: {input_ids.shape}")
                    logger.debug(f"attention_mask.shape: {attention_mask.shape}")
                    logger.debug(f"pixel_values.shape: {pixel_values.shape}")

                logger.debug(f"len(flipped_batch_ctx_input_ids): {len(flipped_batch_ctx_input_ids)}")
                logger.debug(f"len(flipped_batch_ctx_attention_masks): {len(flipped_batch_ctx_attention_masks)}")
                logger.debug(f"len(batch_pixel_values): {len(batch_pixel_values)}")

                batch_input_ids = pad_sequence(flipped_batch_ctx_input_ids, batch_first=True, padding_value=pad_token_id).flip(dims=[1])
                batch_attention_masks = pad_sequence(flipped_batch_ctx_attention_masks, batch_first=True, padding_value=0).flip(dims=[1])
                batch_pixel_values = torch.cat(batch_pixel_values, dim=0)

                collated_batch_inputs =  {'input_ids':  batch_input_ids,
                                          'attention_mask': batch_attention_masks,
                                          'pixel_values': batch_pixel_values,
                                          'is_test': True,
                                        }
                logger.debug(f"collated_batch_inputs: {collated_batch_inputs}")
                
                preds = model(**move_to_device(collated_batch_inputs, device)).logits.float().detach().cpu()
                logger.debug(f"preds.size(): {preds.size()}")
                logger.debug(f"preds: {preds}")
                
                del collated_batch_inputs
                
                all_preds += [preds]
                i = i + batch_size

    # sofrmaxt scores for each label of each evidence.           
    all_preds = torch.cat(all_preds, dim=0)
    logger.debug(f"all_preds.size(): {all_preds.size()}")
    logger.debug(f"all_preds: {all_preds}")
    softmax_scores = F.softmax(all_preds, dim=1)
    logger.debug(f"softmax_scores.size(): {softmax_scores.size()}")
    logger.debug(f"softmax_scores: {softmax_scores}")
    
    if args.num_labels == 3:
        score_criteria = 1 - softmax_scores[:, 1]
    else:
        score_criteria = softmax_scores[:, 1]
    logger.debug(f"score_criteria: {score_criteria}")

    topk_preds = torch.argsort(score_criteria, descending=True)
    logger.debug(f"topk_preds: {topk_preds}")

    return topk_preds, score_criteria


def eval(args, 
         output_path: str=None,
         ):
    ##################################################################################################################################
    ##### Prepare Model  #############################################################################################################
    ##################################################################################################################################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed evaluation %r",
                device, n_gpu, bool(args.local_rank != -1))
    
    if n_gpu > 0:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        compute_dtype = torch.float32

    bnb_model_from_pretrained_args = {}
    if args.bits in [4, 8] and n_gpu > 0:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map = 'auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                attn_implementation="flash_attention_2",
                llm_int8_skip_modules=["mm_projector", "multi_modal_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type # {'fp4', 'nf4'}
            )
        ))
    logger.info(f"bnb_model_from_pretrained_args: {bnb_model_from_pretrained_args}")

    back_bone_model = LlavaForConditionalGeneration.from_pretrained(args.model_type, 
                                                                    **bnb_model_from_pretrained_args,
                                                                    )

    back_bone_model.config.text_config.use_cache = False
    back_bone_model.config.text_config.pretraining_tp = 1

    if args.bits in [4, 8] and n_gpu > 0:
        from peft import prepare_model_for_kbit_training
        back_bone_model.config.torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        model = RankLLaVA(
                            back_bone_model=back_bone_model,
                            num_labels=args.num_labels,
                        )
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        model = RankLLaVA(
                            back_bone_model=back_bone_model,
                            num_labels=args.num_labels,
                        )
    
    logger.info(f"model.config: {model.config}")
    logger.info(f"model.config.text_config.use_cache: {model.config.text_config.use_cache}")
    logger.info(f"model.config.text_config.pretraining_tp: {model.config.text_config.pretraining_tp}")
    logger.info(f"model.model.config: {model.model.config}")
    logger.info(f"model.model.config.text_config.use_cache: {model.model.config.text_config.use_cache}")
    logger.info(f"model.model.config.text_config.pretraining_tp: {model.model.config.text_config.pretraining_tp}")

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    target_modules = find_all_linear_names(model)
    logger.info(f"target_modules: {target_modules}")

    if args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.SEQ_CLS,
        )
        if args.bits == 16:
            if torch.cuda.is_bf16_supported():
                model.to(torch.bfloat16)
            else:
                model.to(torch.float16)

        logger.info("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.bits in [4, 8] and n_gpu > 0:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if torch.cuda.is_bf16_supported():
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if torch.cuda.is_bf16_supported() and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if _is_peft_model(model) and args.init_checkpoint:
        # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
            if os.path.exists(args.checkpoint_path):
                logger.info("model.load_adapter....")
                model.load_adapter(args.checkpoint_path, model.active_adapter, is_trainable=True)
            else:
                logger.warning(
                    "The intermediate checkpoints of PEFT may not be saved correctly, "
                    f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                    "Check some examples here: https://github.com/huggingface/peft/issues/96"
                )
        else:
            logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")

    model.to(device)
    logger.debug(f"model.device: {model.device}")
    ##################################################################################################################################
    ##### Making Inference  ##########################################################################################################
    ##################################################################################################################################

    first_hop_search_results = read_jsonl(args.reranking_test_file)
    if args.debug:
        first_hop_search_results = first_hop_search_results[:100]

    processor = AutoProcessor.from_pretrained(args.model_type)
    sample_num = processor.image_processor.resample
    h, w = processor.image_processor.crop_size.values()
    pad_token_id=processor.tokenizer.pad_token_id

    text_corpus_dict = read_json(args.text_corpus_path)
    image_corpus_dict = read_json(args.image_corpus_path)

    print(f"Start ranking sentences in the top docs for each claim.")
    first_hop_reranking_results = []
    for item in tqdm(first_hop_search_results, desc=f"Evaluating Multi-Modal Reranker: {args.model_type}"):    
        question = item['question']

        evidence_list = item['context']

        logger.info(f"len(evidence_list): {len(evidence_list)}")
        logger.debug(f"evidence_list: {evidence_list}")
        
        model_inputs = []
        for evi in evidence_list:
            evi_id = evi['id']
            evi_type = evi['data_type']

            evi_title = evi['title']
            if evi_title:
                evi_text = evi_title + ' . ' + evi['text']
            else:
                evi_text = evi['text']

            logger.debug(f"evi_title: {evi_title}")
            logger.debug(f"evi_text: {evi_text}")

            if evi_type != 'img':
                inputs =  (str(question), None, str(evi_text))
            else:
                try:
                    inputs =  (str(question), str(evi_id), str(evi_text))
                except Exception as e:
                    logger.debug(f"type(question): {type(question)}")
                    logger.debug(f"evi_text: {evi_text}")
                    raise e
                
            question, image_doc_id, text = inputs
            prompt = f"Question: {question}\nEvidence: <image> {text}</s>"

            input_ids, attention_mask = processor.tokenizer(prompt,
                                                            padding=True,
                                                            truncation=True,
                                                            return_tensors="pt",
                                                            max_length=args.max_seq_len,
                                                            ).values()
            
            if image_doc_id:
                image_path = image_corpus_dict[image_doc_id]['path']
                try:
                    # Open the zipped folder
                    with zipfile.ZipFile(args.image_zip_file_path, 'r') as zip_file:  

                        file_name = 'final_dataset_images/' + image_path
                        with zip_file.open(file_name) as image_file:
                            # Load the image into memory
                            img = Image.open(image_file)
                            try:
                                pixel_values = processor.image_processor(img, return_tensors='pt')["pixel_values"]
                            except Exception as e:
                                logger.info(f"img: {img}")
                                logger.info(f"image_doc_id: {image_doc_id}")
                                logger.info(f"question: {question}")
                                logger.info(f"text: {text}")
                                logger.info(f"WARNING: {e}")
                                pixel_values = torch.zeros((1, sample_num, h, w))
                except Exception as e:
                    logger.info(f"image_doc_id: {image_doc_id}")
                    logger.info(f"question: {question}")
                    logger.info(f"text: {text}")
                    logger.info(f"WARNING: {e}")
                    pixel_values = torch.zeros((1, sample_num, h, w))
            else:
                pixel_values = torch.zeros((1, sample_num, h, w))
               
            model_inputs.append({'input_ids': input_ids,
                                'attention_mask': attention_mask,
                                'pixel_values': pixel_values,
                                }
                            )

        if model_inputs == []:
            logger.info(f"WARNING: No evidence is available.")
            continue
        
        if args.left_pad:
            topk_preds, score_criteria = get_topk_reranked_evidence_left_pad(
                                                                                model = model,
                                                                                model_inputs = model_inputs, 
                                                                                batch_size = args.retrank_batch_size,
                                                                                pad_token_id = pad_token_id,
                                                                                device=device,
                                                                                )
        else:
            topk_preds, score_criteria = get_topk_reranked_evidence_right_pad(
                                                                                model = model,
                                                                                model_inputs = model_inputs, 
                                                                                batch_size = args.retrank_batch_size,
                                                                                pad_token_id = pad_token_id,
                                                                                device=device,
                                                                                )

        topk_evidence = []
        for top in topk_preds:
            evi_id = evidence_list[top]['id']
            evi_text = evidence_list[top]['text']
            evi_type = evidence_list[top]['data_type']
            evi_type_title = evidence_list[top]['title']

            topk_evidence.append({'id': str(evi_id),
                                  'evi_type': evi_type, 
                                  'title': evi_type_title,
                                  'text': evi_text, 
                                  'score': float(score_criteria[top])})

        item['context'] = topk_evidence
        logger.info(f"len(item['context']): {len(item['context'])}")
        if 'multihop_context' in item:
            item.pop('multihop_context') 
        first_hop_reranking_results.append(item)
    
    logger.info(f"Saving reranker predictions to: {output_path}")
    save_jsonl(first_hop_reranking_results, output_path)
    return first_hop_reranking_results


def get_scores(first_hop_reranking_results):
    ##################################################################################################################################
    ##### Evaluating Results  ########################################################################################################
    ##################################################################################################################################
    max_evidence = [1, 2, 3, 4, 5, 8, 12, 16, 20]
    for me in tqdm(max_evidence, desc='Evaluating topk retrieval results.'):
        logger.info("#####################################################")
        logger.info("#####################################################")
        logger.info(f"me: {me}")
        custom_eval(first_hop_reranking_results, max_evidence=me)
    

if __name__ == "__main__":
    make_directory(args.reranking_dir)
    prepare_logger(logger, 
                   debug=args.debug, 
                   save_to_file=os.path.join(args.reranking_dir, 
                                             "reranker_eval.log"))
    logger.info(args)
    output_path = os.path.join(args.reranking_dir, 'rerankOutput_' + get_file_name(args.reranking_test_file) + '.jsonl')
    logger.info(f"output_path: {output_path}")

    if os.path.exists(output_path):
        logger.info(f"Loading first_hop_reranking_results from: {output_path}...")
        first_hop_reranking_results = read_jsonl(output_path)
    else:
        logger.info(f"Start making predictions...")
        first_hop_reranking_results = eval(args, 
                                        output_path=output_path,
                                        )
    
    if 'positive_ctxs' in first_hop_reranking_results[0]:
        logger.info(f"Start scoring...")
        get_scores(first_hop_reranking_results)
        
    logger.info("ALL DONE!")
