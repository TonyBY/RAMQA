from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
sys.path.append(pwd)
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc
os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.

import logging
from functools import partial
import numpy as np
import pathlib
import random

from peft import TaskType

import transformers
from transformers import (
                            AutoProcessor,
                            TrainingArguments,
                            Trainer,
                            LlavaForConditionalGeneration,
                        )
from transformers.trainer_utils import EvalPrediction

import torch

from src.utils.config import parser
from src.utils.args import prepare_logger
from src.utils.data_utils import make_directory, read_jsonl
from src.utils.model_utils import find_all_linear_names, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from src.RankLLaVA.models.RankLLaVA_model import RankLLaVA
from src.RankLLaVA.data.reranking_dataset_class_webqa import RankingDatasetWebQA
from src.RankLLaVA.data.collate_fn_webqa import collate_fn_leftPad, collate_fn_rightPad

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

torch.set_float32_matmul_precision('medium')


def compute_metrics(predictions: EvalPrediction=None):
    logger.debug(f"predictions: {predictions}")
    logger.debug(f"predictions.__dir__(): {predictions.__dir__()}")
    logger.debug(f"predictions.predictions: {predictions.predictions}")
    logger.debug(f"predictions.predictions[1]: {predictions.predictions[1]}")
    logger.debug(f"predictions.label_ids: {predictions.label_ids}")
    logger.debug(f"predictions.inputs: {predictions.inputs}")
    logger.debug(f"torch.tensor(predictions.predictions[1]) == torch.tensor(predictions.label_ids): {torch.tensor(predictions.predictions[1]) == torch.tensor(predictions.label_ids)}")
    
    pred_logits = torch.tensor(predictions.predictions[0])
    logger.debug(f"pred_logits: {pred_logits}")

    pre_labels = torch.argmax(pred_logits, dim=1)
    logger.debug(f"pre_labels: {pre_labels}")

    accuracy = torch.sum(pre_labels == torch.tensor(predictions.label_ids)).item() / (len(predictions.label_ids) * 1.0)
    logger.debug(f"accuracy: {accuracy}")
    return {'accuracy': accuracy}


def train(args):
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
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))
    
    train_dataset = read_jsonl(args.reranking_train_file)
    
    dev_dataset = read_jsonl(args.reranking_dev_file)

    processor = AutoProcessor.from_pretrained(args.model_type)

    train_dataset = RankingDatasetWebQA(
        data=train_dataset, 
        weighted_sampling=args.weighted_sampling,
        seed=args.seed,
        is_debug=args.debug, 
        is_train=True,
        processor=processor,
        lineidx_path=args.imgs_lineidx_path,
        img_tsv_path=args.img_tsv_path,
        max_length=args.max_seq_len,
    )

    dev_dataset = RankingDatasetWebQA(
        data=dev_dataset,
        weighted_sampling=args.weighted_sampling,
        seed=args.seed,
        is_debug=args.debug, 
        is_train=False,
        processor=processor,
        lineidx_path=args.imgs_lineidx_path,
        img_tsv_path=args.img_tsv_path,
        max_length=args.max_seq_len,
    )

    if args.left_pad:
        logger.info(f"Left Padding")
        collate_fn = collate_fn_leftPad
    else:
        logger.info(f"Right Padding")
        collate_fn = collate_fn_rightPad

    partial_collate_fn = partial(
        collate_fn, 
        pad_token_id=processor.tokenizer.pad_token_id, 
    )

    total_steps = len(train_dataset) // (int(args.retrank_batch_size) * int(args.accumulate_gradients))
    eval_steps = total_steps // 8
    logger.info(f"len(train_dataset): {len(train_dataset)}")
    logger.info(f"int(args.retrank_batch_size): {int(args.retrank_batch_size)}")
    logger.info(f"int(args.accumulate_gradients): {int(args.accumulate_gradients)}")
    logger.info(f"total_steps: {total_steps}")
    logger.info(f"eval_steps: {eval_steps}")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    bnb_model_from_pretrained_args = {}
    if args.bits in [4, 8]:
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

    if args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        back_bone_model.config.torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        model = RankLLaVA(
                            back_bone_model=back_bone_model,
                            num_labels=args.num_labels,
                            class_weights=train_dataset.class_weights,
                        )
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        model = RankLLaVA(
                            back_bone_model=back_bone_model,
                            num_labels=args.num_labels,
                            class_weights=train_dataset.class_weights,
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

    if args.bits in [4, 8]:
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

    # Set training parameters
    training_arguments = TrainingArguments(
                                            do_train=args.do_train,
                                            do_eval=True,
                                            output_dir=args.reranking_dir,
                                            overwrite_output_dir=False,
                                            dataloader_drop_last=True,
                                            evaluation_strategy="steps",
                                            save_strategy='steps',
                                            logging_strategy="steps",
                                            num_train_epochs=args.num_train_epochs,
                                            eval_steps=1 if args.debug else eval_steps,
                                            save_steps=1 if args.debug else eval_steps, # Save checkpoint every X updates steps
                                            eval_delay=0,
                                            logging_steps=100,
                                            per_device_train_batch_size=int(args.retrank_batch_size),
                                            per_device_eval_batch_size=int(0.5 * args.retrank_batch_size),
                                            optim="paged_adamw_32bit",
                                            learning_rate=args.learning_rate,
                                            lr_scheduler_type="linear", # https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup
                                            warmup_steps=30,
                                            gradient_accumulation_steps=args.accumulate_gradients,
                                            gradient_checkpointing=args.gradient_checkpointing,
                                            gradient_checkpointing_kwargs={'use_reentrant': True},
                                            weight_decay=args.weight_decay,  # Weight decay to apply to all layers except bias/LayerNorm weights
                                            report_to="wandb",
                                            load_best_model_at_end=True,
                                            save_total_limit=1,
                                            bf16=True if torch.cuda.is_bf16_supported() else False,
                                            fp16=False if torch.cuda.is_bf16_supported() else True,
                                            max_grad_norm=0.3, # Maximum gradient normal (gradient clipping)
                                            max_steps=-1, # Number of training steps (overrides num_train_epochs)
                                            group_by_length=False, # Group sequences into batches with same length. Saves memory and speeds up training considerably
                                            run_name='RankingLLaVaTraining_WebQA',
                                            disable_tqdm=False,
                                            ddp_find_unused_parameters=False,
                                            dataloader_num_workers=args.num_workers,
                                            dataloader_pin_memory=True,
                                            data_seed=args.seed,
                                            torch_compile=False,
                                            log_level='debug' if args.debug else 'info', # 'info', 'passive', 'debug', 'error'
                                        )

    logger.info(f"training_arguments: {training_arguments}")
    
    # Set supervised fine-tuning parameters
    logger.debug(f"type(model): {type(model)}")

    # logger.info(f"training_arguments: {training_arguments}")
    logger.info(f"Initializting trainer...")
    trainer = Trainer(
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset,
                    data_collator=partial_collate_fn,
                    args=training_arguments,
                    compute_metrics=compute_metrics,
                )

    # Train model
    logger.info("Start training...")
    if list(pathlib.Path(args.reranking_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if args.local_rank == 0 or args.local_rank == -1:
            model.config.save_pretrained(args.reranking_dir)
            model.save_pretrained(args.reranking_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(args.reranking_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=args.reranking_dir)


if __name__ == "__main__":
    make_directory(args.reranking_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.reranking_dir, "reranker_training.log"))
    logger.info(args)
    train(args)
    logger.info("ALL DONE!")
