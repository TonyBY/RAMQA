from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-6]))
sys.path.append('/'.join(pwd.split('/')[:-5]))
sys.path.append('/'.join(pwd.split('/')[:-4]))
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from datasets import Dataset, DatasetDict
import numpy as np
import logging
from PIL import ImageFile
import pathlib
import random

import torch
import transformers
from transformers import TrainingArguments, EarlyStoppingCallback

from trl import SFTTrainer

from unsloth import FastLanguageModel

from RAMQA.src.utils.config import parser
from RAMQA.src.utils.args import prepare_logger
from RAMQA.src.utils.data_utils import read_jsonl, make_directory

np.set_printoptions(precision=4)
ImageFile.LOAD_TRUNCATED_IMAGES = True

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Dataset
    training_data = read_jsonl(args.train_file)
    val_data = read_jsonl(args.development_file)

    if args.debug:
        training_data = training_data[:(int(args.train_batch_size) * int(args.accumulate_gradients))]
        val_data = val_data[:int(args.train_batch_size)]

    traing_dataset = Dataset.from_list(training_data)
    val_dataset = Dataset.from_list(val_data)

    dataset = DatasetDict({
                        'train': traing_dataset,
                        'validation': val_dataset,
                    })
    
    logger.info(f"dataset: {dataset}")

    total_steps = len(training_data) // (int(args.train_batch_size) * int(args.accumulate_gradients))
    eval_steps = total_steps // args.patience

    logger.info(f"len(train_dataset): {len(training_data)}")
    logger.info(f"int(args.train_batch_size): {int(args.train_batch_size)}")
    logger.info(f"int(args.accumulate_gradients): {int(args.accumulate_gradients)}")
    logger.info(f"total_steps / Epoch: {total_steps}")
    logger.info(f"args.patience: {args.patience}")
    logger.info(f"eval_steps: {eval_steps}")

    
    # 2. Load Llama3 model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.llama_model_name,
        max_seq_length = args.max_seq_len,
        dtype = None,
        load_in_4bit = args.bits == 4,
    )

    # 3 Before training
    def generate_text(text):
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=100)
        logger.info(tokenizer.decode(outputs[0], skip_special_tokens=True))

    logger.info("Before training\n")
    generate_text(val_dataset[0]['text'].split('### Response:')[0] + '### Response:\n        ')

    # 4. Do model patching and add fast LoRA weights and training
    model = FastLanguageModel.get_peft_model(
                                                model,
                                                r = args.lora_r,
                                                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                                "gate_proj", "up_proj", "down_proj",],
                                                lora_alpha = args.lora_alpha,
                                                lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
                                                bias = "none",    # Supports any, but = "none" is optimized
                                                use_gradient_checkpointing = args.gradient_checkpointing,
                                                random_state = args.seed,
                                                max_seq_length = args.max_seq_len,
                                                use_rslora = False,  # Rank stabilized LoRA
                                                loftq_config = None, # LoftQ
                                            )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        num_train_epochs=1 if args.debug else args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=int(1 * args.train_batch_size),
        gradient_accumulation_steps=args.accumulate_gradients,
        # optim='paged_adamw_32bit',
        optim="adamw_8bit",
        eval_steps=1 if args.debug else eval_steps,
        save_steps=1 if args.debug else eval_steps, # Save checkpoint every X updates steps
        save_total_limit=1,
        eval_delay=0,
        # logging_steps=1 if args.debug else 100,
        logging_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        max_grad_norm=0.3,                          # Maximum gradient normal (gradient clipping)
        max_steps=-1,                               # Number of training steps (overrides num_train_epochs)
        # warmup_ratio=0.03,                          # Ratio of steps for a linear warmup (from 0 to learning rate)
        warmup_steps = args.warmup_steps,
        group_by_length=True,                       # Group sequences into batches with same length; Saves memory and speeds up training considerably
        # lr_scheduler_type='cosine',                 # Learning rate schedule
        lr_scheduler_type='linear',                 # Learning rate schedule
        report_to="wandb",
        # report_to="tensorboard",
        load_best_model_at_end=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': args.use_reentrant},
        disable_tqdm=False,
        seed=args.seed,
        # hub_token=args.huggingface_token,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.add_callback(EarlyStoppingCallback(args.patience, 0.0))

    logger.info(f"trainer.args: {trainer.args}")

    # Train model
    logger.info("Start training...")
    if list(pathlib.Path(args.output_dir).glob("checkpoint-*")) and not args.debug:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # 5. After training
    logger.info("\n ######## \nAfter training\n")
    generate_text(val_dataset[0]['text'].split('### Response:')[0] + '### Response:\n        ')

    # Save trained model
    trainer.save_state()
    trainer.model.save_pretrained(os.path.join(args.output_dir, str(args.llama_model_name).split('/')[-1] + '_ramqa'))

if __name__=='__main__':
    make_directory(args.output_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.output_dir, "llama3_ramqa_training.log"))
    logger.info(args)
    train(args)
    logger.info("ALL DONE!")
    