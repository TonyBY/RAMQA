# RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering

Yang Bai, Christan Grant, Daisy Zhe Wang<br>
University of Florida <br>

RAMQA introduces a unified framework for Retrieval-Augmented Multi-Modal Question Answering by integrating learning-to-rank and generative ranking techniques to address challenges in multi-modal retrieval tasks, achieving state-of-the-art results on benchmark datasets [WebQA](https://eval.ai/web/challenges/challenge-page/1255/leaderboard/3168) and [MultiModalQA](https://allenai.github.io/multimodalqa/). 

More details about our approach are described in our NAACL paper [RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal
Question Answering](https://www.arxiv.org/abs/2501.13297)

<p align="center"><img width="100%" src="imgs/ramqa_overview.png" /></p>

# Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Download Data and Models](#2-download-data-and-models)
  - [2.1 Data Directory Structure](#21-data-directory-structure)
- [3. Training and Evaluation of RankLLaVA](#3-training-and-evaluation-of-rankllava)
  - [3.1 Data Preparation for RankLLaVA](#31-data-preparation-for-rankllava)
    - [3.1.1 Prepare WebQA Data](#311-prepare-webqa-data)
    - [3.1.2 Prepare MultiModalQA Data](#312-prepare-multimodalqa-data)
  - [3.2 Train RankLLaVA](#32-train-rankllava)
    - [3.2.1 Training with WebQA Data](#321-training-with-webqa-data)
    - [3.2.2 Training with MultiModalQA Data](#322-training-with-multimodalqa-data)
  - [3.3 Evaluate RankLLaVA](#33-evaluate-rankllava)
    - [3.3.1 Evaluation on WebQA Data](#331-evaluation-on-webqa-data)
    - [3.3.2 Evaluation on MultiModalQA Data](#332-evaluation-on-multimodalqa-data)
- [4. Training and Evaluation of RAMLLaMA](#4-training-and-evaluation-of-ramllama)
  - [4.1 Data Preparation for RAMLLaMA](#41-data-preparation-for-ramllama)
    - [4.1.1 Prepare WebQA Data](#411-prepare-webqa-data)
      - [4.1.1.1 Generate Image Descriptions](#4111-generate-image-descriptions)
      - [4.1.1.2 Create Dataset for Training and Testing](#4112-create-dataset-for-training-and-testing)
    - [4.1.2 Prepare MultiModalQA Data](#412-prepare-multimodalqa-data)
      - [4.1.2.1 Generate Image Descriptions](#4121-generate-image-descriptions)
      - [4.1.2.2 Create Dataset for Training and Testing](#4122-create-dataset-for-training-and-testing)
  - [4.2 Train RAMLLaMA](#42-train-ramllama)
    - [4.2.1 Train with WebQA Data](#421-train-with-webqa-data)
    - [4.2.2 Train with MultiModalQA Data](#422-train-with-multimodalqa-data)
  - [4.3 Evaluate RAMLLaMA](#43-evaluate-ramllama)
    - [4.3.1 Evaluation on WebQA Data](#431-evaluation-on-webqa-data)
      - [4.3.1.1 Generate Predictions](#4311-generate-predictions)
      - [4.3.1.2 Score Predictions and Format Submission](#4312-score-predictions-and-format-submission)
    - [4.3.2 Evaluation on MultiModalQA Data](#432-evaluation-on-multimodalqa-data)
      - [4.3.2.1 Generate Predictions](#4321-generate-predictions)
      - [4.3.2.2 Score Predictions](#4322-score-predictions)


## 1. Environment Setup

Requirement: python >= 3.9

```bash
conda create --name ramqa python=3.9
conda activate ramqa
git clone https://github.com/TonyBY/RAMQA.git
cd RAMQA 
pip install -r requirements.txt
```


## 2. Download Data and Models
```bash
bash ./download_data.sh
```


### 2.1 Data Directory Structure
```
RAMQA/data/
└───WebQA
│   └───main_data
│   │    │WebQA_train.json
│   │    │WebQA_val.json
│   │    │WebQA_test.json
│   │
│   └───RankLLaVA_data
│   │    │ranking_train_data_webqa.jsonl
│   │    │ranking_val_data_webqa.jsonl
│   │    │ranking_test_data_webqa.jsonl
│   │
│   └───RAMLLaMA_data
│   │    │webqa_train_top15_permutedTimes-5.jsonl
│   │    │webqa_val_top15_permutedTimes-5_val.jsonl
│   │    │webqa_test_top15_permutedTimes-5_test.jsonl
│   │
│   └───imgs
│   │    │imgs.tsv
│   │    │imgs.lineidx
│   │    │imgs.7z.001
│   │    │...
│   │    │imgs.7z.052
│   │
│   └───webqa_with_image_description
│        │WebQA_with_ImageDescription_train.jsonl
│        │WebQA_with_ImageDescription_val.jsonl
│        │WebQA_with_ImageDescription_test.jsonl
│        │webqa_wiki_dict_with_imgDes.json
│        
└───multimodalqa
│   └───dataset
│   │    │final_dataset_images.zip
│   │    │MMQA_train.jsonl
│   │    │MMQA_dev.jsonl
│   │    │MMQA_test.jsonl
│   │    │MMQA_images.jsonl
│   │    │MMQA_texts.jsonl
│   │    │MMQA_tables.jsonl
│   │ 
│   └───RankLLaVA_data
│   │    │ranking_dev_data_mmqa.json
│   │    │ranking_train_data_mmqa.json
│   │    │image_corpus_mmqa.json
│   │    │text_corpus_mmqa.json
│   │
│   └───RAMLLaMA_data
│   │    │mmqa_train_top20_permutedTimes-5.jsonl
│   │    |mmqa_dev_top20_permutedTimes-5_val.jsonl
│   │
│   └───mmqa_with_image_description
│        │MMQA_with_ImageDescription_train.jsonl
│        │MMQA_with_ImageDescription_val.jsonl
│        │mmqa_wiki_dict_with_imgDes.json
│ 
└───results
    └───webqa
    │    └───RankLLaVA
    │    └───RAMLLaMA
    │
    └───multimodalqa
         └───RankLLaVA        
         └───RAMLLaMA
            
```

## 3. Training and Evaluation of RankLLaVA
### 3.1 Data Preparation for RankLLaVA
#### 3.1.1 Prepare WebQA Data
``` bash
python RAMQA/src/RankLLaVA/data/construct_dataset_for_reranker_webqa.py \
    --webqa_data_path RAMQA/data/WebQA/main_data/WebQA_train.json \
    --is_test False \
    --output_path RAMQA/data/WebQA/RankLLaVA_data/ranking_train_data_webqa.jsonl \
    --img_tsv_path RAMQA/data/WebQA/imgs/imgs.tsv \
    --imgs_lineidx_path RAMQA/data/WebQA/imgs/imgs.lineidx
```

* Input: Original WebQA training/validation data and image index files.
* Output: Traiing/validate data for RankLLaVA training.

#### 3.1.2 Prepare MultiModalQA Data
```bash
python RAMQA/src/RankLLaVA/data/construct_dataset_for_reranker_mmqa.py \
    --mmqa_data_path RAMQA/data/multimodalqa/dataset/MMQA_train.jsonl \
    --output_path RAMQA/data/multimodalqa/ranking_data/ranking_train_data_mmqa.jsonl
```

* Input: Original MMQA training/validation data and image index files.
* Output: Traiing/validate data for RankLLaVA training.

### 3.2 Train RankLLaVA.
#### 3.2.1 Training with WebQA Data
```bash
python RAMQA/src/RankLLaVA/train_rankLlava_webqa.py \
        --reranking_dir ${reranking_dir} \
        --reranking_train_file ${reranking_train_file} \
        --reranking_dev_file ${reranking_dev_file} \
        --imgs_lineidx_path ${imgs_lineidx_path} \
        --img_tsv_path ${img_tsv_path} \
        --model_type llava-hf/llava-1.5-7b-hf \
        --do_train True \
        --left_pad False \
        --seed 7 \
        --retrank_batch_size 8 \
        --num_workers 1 \
        --accumulate_gradients 4 \
        --num_train_epochs 1 \
        --debug False \
        --weighted_sampling False \
        --learning_rate 2e-4 \
        --num_labels 2 \
        --num_gpus 1 \
        --max_seq_len 2048 \
        --gradient_checkpointing True \
        --bits 4 \
        --double_quant False \
        --quant_type nf4 \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_bias none
```

* Input: 
    1. reranking_dir='RAMQA/data/results/webqa/RankLLaVA/<run_name>'
    2. reranking_train_file='RAMQA/data/WebQA/main_data/WebQA_train.json'; 
    3. reranking_dev_file='RAMQA/data/WebQA/main_data/WebQA_val.json';
    4. imgs_lineidx_path='RAMQA/data/WebQA/imgs/imgs.lineidx'
    5. img_tsv_path='RAMQA/data/WebQA/imgs/imgs.tsv'
* Output: A trained RankLLaVA model in the ${reranking_dir}.

#### 3.2.2 Training with MultiModalQA Data
```bash
python RAMQA/src/RankLLaVA/train_rankLlava_mmqa.py \
        --reranking_dir ${reranking_dir} \
        --reranking_train_file ${reranking_train_file} \
        --reranking_dev_file ${reranking_dev_file} \
        --image_zip_file_path ${image_zip_file_path} \
        --image_corpus_path ${image_corpus_path} \
        --text_corpus_path ${text_corpus_path} \
        --model_type llava-hf/llava-1.5-7b-hf \
        --do_train True \
        --left_pad False \
        --seed 7 \
        --retrank_batch_size 8 \
        --num_workers 1 \
        --accumulate_gradients 4 \
        --num_train_epochs 1 \
        --debug False \
        --weighted_sampling False \
        --learning_rate 2e-4 \
        --num_labels 2 \
        --num_gpus 1 \
        --max_seq_len 2048 \
        --gradient_checkpointing True \
        --bits 4 \
        --double_quant False \
        --quant_type nf4 \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_bias none 
```

* Input: 
    1. reranking_dir='RAMQA/data/results/multimodalqa/RankLLaVA/<run_name>'
    2. reranking_train_file='RAMQA/data/WebQA/main_data/WebQA_train.json'; 
    3. reranking_dev_file='RAMQA/data/WebQA/main_data/WebQA_val.json';
    4. image_zip_file_path='RAMQA/data/multimodalqa/dataset/final_dataset_images.zip'
    5. image_corpus_path='RAMQA/data/multimodalqa/RankLLaVA_data/image_corpus_mmqa.json'
    6. text_corpus_path='RAMQA/data/multimodalqa/RankLLaVA_data/text_corpus_mmqa.json'

* Output: A trained RankLLaVA model in the ${reranking_dir}.

### 3.3 Evaluating RankLLaVA
#### 3.3.1 Evaluation on WebQA Data
```bash
python RAMQA/src/RankLLaVA/eval/rankLLaVaEval_webqa.py \
        --reranking_dir ${reranking_dir} \
        --checkpoint_path ${checkpoint_path} \
        --reranking_test_file ${reranking_test_file} \
        --imgs_lineidx_path ${imgs_lineidx_path} \
        --img_tsv_path ${img_tsv_path} \
        --init_checkpoint True \
        --model_type llava-hf/llava-1.5-7b-hf \
        --seed 7 \
        --left_pad False \        
        --retrank_batch_size 16 \
        --debug False \
        --num_labels 2 \
        --max_seq_len 2048 \
        --gradient_checkpointing True \
        --bits 4\
        --double_quant False \
        --quant_type nf4 \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_bias none 
```

* Input: 
    1. reranking_dir='RAMQA/data/results/webqa/RankLLaVA/<run_name>';
    2. checkpoint_path='<reranking_dir>/<best_checkpoint>';
    3. reranking_test_file='RAMQA/data/WebQA/main_data/WebQA_<val/test>.json';
    4. imgs_lineidx_path='RAMQA/data/WebQA/imgs/imgs.lineidx'
    5. img_tsv_path='RAMQA/data/WebQA/imgs/imgs.tsv'
* Output: 
    1. A file with ranking predictions saved at '<reranking_dir>/eval_results/ranking_output_webqa_<val/test>.jsonl'.
    2. A log file that prints out the detailed ranking evaluation scores.


#### 3.3.2 Evaluation on MultiModalQA Data
```bash
python RAMQA/src/RankLLaVA/eval/rankLLaVaEval_mmqa.py \
        --reranking_dir ${reranking_dir} \
        --checkpoint_path ${checkpoint_path} \
        --reranking_test_file ${reranking_test_file} \
        --image_zip_file_path ${image_zip_file_path} \
        --image_corpus_path ${image_corpus_path} \
        --text_corpus_path ${text_corpus_path} \
        --model_type llava-hf/llava-1.5-7b-hf \
        --seed 7 \
        --left_pad False \
        --retrank_batch_size 16 \
        --debug False \
        --num_labels 2 \
        --max_seq_len 2048 \
        --gradient_checkpointing ${gradient_checkpointing} \
        --bits ${bits} \
        --double_quant ${double_quant} \
        --quant_type ${quant_type} \
        --lora_enable ${lora_enable} \
        --lora_r ${lora_r} \
        --lora_alpha ${lora_alpha} \
        --lora_dropout ${lora_dropout} \
        --lora_bias ${lora_bias} \
        --init_checkpoint True
```

* Input: 
    1. reranking_dir='RAMQA/data/results/multimodalqa/RankLLaVA/<run_name>';
    2. checkpoint_path='<reranking_dir>/<best_checkpoint>';
    3. reranking_test_file='RAMQA/data/multimodalqa/dataset/MMQA_dev.jsonl';
    4. image_zip_file_path='RAMQA/data/multimodalqa/dataset/final_dataset_images.zip'
    5. image_corpus_path='RAMQA/data/multimodalqa/RankLLaVA_data/image_corpus_mmqa.json'
    6. text_corpus_path='RAMQA/data/multimodalqa/RankLLaVA_data/text_corpus_mmqa.json'
* Output: 
    1. A file with ranking predictions saved at '<reranking_dir>/eval_results/ranking_output_mmqa_dev.jsonl'.
    2. A log file that prints out the detailed ranking evaluation scores.


## 4. Training and Evaluation of RAMLLaMA
### 4.1 Data Preparation for RAMLLaMA
#### 4.1.1 Prepare WebQA Data
##### 4.1.1.1 Generate Image Descriptions

```bash
python RAMQA/src/RAMLLaMA/data/img_to_text_webqa.py \
        --data_path ${data_path} \
        --output_dir ${output_dir} \
        --imgs_lineidx_path ${imgs_lineidx_path} \
        --img_tsv_path ${img_tsv_path} \
        --debug False \
        --is_test False \
        --llava_model_name llava-hf/llava-1.5-7b-hf \
        --max_new_tokens 100
```

* Input: 
    1. data_path='RAMQA/data/results/multimodalqa/RankLLaVA/<run_name>';
    2. output_dir='RAMQA/data/WebQA/webqa_with_image_description';
    3. imgs_lineidx_path='RAMQA/data/WebQA/imgs/imgs.lineidx'
    4. img_tsv_path='RAMQA/data/WebQA/imgs/imgs.tsv'
* Output: 
    1. A new WebQA dataset with all image-based documents converted into text, saved at: RAMQA/data/WebQA/webqa_with_image_description/WebQA_with_ImageDescription_<train/val/test>.jsonl

##### 4.1.1.2 Create Dataset for Training and Testing

```bash
python RAMQA/src/RAMLLaMA/data/construct_dataset_to_finetune_RAMLLaMA_webqa.py \
        --rank_result_data_path ${rank_result_data_path} \
        --is_test ${is_test} \
        --train_data_path ${train_data_path} \
        --dev_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --output_path ${output_path} \
        --prompt_style llama3 \
        --prompt_task JOINT \
        --topk 15 \
        --permute_times 5
```

* Input: 
    1. rank_result_data_path='RAMQA/data/results/webqa/RankLLaVA/<run_name>/eval_results/ranking_output_webqa_<train/val/test>.jsonl';
    2. is_test=False/True (depends on <webqa_rank_result_data_path>);
    3. train_data_path='RAMQA/data/WebQA/webqa_with_image_description/WebQA_with_ImageDescription_train.jsonl'
    4. dev_data_path='RAMQA/data/WebQA/webqa_with_image_description/WebQA_with_ImageDescription_val.jsonl'
    5. test_data_path='RAMQA/data/WebQA/webqa_with_image_description/WebQA_with_ImageDescription_test.jsonl'
    6. output_path='RAMQA/data/WebQA/RAMLLaMA_data/webqa_<train/val/test>_top15_permutedTimes-5.jsonl'
    5. global_evi_dict_path='RAMQA/data/WebQA/webqa_with_image_description/webqa_wiki_dict_with_imgDes.json'
* Output: 
    1. <train/val/test> data for training/evalauting RAMLLaMA, saved at ${output_path}.
    2. A map between doc_id to textual content (including image descriptions), saved at: ${global_evi_dict_path}.


#### 4.1.2 Prepare MultiModalQA Data
##### 4.1.2.1 Generate Image Descriptions
```bash
python RAMQA/src/RAMLLaMA/data/img_to_text_mmqa.py \
        --data_path ${data_path} \
        --output_dir ${output_dir} \
        --text_corpus_path ${text_corpus_path} \
        --image_corpus_path ${image_corpus_path} \
        --image_zip_file_path ${image_zip_file_path} \
        --use_mixed_txt_img True \
        --debug False \
        --is_test False \
        --llava_model_name llava-hf/llava-1.5-7b-hf \
        --max_new_tokens 100 
```

* Input: 
    1. data_path='RAMQA/data/results/multimodalqa/RankLLaVA/<run_name>';
    2. output_dir='RAMQA/data/WebQA/mmqa_with_image_description';
    3. text_corpus_path='RAMQA/data/multimodalqa/RankLLaVA_data/text_corpus_mmqa.json'
    4. image_corpus_path='RAMQA/data/multimodalqa/RankLLaVA_data/image_corpus_mmqa.json'
    5. image_zip_file_path='RAMQA/data/multimodalqa/dataset/final_dataset_images.zip'
* Output: 
    1. A new MultiModalQA dataset with all image-based documents converted into text, saved at: RAMQA/data/maltumodalqa/mmqa_with_image_description/MMQA_with_ImageDescription_<train/dev>.jsonl

##### 4.1.2.2 Create Dataset for Training and Testing

```bash
python RAMQA/src/RAMLLaMA/data/construct_dataset_to_finetune_RAMLLaMA_mmqa.py \
        --rank_result_data_path ${rank_result_data_path} \
        --is_test False \
        --train_data_path ${train_data_path} \
        --dev_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --output_path ${output_path} \
        --global_evi_dict_path ${global_evi_dict_path}
        --prompt_style llama3 \
        --prompt_task JOINT \
        --topk 15 \
        --permute_times 5
```

* Input: 
    1. rank_result_data_path='RAMQA/data/results/multimodalqa/RankLLaVA/<run_name>/eval_results/ranking_output_mmqa_<train/dev>.jsonl';
    2. train_data_path='RAMQA/data/multimodalqa/mmqa_with_image_description/MMQA_with_ImageDescription_train.jsonl'
    3. dev_data_path='RAMQA/data/multimodalqa/mmqa_with_image_description/MMQA_with_ImageDescription_dev.jsonl'
    4. output_path='RAMQA/data/multimodalqa/RAMLLaMA_data/mmqa_<train/dev>_top15_permutedTimes-5.jsonl'
    5. global_evi_dict_path='RAMQA/data/multimodalqa/mmqa_with_image_description/mmqa_wiki_dict_with_imgDes.json'
* Output: 
    1. <train/dev> data for training/evalauting RAMLLaMA, saved at ${output_path}.
    2. A map between doc_id to textual content (including image descriptions), saved at: ${global_evi_dict_path}.

### 4.2 Train RAMLLaMA
#### 4.2.1 Train with WebQA Data
```bash
python RAMQA/src/RAMLLaMA/train_RAMLLaMA.py \
        --train_file ${train_file} \
        --development_file ${development_file} \
        --output_dir ${output_dir} \
        --llama_model_name meta-llama/Meta-Llama-3-70B \
        --debug False \
        --seed 3407 \
        --train_batch_size 2 \
        --learning_rate learning_rate=1e-4  \
        --warmup_steps 64 \
        --patience 40 \
        --num_workers ${num_workers} \
        --accumulate_gradients 16 \
        --weight_decay 0.01 \
        --num_train_epochs 10 \
        --max_seq_len 8192 \
        --gradient_checkpointing True \
        --use_reentrant False \
        --bits 4 \
        --double_quant False \
        --quant_type nf4 \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0
```
* Input: 
    1. train_file='RAMQA/data/WebQA/RAMLLaMA_data/webqa_train_top15_permutedTimes-5.jsonl';
    2. development_file='RAMQA/data/WebQA/RAMLLaMA_data/webqa_val_top15_permutedTimes-5.jsonl'
    3. output_dir='RAMQA/data/results/webqa/RAMLLaMA/<run_name>'

* Output: 
    1. A trained RAMLLaMA model saved at ${output_dir}.

#### 4.2.2 Train with MultiModalQA Data
```bash
python RAMQA/src/RAMLLaMA/train_RAMLLaMA.py \
        --train_file ${train_file} \
        --development_file ${development_file} \
        --output_dir ${output_dir} \
        --llama_model_name meta-llama/Meta-Llama-3-70B \
        --debug False \
        --seed 3407 \
        --train_batch_size 2 \
        --learning_rate learning_rate=1e-4  \
        --warmup_steps 64 \
        --patience 40 \
        --num_workers 1 \
        --accumulate_gradients 16 \
        --weight_decay 0.01 \
        --num_train_epochs 10 \
        --max_seq_len 8192 \
        --gradient_checkpointing True \
        --use_reentrant False \
        --bits 4 \
        --double_quant False \
        --quant_type nf4 \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0
```
* Input: 
    1. train_file='RAMQA/data/multimodalqa/RAMLLaMA_data/mmqa_train_top20_permutedTimes-5.jsonl';
    2. development_file='RAMQA/data/multimodalqa/RAMLLaMA_data/mmqa_dev_top20_permutedTimes-5.jsonl'
    3. output_dir='RAMQA/data/results/multimodalqa/RAMLLaMA/<run_name>'

* Output: 
    1. A trained RAMLLaMA model saved at ${output_dir}.

### 4.3 Evaluate RAMLLaMA
#### 4.3.1 Evaluation on WebQA Data
##### 4.3.1.1 Generate Predictions
```bash
python RAMQA/src/RAMLLaMA/eval/eval_RAMLLaMA.py \
        --test_file ${test_file} \
        --model_path ${model_path} \
        --output_dir ${output_dir} \
        --llama_model_name meta-llama/Meta-Llama-3-70B \
        --debug ${debug} \
        --init_model True \
        --seed 3407 \
        --max_seq_len 8096 \
        --max_new_tokens 100 \
        --predict_batch_size 2 \
        --bits 4 \
        --double_quant False \
        --quant_type nf4
```
* Input: 
    1. test_file='RAMQA/data/WebQA/RAMLLaMA_data/webqa_<val/test>_top15_permutedTimes-5.jsonl';
    2. model_path='RAMQA/data/results/webqa/RAMLLaMA/<run_name>/<best_checkpoint_name>'
    3. output_dir='RAMQA/data/results/webqa/RAMLLaMA/<run_name>'

* Output: 
    1. A trained RAMLLaMA model saved at ${output_dir}.

##### 4.3.1.2 Score Predictions and Format Submission
```bash
python RAMQA/src/RAMLLaMA/eval/scorer_webqa.py \
        --web_qa_global_evi_dict_path ${web_qa_global_evi_dict_path} \
        --webqa_val_data_path ${webqa_val_data_path} \
        --webqa_test_data_path ${webqa_test_data_path} \
        --webqa_val_rank_data_path ${webqa_val_rank_data_path} \
        --webqa_test_rank_data_path ${webqa_test_rank_data_path} \
        --webqa_val_RAMLLaMA_data_path ${webqa_val_RAMLLaMA_data_path} \
        --webqa_test_RAMLLaMA_data_path ${webqa_test_RAMLLaMA_data_path}
```
* Input: 
    1. web_qa_global_evi_dict_path='RAMQA/data/WebQA/webqa_with_image_description/webqa_wiki_dict_with_imgDes.json';
    2. webqa_val_data_path='RAMQA/data/WebQA/main_data/WebQA_val.json'
    3. webqa_test_data_path='RAMQA/data/WebQA/main_data/WebQA_test.json'
    4. webqa_val_rank_data_path='RAMQA/data/results/webqa/RankLLaVA/<run_name>/eval_results/ranking_output_webqa_val.jsonl'
    5. webqa_test_rank_data_path='RAMQA/data/results/webqa/RankLLaVA/<run_name>/eval_results/ranking_output_webqa_test.jsonl'
    6. webqa_val_RAMLLaMA_data_path='RAMQA/data/results/webqa/RAMLLaMA/<run_name>/RAMLLaMA_webqa_val.jsonl'
    7. webqa_test_RAMLLaMA_data_path='RAMQA/data/results/webqa/RAMLLaMA/<run_name>/RAMLLaMA_webqa_test.jsonl'

* Output: 
    1. A log file contains scores on validaiton data saved at: 'RAMQA/data/results/webqa/RAMLLaMA/<run_name>/scorer_webqa.log'.
    2. A submission ready test file for the WebQA leaderboard saved at: 'RAMQA/data/results/webqa/RAMLLaMA/<run_name>/official_result.json'.

#### 4.3.2 Evaluation on MultiModalQA Data
##### 4.3.2.1 Generate Predictions
```bash
python RAMQA/src/RAMLLaMA/eval/eval_RAMLLaMA.py \
        --test_file ${test_file} \
        --model_path ${model_path} \
        --output_dir ${output_dir} \
        --llama_model_name meta-llama/Meta-Llama-3-70B \
        --debug ${debug} \
        --init_model True \
        --seed 3407 \
        --max_seq_len 8096 \
        --max_new_tokens 100 \
        --predict_batch_size 2 \
        --bits 4 \
        --double_quant False \
        --quant_type nf4
```
* Input: 
    1. test_file='RAMQA/data/multimodalqa/RAMLLaMA_data/mmqa_<dev>_top15_permutedTimes-5.jsonl';
    2. model_path='RAMQA/data/results/mmqa/RAMLLaMA/<run_name>/<best_checkpoint_name>'
    3. output_dir='RAMQA/data/results/mmqa/RAMLLaMA/<run_name>'

* Output: 
    1. A trained RAMLLaMA model saved at ${output_dir}.

##### 4.3.2.2 Score Predictions
```bash
python RAMQA/src/RAMLLaMA/eval/scorer_mmqa.py\
        --mmqa_global_evi_dict_path ${mmqa_global_evi_dict_path} \
        --mmqa_dev_data_path ${mmqa_dev_data_path} \
        --mmqa_dev_rank_data_path ${mmqa_dev_rank_data_path} \
        --mmqa_dev_RAMLLaMA_data_path ${mmqa_dev_RAMLLaMA_data_path}
```
* Input: 
    1. mmqa_global_evi_dict_path='RAMQA/data/multimodalqa/mmqa_with_image_description/webqa_wiki_dict_with_imgDes.json';
    2. mmqa_dev_data_path='RAMQA/data/multimodalqa/dataset/MMQA_dev.jsonl'
    3. mmqa_dev_rank_data_path='RAMQA/data/results/multimodalqa/RankLLaVA/<run_name>/eval_results/ranking_output_mmqa_dev.jsonl'
    4. mmqa_dev_RAMLLaMA_data_path='RAMQA/data/results/multimodalqa/RAMLLaMA/<run_name>/RAMLLaMA_mmqa_dev.jsonl'

* Output: 
    1. A log file contains scores on validaiton data saved at: 'RAMQA/data/results/multimodalqa/RAMLLaMA/<run_name>/scorer_mmqa.log'.

## Cite
```
@misc{bai2025ramqaunifiedframeworkretrievalaugmented,
      title={RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering}, 
      author={Yang Bai and Christan Earl Grant and Daisy Zhe Wang},
      year={2025},
      eprint={2501.13297},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.13297}, 
}
```

## License
CC-BY-NC 4.0
