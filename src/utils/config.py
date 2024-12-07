from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

from RAMQA.src.utils.args import ArgumentGroup

parser = argparse.ArgumentParser(__doc__)

common_g = ArgumentGroup(parser, "common", "common options.")
common_g.add_arg("num_workers",                      int,           1,
                "Number of workers of DataLoader.")
common_g.add_arg("do_train",                         bool,          True,  
                "Whether to perform training.")
common_g.add_arg("resume_from_checkpoint",           bool,          False,  
                "A hugging face trainer parameter. \
                If a str, local path to a saved checkpoint as saved by a previous instance of Trainer. \
                If a bool and equals True, load the last checkpoint in args.output_dir as saved by a previous instance of Trainer. \
                If present, training will resume from the model/optimizer/scheduler states loaded here."
                )
common_g.add_arg("do_pre_val",                       bool,          True,  
                "Whether to perform val over the initial model before start training.")
common_g.add_arg("do_test",                          bool,          False,
                "Whether to run eval on the dev set.")
common_g.add_arg("train_mode",                       str,           "JOINT",
                "Training mode, choices from ('RETRIEVE_ONLY', 'QA_ONLY', 'JOINT')")
common_g.add_arg("use_cuda",                         bool,          True,  
                "If set, use GPU for training.")
common_g.add_arg("local_rank",                       int,           -1,
                "local_rank for distributed training on gpus")
common_g.add_arg("no_cuda",                          bool,          False,
                "Whether not to use CUDA when available")
common_g.add_arg("debug",                            bool,          False,
                "Controls log level and other logics for debugging.")
common_g.add_arg("num_gpus",                         int,           1,
                "Number of gpus running on.")
common_g.add_arg("huggingface_token",                str,           '',
                "Huggingface token to access gated models.")


model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bart_text_model_name_or_path",      str,           "",
                "Path to a pretrained BART model.")
model_g.add_arg("bart_fusion_model_name_or_path",    str,           "",
                "Path to a pretrained BART model.")
model_g.add_arg("ofa_model_name_or_path",            str,           "",
                "Path to a pretrained FOA model.")
model_g.add_arg("llama_model_name",                  str,           "",
                "Path to a pretrained llama model.")
model_g.add_arg("llava_model_name",                  str,           "",
                "Path to a pretrained llava model.")
model_g.add_arg("init_checkpoint",                   bool,          False,
                "Whether to init model with checkpoint.")
model_g.add_arg("checkpoint_path",                   str,           "",
                "Path to a pretrained model.")
model_g.add_arg("init_model",                        bool,          False,
                "Whether to init model from local pretrained weights.")
model_g.add_arg("local_ofa_model_path",              str,           "",
                "Path to a local pretrained ofa_model.")
model_g.add_arg("local_bart_text_model_path",        str,           "",
                "Path to a local pretrained bart_text_model.")
model_g.add_arg("continue_training",                 bool,          True,
                "Whether to inherent previous best results when initiating from pretrained models.")
model_g.add_arg("output_dir",                        str,           "/home/tony/projects/FactChecking/data/checkpoints",  
                "Directory to save checkpoints.")
model_g.add_arg("use_fusion_encoder",                bool,          True,
                "Whether to use fusion encoder.")
model_g.add_arg("query_encoder_name",                str,           "facebook/dpr-question_encoder-multiset-base",
                "Name/dir of query encoder model.")
model_g.add_arg("ctx_encoder_name",                  str,           "facebook/dpr-ctx_encoder-multiset-base",
                "Name/dir of context encoder model.")
model_g.add_arg("single_encoder",                 bool,          True,
                "Whether to only use single encoder to encode both query and contex for retrieval.")



data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file",                         str,           "", 
                "Path to the training data.")
data_g.add_arg("development_file",                   str,           "", 
                "Path to the development data file.")
data_g.add_arg("test_file",                          str,           "", 
                "Path to the test data file.")
data_g.add_arg("max_seq_len",                        int,           64,   
                "Number of tokens of the longest seqence.")
data_g.add_arg("max_query_len",                      int,           32,   
                "Number of tokens of the longest question seqence.")
data_g.add_arg("max_caption_len",                    int,           32,   
                "Number of tokens of the longest image caption seqence.")
data_g.add_arg("img_tsv_path",                       str,           "",   
                "Path to the WebQA image tsv file.")
data_g.add_arg("imgs_lineidx_path",                  str,           "",   
                "Path to the WebQA image lineidx file.")
data_g.add_arg("use_cache",                          bool,          True,   
                "Whether to use cache when initiating the Dataset Object.")
data_g.add_arg("num_hard_negs",                      int,           2,    
                "Number of hard negatives per example.")
data_g.add_arg("num_txt_negs",                       int,           32,   
                "Number of negative text evidence examples to include for when doing data processing.")
data_g.add_arg("num_img_negs",                       int,           32,   
                "Number of negative iamge evidence examples to include for when doing data processing..")
data_g.add_arg("image_zip_file_path",                str,           "",   
                "Path to the zip file where saves all the images in the MMQA dataset.")
data_g.add_arg("image_corpus_path",                  str,           "",   
                "Path to the json file where saves a dictionary the maps image_doc_id to image_path in the zip file of the image corpus.")
data_g.add_arg("text_corpus_path",                   str,           "",   
                "Path to the json file where saves a dictionary the maps text_doc_id to text_content.")
data_g.add_arg("left_pad",                           bool,          False,   
                "Whether to pad sequence to the left when creating batches. Generative models usually pad sequence to their left.")



train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg('prefix',                            str,           "eval",
                "Run name of experiment.")
train_g.add_arg("num_train_epochs",                  int,           3,       
                "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",                     float,         1e-5,    
                "Learning rate used to train with warmup.")
train_g.add_arg("use_adam",                          bool,          True,
                "Whether to use adam optimizer.")
train_g.add_arg("adam_epsilon",                      float,         1e-8,
                "Epsilon for Adam optimizer.")
train_g.add_arg("lr_scheduler",                      str,           "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",                      float,         0.01,    
                "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_ratio",                      float,         0.1,
                "Proportion of training steps to perform linear learning rate warmup.")
train_g.add_arg("warmup_steps",                      int,           50,
                "Number of training steps to perform linear learning rate warmup.")

train_g.add_arg("save_checkpoints_steps",            int,           10000,   
                "The steps interval to save checkpoints.")
train_g.add_arg("weighted_sampling",                 bool,          False,
                "Whether to do over sampling for rare classes.")
train_g.add_arg("train_batch_size",                  int,           1,    
                "Total examples' number in batch for training.")
train_g.add_arg("extra_batch_size",                  int,           1,    
                "Total examples' number in batch for extra training dataset.")
train_g.add_arg("predict_batch_size",                int,           1,    
                "Total examples' number in batch for training.")
train_g.add_arg("accumulate_gradients",              int,           1,
                "Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
train_g.add_arg('gradient_accumulation_steps',       int,           1,
                "Number of updates steps to accumualte before performing a backward/update pass.")
train_g.add_arg('seed',                              int,           3,
                "Random seed for initialization")
train_g.add_arg("max_grad_norm",                     float,         2, 
                "Max gradient norm.")
# train_g.add_arg("train_qa_only",                  bool,          False,
#                 "Train with only the QA objective.")
# train_g.add_arg("train_retri_only",                  bool,          False,
#                 "Train with only the retrieval objective.")
train_g.add_arg("use_qa_best_score",                 bool,          False,
                "Whether to use best QA score to decide when to cache model.")
train_g.add_arg("use_retri_best_score",              bool,          False,
                "Whether to use best retrieval recall score to decide when to cache model.")
train_g.add_arg("use_joint_best_score",              bool,          False,
                "Whether to use best joint score to decide when to cache model.")
train_g.add_arg("sim_type",                          str,           'dot',
                "Whether to use dot product or consine similarity to evalate coherence between two sentence embeddings. Option: 'dot' or 'cosine'. ")
train_g.add_arg("temp",                              float,          1.0,
                "Temperatur, a hyper-parameter that used to scale similarity score between two sentence embeddings, e.g., score = score/temp.")
train_g.add_arg("use_ce_loss",                       bool,           False,
                "Whether to use cross-entropy loss, if not, negative log likelihood loss would be used.")
train_g.add_arg("senteval_data_path",                str,            "",
                "Path to the SentEval dataset for universal sentence embedding evaluation.")
train_g.add_arg("caching_metric",                    str,            "",
                "Metric name that used to caching the best model during training. (stsb, stskr, joint_sts)")
train_g.add_arg("gradient_checkpointing",            bool,           False,
                "Whether to use cross-gradient_checkpointing to train the model.")
train_g.add_arg("bits",                              int,            16,    
                "Activate 4-/8-/16-bit precision base model loading.")
train_g.add_arg("double_quant",                      bool,           False,
                "Activate nested quantization for 4-bit base models (double quantization).")
train_g.add_arg("quant_type",                        str,            "nf4",
                "Quantization type (fp4 or nf4)")
train_g.add_arg("lora_enable",                       bool,           False,
                "Whether to use lora to train the model.")
train_g.add_arg("lora_r",                            int,            8,    
                "Lora rank, only used if lora_enable is True")
train_g.add_arg("lora_alpha",                        int,            32,    
                "Lora alpha, only used if lora_enable is True")
train_g.add_arg("lora_dropout",                      float,          0.1,
                "Dropout probability for LoRA layers, only used if lora_enable is True.")
train_g.add_arg("lora_bias",                         str,            "none",
                "Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if lora_enable is True")
train_g.add_arg("use_reentrant",                     bool,           False,
                "Whether to use use_reentrant when using gradient checkpointing to train the model.")
train_g.add_arg("patience",                          int,            4,
                "Specify the number of evaluations to perform after achieving the best evaluation metric before stopping the training.")



index_g = ArgumentGroup(parser, "index", "index options.")
index_g.add_arg('encode_only',                       bool,          False,
                "Whether to only generate sentence embeddings without doing indexing.")
index_g.add_arg('index_dir',                         str,           "",
                "Directory to save generated index and relevant files.")
index_g.add_arg('corpus_path',                       str,           "",
                "Processed corpus '.jsonl' file path.")
index_g.add_arg('dataset',                           str,           "",
                "Indicating which dataset is in using: webqa or mmqa.")
index_g.add_arg('np_embedding_path',                 str,           '',
                "Path to the npy file of generated evidence embeddings.")
index_g.add_arg('encoding_batch_size',               int,           128,
                "Batch size for encoding sentences.")
index_g.add_arg('indexing_batch_size',               int,           50000,
                "Buffer size for generating index.")
index_g.add_arg('index_type',                        str,           'HNSWFlat',
                "Faisee index type, choices from ('IndexFlatIP', 'IVF16384_HNSW32', 'HNSWFlat')")


# index_g.add_arg('shard_id',                          int,           0,
#                 "Chunk/shard id for parrallel encoding.")
# index_g.add_arg('index_dim',                         int,           768,
#                 "Hidden size. 768 for roberta-base and bert-base, 1024 for roberta-large and bert-large.")


search_g = ArgumentGroup(parser, "search", "search options.")
search_g.add_arg('data_path',                        str,           '',
                "Query data path.")
search_g.add_arg('output_path',                      str,           '',
                "Searching results output path.")
search_g.add_arg("wiki_dict_path",                   str,           '',
                "Path to a json file of a map between evidence id to evidence content.")
search_g.add_arg('index_in_gpu',                     bool,          True,
                 "Whether to search with index in gpus.")
search_g.add_arg('cache_searching_result',           bool,          True,
                 "Cache intermediate reults every n steps while doing searching.")
search_g.add_arg('topk',                             int,           100,
                 "Whether to search with index in gpus.")
search_g.add_arg('query_batch_size',                 int,           64,
                "Query batch size for searching.")


ir_eval_g = ArgumentGroup(parser, "ir_eval", "retrieval evaluation options.")
ir_eval_g.add_arg('retrieval_result_path',           str,           '',
                  "Path to pickl file of the retrieval results that to be evaluated.")
ir_eval_g.add_arg('strict',                          bool,          True,
                  "Wheter to perform strict retrieval evaluaton for the multi-hop/multi-annotation examples.")


reranker_g = ArgumentGroup(parser, "rerank", "sentence rerank options.")
reranker_g.add_arg('first_hop_search_results_path',  str,           '',
                 "Path to the frist hop search results by DPR-singleHop.")
reranker_g.add_arg('is_test',                        bool,          False,
                 "Whether the dataset is for testing.")
reranker_g.add_arg('webqa_to_reranking',             bool,          False,
                 "Whether to transform the dataset format from webqa to the format for reranking evaluation.")
reranker_g.add_arg('retrank_batch_size',             int,           128,
                 "Batch size for reranking.")
reranker_g.add_arg('reranking_dir',                  str,           '',
                 "Reranking output directory.")
reranker_g.add_arg('reranking_train_file',           str,           "",
                 "Training set file path.")
reranker_g.add_arg('reranking_dev_file',             str,           "",
                 "Developement set file path.")
reranker_g.add_arg('reranking_test_file',             str,           "",
                 "Tesing set file path.")
reranker_g.add_arg('model_path',                     str,           '',
                 "Path to the pretrained model.")
reranker_g.add_arg('model_type',                     str,           '',
                 "Model type for doing reranking, e.g., roberta-large.")
reranker_g.add_arg('num_labels',                     int,           3,
                 "Number of labels the reranker is trained with, either 2(e.g., relevant/not relevant) or 3(e.g., supporting, refuting, not enough info).")
reranker_g.add_arg('fist_hop_topk',                  int,           128,
                 "Top-k sentences from the first hop search results as input to the reranker.")
reranker_g.add_arg('num_neg_samples',                int,           100,
                 "Number of negative samples per claim when constructing dataset to train sentence reranker.")
          

msrr_g = ArgumentGroup(parser, "multihop_sentence_reranking_and_result_merging.", "Multi-hop reranking options.")
msrr_g.add_arg('joint_reranking',                     bool,           False,
               "Whether to constructe dataset for a joint ranking task.")
msrr_g.add_arg('singleHopNumbers',                    int,            0,
               "Number of single-hop examples will be included when doing multi-hop retrieval/evaluation/dataset construction for joint reranking.")
msrr_g.add_arg('multiHopNumbers',                     int,            0,
               "Number of multi-hop examples will be included when constructing dataset for joint reranking.")



msrr_g.add_arg('merged_reranked_results_dir',         str,           '',
               "Output directory of merged results.")
msrr_g.add_arg('msrr_result_path',                    str,           '',
               "Input data file, the output of multihop sentence reranker.")
msrr_g.add_arg('msrr_merge_metric',                   str,           '',
               "Metric name the used to calculate scores to filter irrelevant multi-hop retrieval paths. Choices: ['path', 'sum']")
msrr_g.add_arg('mhth',                                float,          0.9,
               "Threshold of the irrelevant multi-hop retrieval paths filter.")
msrr_g.add_arg('alpha',                               float,          1.0,
               "Weight of score when merging two retriever's results.")
msrr_g.add_arg('normalization',                       bool,           True,
               "Whether to normalize ranking scores before merging two retrievers' results.")
msrr_g.add_arg('weight_on_dense',                     bool,           False,
               "Whether to apply the alpha/weight over the dense_retriever/singlehop_reranker or over the sparse_retriever/multihop_reranker.")
msrr_g.add_arg('save_evi_path',                       bool,           True,
               "Whether to save previous evidence when doing multihop sentence reranking.")
msrr_g.add_arg('concat_claim',                        bool,           True,
               "Whether to concate claim before an first-hop evidence as a new claim when doing sencond-hop reranking, otherwise, only the first-hop evidence will be used as the claim.")
msrr_g.add_arg('naive_merge',                         bool,           False,
               "Whether to do naive merging when combining sing-hop and multi-hop sentence reranking results. Otherwise, complex joint reranking will be applied.")
msrr_g.add_arg('naive_merge_discount_factor',         float,          0.95,
               "Discount factor applied over the multi-hop evidence when combining single-hop and multi-hop sentence reranking results naively.")
msrr_g.add_arg('tune_params',                         bool,           True,
               "Whether to tune parames or use given parameters when doing complext joint reranking.")


mdr_eval_g = ArgumentGroup(parser, "multihop_doc_retrieval", "Retrieve and evaluate document-level multihop evidence.")
mdr_eval_g.add_arg('multi_hop_dense_retrieval',      bool,          False,
                  "Whether to do multi-hop dense retrieval.")


mdr_eval_g.add_arg('multihop_doc_retrieval_dir',     str,           '',
                  "Output directory of saving evalution results of doc-level multi-hop retrieval.")
mdr_eval_g.add_arg('sufficiency_checking_results_path', str,        '',
                  "Output file path of the sufficiency checking module, which will be used as input of this module.")
mdr_eval_g.add_arg('similarity_func_name',           str,           'jaccard',
                  "Similarity function used for document-level retrieval. ('jaccard', 'cosine', 'containment')")
mdr_eval_g.add_arg('similarity_threshold',           float,         0.5,
                  "Similarity thrshold for doc-level multihop evidence retrieval.")
mdr_eval_g.add_arg('mdr_topk',                       int,           2,
                  "topk docs to retrieve for each hyperlink. ")
mdr_eval_g.add_arg('maxhop',                         int,           2,
                  "Only do multihop retrieval for those multihop examples whose minimum evidence hop is less than or equal to 'maxhop'. ")
mdr_eval_g.add_arg('srr_th',                         float,         1.0,
                  "Sentnence reranking score threshold, used for multi_hop evidence selecting.")
mdr_eval_g.add_arg('sf_th',                          float,         1.0,
                  "Sufficiency checking score threshold, used for multi_hop evidence selecting.")


mqa_eval_g = ArgumentGroup(parser, "multimodel_QA", "Evaluate Multi-model QA.")
mqa_eval_g.add_arg('image_source',                    str,          '',
                  "Used to decide use which method to access and process images {webqa, mmqa, path}.")
mqa_eval_g.add_arg('max_new_tokens',                  int,          500,
                  "Number of new tokens to generate.")


mqa_aligner_g = ArgumentGroup(parser, "multimodel_QA_answer_aligner", "Align generated answer with the golden answer.")
mqa_aligner_g.add_arg('webqa_ori_train_dataset_path',  str,          '',
                  "File path to the original WebQA training data.")
mqa_aligner_g.add_arg('webqa_ori_dev_dataset_path',    str,          '',
                  "File path to the original WebQA validation data.")
mqa_aligner_g.add_arg('bart_score_batch_size',         int,          4,
                  "Batch size when computing the BART score.")
mqa_aligner_g.add_arg('bart_score_switch_input',       bool,         False,
                  "Whether to swith input sequence when computing the BART score.")
mqa_aligner_g.add_arg('add_question',                  bool,         True,
                  "Whether to include question in the input sequence.")
