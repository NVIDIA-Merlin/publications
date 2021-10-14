# Experiments reproducibility
The experiments for the Transformers4Rec paper were performed in a former version of the Transformers4Rec library tagged as [recsys2021](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021), which can be used for reproducibility. This document provides instruction for reproducing the experiments with that original (pre-release) codebase we used for the Transformers4Rec paper. 

**IMPORTANT**: For researchers and practioners aiming to perform experiments similar to the ones presented in our paper (e.g. incremental training and evaluation of session-based recommendation with Transformers), we strongly encourage the usage of the [latest version](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/t4rec_paper_experiments) of the experiment scripts which were updated to use the released PyTorch API, because it was completely refactored, is more modularized and documented than the original scripts, and is supported by the NVIDIA Merlin team.

## Pre-processing
We provide scripts for preprocessing the datasets [here](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021/datasets), i.e., for creating features and grouping interactions features by sessions. But for your convenience we also provide the [pre-processed version of the datasets](https://drive.google.com/drive/folders/1fxZozQuwd4fieoD0lmcD3mQ2Siu62ilD?usp=sharing) for download, so that you jump directly into running experiments with Transformers4Rec. 

## Training and evaluation
In our paper we have performed hyperparameter tuning for each experiment group (dataset and algorithm pair), whose search space and best hyperparameters can be found in the paper [Online Appendix C](../Appendices/Appendix_C-Hyperparameters.md). 

The command lines to run each experiment group with the best hyperparameters using the original scripts can be found in the next sections.


## Steps to run the paper experiments
1) Download the preprocessed datasets from this [Google Drive](https://drive.google.com/drive/folders/1fxZozQuwd4fieoD0lmcD3mQ2Siu62ilD?usp=sharing)
2) Copy the datasets folder to a local path and set the `DATA_ROOT_PATH` environment variable to that path, for example:

```bash
DATA_ROOT_PATH=~/transformers4rec_paper_preproc_datasets_public
```

3) Go to the project root path: `cd Transformers4Rec/`
4) Create a `conda` environment and install the library dependencies, according to the instructions [here](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021/hf4rec)
4) Run the command of one of the following experiment groups (dataset, algorithm) to reproduce paper results. P.s. The reported numbers in the paper are the average of the metrics of 5 runs with different random seeds (`--seed`).
5) When you run the commands, it will be generated within `--output_dir` CSV files with the metrics for each evaluation time window and also the Average Over Time (AOT) metrics reported in the paper.  
6) You can chech the metrics plots with Tensorboard (logs saved in the output dir) and with [Weights & Biases](http://wandb.ai/) service (with a free account). For that you need to install wandb (`pip install wandb`) and run `wandb login` to provide your authentication key. The run stats and plots will be saved to the `huggingface` project on W&B service by default.

## Hardware environment
All neural-based models were trained for the paper experiments using a single GV100 GPU with 32 GB RAM, except the baseline Session k-NN models (V-SkNN, STAN and VSTAN) which use only CPUs.

# REES46 ECOMMERCE DATASET

## BASELINES

### V-SKNN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --eval_on_last_item_seq_only --model_type vsknn --eval_baseline_cpu_parallel --workers_count 7 --vsknn-k 600 --vsknn-sample_size 2500 --vsknn-weighting same --vsknn-weighting_score linear --vsknn-idf_weighting 10 --vsknn-remind True --vsknn-push_reminders True  --eval_on_test_set --seed 100 
```

### STAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --eval_on_last_item_seq_only --model_type stan --eval_baseline_cpu_parallel --workers_count 2 --stan-k 500 --stan-sample_size 10000 --stan-lambda_spw 5.49 --stan-lambda_snh 100 --stan-lambda_inh 1.3725 --stan-remind True --eval_on_test_set --seed 100 
```

### VSTAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type vstan --eval_baseline_cpu_parallel --workers_count 2 --vstan-k 1300 --vstan-sample_size 8500 --vstan-lambda_spw 5.49 --vstan-lambda_snh 80 --vstan-lambda_inh 2.745 --vstan-lambda_ipw 5.49 --vstan-lambda_idf 5 --vstan-remind True --eval_on_test_set --seed 100 
```

### GRU4Rec (FT)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 0 --gru4rec-batch_size 192 --gru4rec-learning_rate 0.02987583000164429 --gru4rec-dropout_p_hidden 0.2 --gru4rec-layers 384 --gru4rec-embedding 384 --gru4rec-constrained_embedding True --gru4rec-momentum 0.006354221780881474 --gru4rec-final_act linear --gru4rec-loss bpr-max --eval_on_test_set --seed 100 
```

### GRU4Rec (SWT)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 6 --gru4rec-batch_size 256 --gru4rec-learning_rate 0.09985796371352657 --gru4rec-dropout_p_hidden 0.0 --gru4rec-layers 320 --gru4rec-embedding 256 --gru4rec-constrained_embedding True --gru4rec-momentum 0.008077857652231124 --gru4rec-final_act linear --gru4rec-loss top1-max --eval_on_test_set --seed 100 
```

### GRU

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type gru --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 448 --learning_rate 0.0007107976722774954 --dropout 0.0 --input_dropout 0.2 --weight_decay 4.0070030423993165e-06 --d_model 128 --item_embedding_dim 384 --n_layer 1 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.1 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.095  --eval_on_test_set --seed 100 
```


## TRANSFORMERS WITH ONLY ITEM ID FEATURE - ORIGINAL TRAINING APPROACHES (RQ1)


### GPT-2 (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 384 --learning_rate 0.0008781937894379981 --dropout 0.2 --input_dropout 0.4 --weight_decay 1.4901138106122045e-05 --d_model 128 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.9 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.03 --other_embeddings_init_std 0.034999999999999996 --eval_on_test_set --seed 100 
```

### Transformer-XL (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type transfoxl --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 512 --learning_rate 0.001007765821083962 --dropout 0.1 --input_dropout 0.30000000000000004 --weight_decay 1.0673054163921092e-06 --d_model 448 --item_embedding_dim 320 --n_layer 1 --n_head 1 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.01 --eval_on_test_set --seed 100 
```

### BERT (MLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type albert --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --mlm --num_hidden_groups -1 --inner_group_num 1 --per_device_train_batch_size 192 --learning_rate 0.0004904752786458524 --dropout 0.0 --input_dropout 0.1 --weight_decay 9.565968888623912e-05 --d_model 320 --item_embedding_dim 320 --n_layer 2 --n_head 8 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.025 --mlm_probability 0.6000000000000001 --eval_on_test_set --seed 100 
```

### ELECTRA (RTD)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type electra --mlm --rtd --rtd_tied_generator --rtd_sample_from_batch --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --mlm --rtd --rtd_tied_generator --per_device_train_batch_size 320 --learning_rate 0.0005122969428899831 --dropout 0.0 --input_dropout 0.1 --weight_decay 8.201103665795842e-06 --d_model 384 --item_embedding_dim 448 --n_layer 2 --n_head 2 --label_smoothing 0.5 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.02 --mlm_probability 0.4 --rtd_discriminator_loss_weight 1 --eval_on_test_set --seed 100 
```

### XLNet (PLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --plm --per_device_train_batch_size 320 --learning_rate 0.0003387925502203725 --dropout 0.0 --input_dropout 0.2 --weight_decay 2.1769664191492473e-05 --d_model 384 --item_embedding_dim 384 --n_layer 4 --n_head 16 --label_smoothing 0.7000000000000001 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.005 --plm_probability 0.5 --plm_max_span_length 3 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - XLNET WITH ALTERNATIVE TRAINING APPROACHES (RQ2)


### XLNet (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type uni --per_device_train_batch_size 192 --learning_rate 0.002029182148373514 --dropout 0.30000000000000004 --input_dropout 0.0 --weight_decay 1.5194998098932258e-05 --d_model 320 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.1 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.01  --eval_on_test_set --seed 100 
```

### XLNET (RTD)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --rtd --rtd_tied_generator --rtd_generator_loss_weight 1 --rtd_sample_from_batch --per_device_train_batch_size 384 --learning_rate 0.0004549311268958705 --dropout 0.0 --input_dropout 0.2 --weight_decay 7.698589592102765e-06 --d_model 384 --item_embedding_dim 448 --n_layer 3 --n_head 16 --label_smoothing 0.2 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.065 --mlm_probability 0.5 --rtd_discriminator_loss_weight 1  --eval_on_test_set --seed 101  
```

### XLNet (MLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --per_device_train_batch_size 192 --learning_rate 0.0006667377132554976 --dropout 0.0 --input_dropout 0.1 --weight_decay 3.910060265627374e-05 --d_model 192 --item_embedding_dim 448 --n_layer 3 --n_head 16 --label_smoothing 0.0 --stochastic_shared_embeddings_replacement_prob 0.1 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.02 --mlm_probability 0.30000000000000004 --eval_on_test_set --seed 100 
```


## TRANSFORMERS WITH MULTIPLE FEATURES - XLNet (MLM) (RQ3)

### CONCAT

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 256 --learning_rate 0.00020171456712823088 --dropout 0.0 --input_dropout 0.0 --weight_decay 2.747484129693843e-05 --d_model 448 --item_embedding_dim 448 --n_layer 2 --n_head 8 --label_smoothing 0.5 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.015 --mlm_probability 0.1 --embedding_dim_from_cardinality_multiplier 3.0 --eval_on_test_set --seed 100 
```

### CONCAT + SOFT ONE-HOT ENCODING
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 256 --learning_rate 0.00034029107417129616 --dropout 0.0 --input_dropout 0.1 --weight_decay 3.168336235732841e-05 --d_model 448 --item_embedding_dim 384 --n_layer 2 --n_head 8 --label_smoothing 0.6000000000000001 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.085 --mlm_probability 0.30000000000000004 --embedding_dim_from_cardinality_multiplier 1.0 --numeric_features_project_to_embedding_dim 20 --numeric_features_soft_one_hot_encoding_num_embeddings 5 --eval_on_test_set --seed 100 
```

### ELEMENTWISE
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/ecommerce --feature_config datasets/ecommerce_rees46/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 30 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --dataloader_drop_last --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation elementwise_sum_multiply_item_embedding --per_device_train_batch_size 384 --learning_rate 0.00038145834036682044 --dropout 0.0 --input_dropout 0.1 --weight_decay 3.161837643493792e-05 --d_model 448 --item_embedding_dim 448 --n_layer 3 --n_head 16 --label_smoothing 0.1 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.06 --mlm_probability 0.5 --embedding_dim_from_cardinality_multiplier 8.0 --eval_on_test_set --seed 100 
```








# YOOCHOOSE ECOMMERCE DATASET

## BASELINES


### V-SKNN
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid_ts.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type vsknn --eval_baseline_cpu_parallel --workers_count 2 --vsknn-k 500 --vsknn-sample_size 1000 --vsknn-weighting quadratic --vsknn-weighting_score quadratic --vsknn-idf_weighting 10 --vsknn-remind False --vsknn-push_reminders False --eval_on_test_set --seed 100 
```

### STAN
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid_ts.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type stan --eval_baseline_cpu_parallel --workers_count 2 --stan-k 950 --stan-sample_size 8000 --stan-lambda_spw 1e-05 --stan-lambda_snh 5 --stan-lambda_inh 1.915 --stan-remind False --eval_on_test_set --seed 100 
```

### VSTAN
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid_ts.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type vstan --eval_baseline_cpu_parallel --workers_count 2 --vstan-k 450 --vstan-sample_size 4500 --vstan-lambda_spw 0.9575 --vstan-lambda_snh 5 --vstan-lambda_inh 3.83 --vstan-lambda_ipw 0.47875 --vstan-lambda_idf 1 --vstan-remind False  --eval_on_test_set --seed 100 
```

### GRU4Rec (FT)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid_ts.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 0  --gru4rec-batch_size 128 --gru4rec-learning_rate 0.048359632063470666 --gru4rec-dropout_p_hidden 0.30000000000000004 --gru4rec-layers 320 --gru4rec-embedding 256 --gru4rec-constrained_embedding True --gru4rec-momentum 0.024011023365380108 --gru4rec-final_act linear --gru4rec-loss top1-max --eval_on_test_set --seed 100 
```

### GRU4Rec (SWT)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid_ts.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 36  --gru4rec-batch_size 128 --gru4rec-learning_rate 0.025515299727070073 --gru4rec-dropout_p_hidden 0.2 --gru4rec-layers 384 --gru4rec-embedding 320 --gru4rec-constrained_embedding True --gru4rec-momentum 0.014195421804280536 --gru4rec-final_act linear --gru4rec-loss bpr-max --eval_on_test_set --seed 100 
```

### GRU
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type gru --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 384 --learning_rate 0.00034691438611908384 --dropout 0.30000000000000004 --input_dropout 0.2 --weight_decay 2.209927960504878e-06 --d_model 192 --item_embedding_dim 448 --n_layer 1 --label_smoothing 0.5 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.01 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - ORIGINAL TRAINING APPROACHES (RQ1)

### GPT-2 (CLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 320 --learning_rate 0.00026223148256677574 --dropout 0.1 --input_dropout 0.30000000000000004 --weight_decay 2.916505538650302e-06 --d_model 192 --item_embedding_dim 448 --n_layer 2 --n_head 1 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.05 --other_embeddings_init_std 0.049999999999999996 --eval_on_test_set --seed 100 
```

### Transformer-XL (CLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type transfoxl --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 512 --learning_rate 0.0005964244796384939 --dropout 0.1 --input_dropout 0.0 --weight_decay 3.962727671279498e-06 --d_model 256 --item_embedding_dim 320 --n_layer 1 --n_head 1 --label_smoothing 0.8 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.015 --eval_on_test_set --seed 100 
```

### BERT (MLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type albert --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --mlm --num_hidden_groups -1 --inner_group_num 1 --per_device_train_batch_size 512 --learning_rate 0.000290721137663469 --dropout 0.0 --input_dropout 0.1 --weight_decay 1.8538139842177703e-06 --d_model 448 --item_embedding_dim 448 --n_layer 4 --n_head 1 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.055 --mlm_probability 0.30000000000000004 --eval_on_test_set --seed 100 
```

### ELECTRA (RTD)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type electra --mlm --rtd --rtd_tied_generator --rtd_sample_from_batch --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh  --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 320 --learning_rate 0.00033695501891338095 --dropout 0.0 --input_dropout 0.0 --weight_decay 3.199522233709421e-06 --d_model 384 --item_embedding_dim 320 --n_layer 2 --n_head 16 --label_smoothing 0.7000000000000001 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.08 --mlm_probability 0.2 --rtd_discriminator_loss_weight 1  --eval_on_test_set --seed 100 
```

### XLNET (PLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --plm --num_train_epochs 10 --per_device_train_batch_size 384 --learning_rate 0.00019342122947416297 --dropout 0.0 --input_dropout 0.1 --weight_decay 7.791224655534517e-06 --d_model 320 --item_embedding_dim 448 --n_layer 1 --n_head 2 --label_smoothing 0.5 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.09000000000000001 --plm_probability 0.7000000000000001 --plm_max_span_length 4 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - XLNET WITH ALTERNATIVE TRAINING APPROACHES (RQ2)


### XLNet (CLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type uni --per_device_train_batch_size 384 --learning_rate 0.0011783394797626335 --dropout 0.30000000000000004 --input_dropout 0.1 --weight_decay 4.130481597461282e-06 --d_model 448 --item_embedding_dim 384 --n_layer 2 --n_head 1 --label_smoothing 0.6000000000000001 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.01 --eval_on_test_set --seed 100 
```

### XLNet (RTD)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --rtd --rtd_tied_generator --rtd_generator_loss_weight 1 --rtd_sample_from_batch --per_device_train_batch_size 320 --learning_rate 0.0002805236563092962 --dropout 0.0 --input_dropout 0.30000000000000004 --weight_decay 3.476641732095646e-06 --d_model 384 --item_embedding_dim 448 --n_layer 4 --n_head 4 --label_smoothing 0.30000000000000004 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.030000000000000002 --mlm_probability 0.30000000000000004 --rtd_discriminator_loss_weight 1 --eval_on_test_set --seed 100
```

### XLNet (MLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/yoochoose --feature_config datasets/ecom_yoochoose/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 180 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --per_device_train_batch_size 384 --learning_rate 0.0005427417424896008 --dropout 0.0 --input_dropout 0.30000000000000004 --weight_decay 5.862734450920207e-06 --d_model 320 --item_embedding_dim 448 --n_layer 2 --n_head 8 --label_smoothing 0.6000000000000001 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.045 --mlm_probability 0.30000000000000004 --eval_on_test_set --seed 100 
```












# ADRESSA NEWS DATASET
 
## BASELINES

### V-SKNN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type vsknn --eval_baseline_cpu_parallel --workers_count 2 --vsknn-k 1200 --vsknn-sample_size 500 --vsknn-weighting quadratic --vsknn-weighting_score quadratic --vsknn-idf_weighting False --vsknn-remind False --vsknn-push_reminders False --eval_on_test_set --seed 100 
```

### VSTAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type vstan --eval_baseline_cpu_parallel --workers_count 2 --vstan-k 1300 --vstan-sample_size 1000 --vstan-lambda_spw 0.355 --vstan-lambda_snh 100 --vstan-lambda_inh 0.355 --vstan-lambda_ipw 2.84 --vstan-lambda_idf False --vstan-remind False --eval_on_test_set --seed 100 
```

### STAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type stan --eval_baseline_cpu_parallel --workers_count 2 --stan-k 1850 --stan-sample_size 500 --stan-lambda_spw 0.355 --stan-lambda_snh 5 --stan-lambda_inh 0.71 --stan-remind False --eval_on_test_set --seed 100 
```

### GRU4Rec (FT)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 0  --gru4rec-batch_size 320 --gru4rec-learning_rate 0.006776399704055072 --gru4rec-dropout_p_hidden 0.1 --gru4rec-layers 448 --gru4rec-embedding 256 --gru4rec-constrained_embedding True --gru4rec-momentum 0.0227154672842521 --gru4rec-final_act tanh --gru4rec-loss top1-max --eval_on_test_set --seed 100 
```

### GRU4Rec (SWT)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 72  --gru4rec-batch_size 512 --gru4rec-learning_rate 0.006604778881284094 --gru4rec-dropout_p_hidden 0.1 --gru4rec-layers 448 --gru4rec-embedding 320 --gru4rec-constrained_embedding True --gru4rec-momentum 0.013164410950880598 --gru4rec-final_act tanh --gru4rec-loss top1-max --eval_on_test_set --seed 100 
```

### GRU

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type gru --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 192 --learning_rate 0.000325395075539853 --dropout 0.30000000000000004 --input_dropout 0.1 --weight_decay 7.835870268598259e-05 --d_model 320 --item_embedding_dim 384 --n_layer 1 --label_smoothing 0.9 --stochastic_shared_embeddings_replacement_prob 0.04 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.085  --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - ORIGINAL TRAINING APPROACHES (RQ1)

### GPT-2 (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 192 --learning_rate 0.0008384381629906059 --dropout 0.4 --input_dropout 0.1 --weight_decay 2.0942953734125855e-05 --d_model 64 --item_embedding_dim 448 --n_layer 1 --n_head 2 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.1  --eval_on_test_set --seed 100 
```

### Transformer-XL (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type transfoxl --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 128 --learning_rate 0.00011178008843697089 --dropout 0.0 --input_dropout 0.4 --weight_decay 2.4487884902547452e-05 --d_model 320 --item_embedding_dim 448 --n_layer 2 --n_head 1 --label_smoothing 0.1 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.095 --eval_on_test_set --seed 100 
```

### BERT (MLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type albert --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --mlm --num_hidden_groups -1 --inner_group_num 1 --per_device_train_batch_size 192 --learning_rate 0.00019043055894413512 --dropout 0.0 --input_dropout 0.2 --weight_decay 2.1271069165192244e-05 --d_model 192 --item_embedding_dim 448 --n_layer 4 --n_head 8 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.030000000000000002 --mlm_probability 0.4 --eval_on_test_set --seed 100 
```

### ELECTRA (RTD)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type electra --mlm --rtd --rtd_tied_generator --rtd_sample_from_batch --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 128 --learning_rate 0.0002434246357751905 --dropout 0.30000000000000004 --input_dropout 0.0 --weight_decay 4.61286191508748e-06 --d_model 256 --item_embedding_dim 320 --n_layer 3 --n_head 8 --label_smoothing 0.30000000000000004 --item_id_embeddings_init_std 0.05 --other_embeddings_init_std 0.06 --mlm_probability 0.30000000000000004 --rtd_discriminator_loss_weight 1  --eval_on_test_set --seed 100  
```

### XLNET (PLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --plm --per_device_train_batch_size 192 --learning_rate 0.0002322934970378783 --dropout 0.1 --input_dropout 0.30000000000000004 --weight_decay 9.323863902137911e-05 --d_model 448 --item_embedding_dim 256 --n_layer 1 --n_head 1 --label_smoothing 0.2 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.09000000000000001 --num_train_epochs 15 --plm_probability 0.4 --plm_max_span_length 2 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - XLNET WITH ALTERNATIVE TRAINING APPROACHES (RQ2)

### XLNET (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type uni --per_device_train_batch_size 192 --learning_rate 0.0002668717028466931 --dropout 0.1 --input_dropout 0.4 --weight_decay 5.778017202444228e-06 --d_model 384 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.015 --eval_on_test_set --seed 100 
```

### XLNET (RTD)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --rtd --rtd_tied_generator --rtd_generator_loss_weight 1 --rtd_sample_from_batch --per_device_train_batch_size 256 --learning_rate 0.00017616008201600807 --dropout 0.0 --input_dropout 0.4 --weight_decay 1.1993723218564998e-06 --d_model 448 --item_embedding_dim 448 --n_layer 4 --n_head 1 --label_smoothing 0.2 --item_id_embeddings_init_std 0.09 --other_embeddings_init_std 0.049999999999999996 --mlm_probability 0.30000000000000004 --rtd_discriminator_loss_weight 1  --eval_on_test_set --seed 100 
```

### XLNET (MLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --per_device_train_batch_size 192 --learning_rate 0.00018955890698218382 --dropout 0.0 --input_dropout 0.5 --weight_decay 1.3143394122186089e-05 --d_model 384 --item_embedding_dim 384 --n_layer 3 --n_head 1 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.04 --mlm_probability 0.2 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH MULTIPLE FEATURES - XLNet (MLM) (RQ3)


### CONCAT

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 128 --learning_rate 0.0003431187996637234 --dropout 0.0 --input_dropout 0.30000000000000004 --weight_decay 5.883403254480229e-06 --d_model 192 --item_embedding_dim 384 --n_layer 2 --n_head 4 --label_smoothing 0.4 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.030000000000000002 --mlm_probability 0.2 --embedding_dim_from_cardinality_multiplier 4.0 --eval_on_test_set --seed 100 
```

### CONCAT + SOFT ONE-HOT ENCODING

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 128 --learning_rate 0.00043766058303741157 --dropout 0.0 --input_dropout 0.2 --weight_decay 1.8848603561851065e-05 --d_model 256 --item_embedding_dim 320 --n_layer 1 --n_head 8 --label_smoothing 0.9 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.06 --mlm_probability 0.4 --embedding_dim_from_cardinality_multiplier 7.0 --numeric_features_project_to_embedding_dim 20 --numeric_features_soft_one_hot_encoding_num_embeddings 20 --eval_on_test_set --seed 100 
```

### ELEMENTWISE

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/adressa --feature_config datasets/news_adressa/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation elementwise_sum_multiply_item_embedding --per_device_train_batch_size 192 --learning_rate 0.00020087285410843827 --dropout 0.0 --input_dropout 0.1 --weight_decay 2.150503366783766e-06 --d_model 384 --item_embedding_dim 448 --n_layer 1 --n_head 8 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.085 --mlm_probability 0.5 --embedding_dim_from_cardinality_multiplier 7.0 --eval_on_test_set --seed 100 
```


# G1 NEWS DATASET

## BASELINES

### V-SKNN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --eval_on_last_item_seq_only --model_type vsknn --eval_baseline_cpu_parallel --workers_count 7 --vsknn-k 800 --vsknn-sample_size 500 --vsknn-weighting quadratic --vsknn-weighting_score quadratic --vsknn-idf_weighting False --vsknn-remind False --vsknn-push_reminders True --eval_on_test_set --seed 100 
```

### STAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type stan --eval_baseline_cpu_parallel --workers_count 2 --stan-k 500 --stan-sample_size 500 --stan-lambda_spw 0.6725 --stan-lambda_snh 100 --stan-lambda_inh 0.6725 --stan-remind False --eval_on_test_set --seed 100 
```

### VSTAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type vstan --eval_baseline_cpu_parallel --workers_count 2 --vstan-k 1250 --vstan-sample_size 500 --vstan-lambda_spw 2.69 --vstan-lambda_snh 80 --vstan-lambda_inh 1.345 --vstan-lambda_ipw 0.33625 --vstan-lambda_idf False --vstan-remind False --eval_on_test_set --seed 100 
```

### GRU4Rec (FT)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 190 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 0  --gru4rec-batch_size 512 --gru4rec-learning_rate 0.0033908599219808163 --gru4rec-dropout_p_hidden 0.4 --gru4rec-layers 448 --gru4rec-embedding 320 --gru4rec-constrained_embedding True --gru4rec-momentum 0.003379534375729152 --gru4rec-final_act tanh --gru4rec-loss bpr-max --eval_on_test_set --seed 100 
```

### GRU4Rec (SWT)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.baselines.recsys_baselines_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --session_seq_length_max 20 --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4  --eval_on_last_item_seq_only --model_type gru4rec --gru4rec-n_epochs 10 --no_incremental_training --training_time_window_size 72  --gru4rec-batch_size 192 --gru4rec-learning_rate 0.0037281749997918803 --gru4rec-dropout_p_hidden 0.30000000000000004 --gru4rec-layers 384 --gru4rec-embedding 64 --gru4rec-constrained_embedding True --gru4rec-momentum 0.02357053155826095 --gru4rec-final_act tanh --gru4rec-loss top1-max --eval_on_test_set --seed 100 
```

### GRU

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type gru --num_train_epochs 10 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 192 --learning_rate 0.0006494976636110262 --dropout 0.1 --input_dropout 0.4 --weight_decay 6.1729942978535e-05 --d_model 128 --item_embedding_dim 448 --n_layer 1 --label_smoothing 0.7000000000000001 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.034999999999999996 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - ORIGINAL TRAINING APPROACHES (RQ1)

### GPT-2 (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type gpt2 --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 320 --learning_rate 0.0004451168155508206 --dropout 0.30000000000000004 --input_dropout 0.0 --weight_decay 5.6391828253625376e-05 --d_model 256 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.2 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.005 --eval_on_test_set --seed 100 
```

### Transformer-XL (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type transfoxl --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 192 --learning_rate 0.0003290060713041317 --dropout 0.0 --input_dropout 0.2 --weight_decay 1.7295468230514789e-06 --d_model 128 --item_embedding_dim 448 --n_layer 1 --n_head 8 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.03 --other_embeddings_init_std 0.030000000000000002 --eval_on_test_set --seed 100 
```

### BERT (MLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type albert --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --mlm --num_hidden_groups -1 --inner_group_num 1 --per_device_train_batch_size 128 --learning_rate 0.00018961089951392257 --dropout 0.0 --input_dropout 0.2 --weight_decay 1.6325081940093224e-05 --d_model 384 --item_embedding_dim 384 --n_layer 4 --n_head 2 --label_smoothing 0.7000000000000001 --stochastic_shared_embeddings_replacement_prob 0.06 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.01 --mlm_probability 0.2 --eval_on_test_set --seed 100 
```

### ELECTRA (RTD)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type electra --mlm --rtd --rtd_tied_generator --rtd_sample_from_batch --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --per_device_train_batch_size 128 --learning_rate 0.00014365473014542037 --dropout 0.0 --input_dropout 0.0 --weight_decay 1.8828426713949708e-05 --d_model 320 --item_embedding_dim 448 --n_layer 4 --n_head 2 --label_smoothing 0.5 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.025 --mlm_probability 0.30000000000000004 --rtd_discriminator_loss_weight 1 --eval_on_test_set --seed 100 
```

### XLNET (PLM)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --plm --per_device_train_batch_size 192 --learning_rate 0.0002623729053132054 --dropout 0.1 --input_dropout 0.2 --weight_decay 1.3263096129998752e-06 --d_model 256 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.8 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.1 --num_train_epochs 13 --plm_probability 0.5 --plm_max_span_length 4 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH ONLY ITEM ID FEATURE - XLNET WITH ALTERNATIVE TRAINING APPROACHES (RQ2)

### XLNET (CLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type uni --per_device_train_batch_size 320 --learning_rate 0.0023217204777992236 --dropout 0.30000000000000004 --input_dropout 0.1 --weight_decay 8.183114311314714e-05 --d_model 128 --item_embedding_dim 448 --n_layer 1 --n_head 1 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.1 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.07500000000000001 --eval_on_test_set --seed 100 
```

### XLNET (RTD)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --rtd --rtd_tied_generator --rtd_generator_loss_weight 1 --rtd_sample_from_batch --per_device_train_batch_size 192 --learning_rate 0.00015230855761611193 --dropout 0.0 --input_dropout 0.2 --weight_decay 5.403368699221602e-05 --d_model 448 --item_embedding_dim 384 --n_layer 2 --n_head 8 --label_smoothing 0.30000000000000004 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.065 --mlm_probability 0.5 --rtd_discriminator_loss_weight 1 --eval_on_test_set --seed 100 
```

### XLNET (MLM)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_itemid.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --per_device_train_batch_size 128 --learning_rate 0.00014265447169107707 --dropout 0.0 --input_dropout 0.0 --weight_decay 8.086567340597752e-05 --d_model 384 --item_embedding_dim 384 --n_layer 4 --n_head 8 --label_smoothing 0.30000000000000004 --stochastic_shared_embeddings_replacement_prob 0.08 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.07500000000000001 --mlm_probability 0.30000000000000004 --eval_on_test_set --seed 100 
```

## TRANSFORMERS WITH MULTIPLE FEATURES - XLNet (MLM) (RQ3)

### CONCAT

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 448 --learning_rate 0.00027034845576974243 --dropout 0.0 --input_dropout 0.0 --weight_decay 1.5408334399513826e-05 --d_model 384 --item_embedding_dim 448 --n_layer 1 --n_head 8 --label_smoothing 0.5 --stochastic_shared_embeddings_replacement_prob 0.02 --item_id_embeddings_init_std 0.11 --other_embeddings_init_std 0.01 --mlm_probability 0.30000000000000004 --embedding_dim_from_cardinality_multiplier 2.0 --eval_on_test_set --seed 100 
```

### CONCAT + SOFT ONE-HOT ENCODING

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation concat --per_device_train_batch_size 128 --learning_rate 0.00022133218557087516 --dropout 0.0 --input_dropout 0.0 --weight_decay 6.4710116073868965e-06 --d_model 128 --item_embedding_dim 384 --n_layer 4 --n_head 16 --label_smoothing 0.4 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.13 --other_embeddings_init_std 0.08 --mlm_probability 0.2 --embedding_dim_from_cardinality_multiplier 1.0 --numeric_features_project_to_embedding_dim 10 --numeric_features_soft_one_hot_encoding_num_embeddings 20 --eval_on_test_set --seed 100 
```

### ELEMENTWISE

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hf4rec.recsys_main --output_dir ./tmp/ --overwrite_output_dir --do_train --do_eval --validate_every 10 --logging_steps 20 --save_steps 0 --data_path $DATA_ROOT_PATH/g1_news --feature_config datasets/news_g1/config/features/session_based_features_all.yaml --fp16 --data_loader_engine nvtabular --start_time_window_index 1 --final_time_window_index 380 --time_window_folder_pad_digits 4 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 10 --compute_metrics_each_n_steps 1 --session_seq_length_max 20 --eval_on_last_item_seq_only --mf_constrained_embeddings --layer_norm_featurewise --attn_type bi --mlm --input_features_aggregation elementwise_sum_multiply_item_embedding --per_device_train_batch_size 192 --learning_rate 0.00028950848604244507 --dropout 0.0 --input_dropout 0.2 --weight_decay 4.7585424526954705e-06 --d_model 320 --item_embedding_dim 384 --n_layer 3 --n_head 8 --label_smoothing 0.5 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.15 --other_embeddings_init_std 0.065 --mlm_probability 0.5 --embedding_dim_from_cardinality_multiplier 9.0 --eval_on_test_set --seed 100 
```


