#  Appendix C - Hypertuning - Search space and best hyperparameters

In this appendix we provide 
- average runtime in minutes of the 100 hypertuning trials by algorithm and dataset
- the detailed search space utilized for hyperparameter tuning and the best hyperparameters found for each experiment group (composed by algorithm, training approach and dataset).

<br>

* [Average runtime (in minutes) of the 100 hypertuning trials by algorithm and dataset](#Table-1.-Average-runtime-(in-minutes)-of-the-100-hypertuning-trials-by-algorithm-and-dataset)
* [Hypertuning Search Space](#Hypertuning-Search-Space)
* [Best Hyperparameters per Algorithm](#Best-Hyperparameters-per-Algorithm)
    * Baselines
       * [GRU4REC (FT)](#GRU4REC-FT)
       * [GRU4REC (SWT)](#GRU4REC-SWT)
       * [V-SkNN](#V-SkNN)
       * [STAN](#STAN)
       * [VSTAN](#VSTAN)
    * Transformers with only item id feature
       * [GRU](#GRU)
       * [GPT2](#GPT2)
       * [TransformerXL](#TransformerXL)
       * [XLNet-CLM](#XLNet-CausalLM)
       * [XLNet-MLM](#XLNet-MLM)
       * [XLNet-PLM](#XLNet-PLM)
       * [XLNet-RTD](#XLNet-RTD)
       * [ELECTRA](#ELECTRA)
       * [ALBERT](#ALBERT)
    * XLNET MLM with side information features
       * [XLNet-MLM-all-concat](#XLNet-MLM-all-concat)
       * [XLNet-MLM-all-concat-numeric_soft_embedding](#XLNet-MLM-all-concat-numeric_soft_embedding)
       * [XLNet-MLM-all-elementwise](#XLNet-MLM-all-elementwise)

#### Table 1. Average runtime (in minutes) of the 100 hypertuning trials by algorithm and dataset
<table class="table-table">
<thead><tr class="table-firstrow"><th colspan=2>&nbsp;</th><th colspan=2>REES46 eCommerce</th><th colspan=2>YOOCHOOSE eCommerce</th><th colspan=2>G1 news</th><th colspan=2>ADRESSA news</th></tr></thead><tbody>
   <tr><td>&nbsp;</td><td><b>Number of sliding windows</b></td><td colspan=2><p align='center'>15 days</p></td><td colspan=2><p align='center'>90 days</p></td><td colspan=2>190 hours (~8 days)</td><td colspan=2>190 hours (~8 days)</td></tr>
 <tr><td>&nbsp;</td><td><b>Algorithm</b></td><td>Avg.</td><td>Std. Dev.</td><td>Avg.</td><td>Std. Dev.</td><td>Avg.</td><td>Std. Dev.</td><td>Avg.</td><td>Std. Dev.</td></tr>
 <tr><td rowspan=6>Baselines</td><td>V-SkNN</td><td>191.4</td><td>15.5</td><td>316.3</td><td>107.5</td><td>282.6</td><td>88.2</td><td>163.0</td><td>95.2</td></tr>
 <tr><td>STAN</td><td>221.3</td><td>30.0</td><td>378.4</td><td>55.6</td><td>101.2</td><td>12.9</td><td>92.6</td><td>13.2</td></tr>
 <tr><td>VSTAN</td><td>293.3</td><td>59.0</td><td>412.2</td><td>55.9</td><td>128.7</td><td>20.1</td><td>105.8</td><td>17.1</td></tr>
 <tr><td>GRU4Rec (FT)</td><td>163.0</td><td>17.5</td><td>756.3</td><td>258.3</td><td>173.1</td><td>31.4</td><td>145.1</td><td>31.6</td></tr>
 <tr><td>GRU4Rec (SWT)</td><td>146.2</td><td>15.8</td><td>497.0</td><td>157.1</td><td>138.6</td><td>30.8</td><td>101.2</td><td>12.5</td></tr>
 <tr><td>GRU</td><td>148.3</td><td>25.7</td><td>122.3</td><td>35.8</td><td>63.5</td><td>10.4</td><td>51.8</td><td>8.5</td></tr>
 <tr><td rowspan=8>Transformers with only the item id feature</td><td>GPT-2 (CLM)</td><td>133.7</td><td>20.7</td><td>94.4</td><td>26.6</td><td>47.5</td><td>9.2</td><td>54.8</td><td>9.8</td></tr>
 <tr><td>Transformer-XL (CLM)</td><td>108.9</td><td>28.0</td><td>125.1</td><td>37.9</td><td>56.7</td><td>11.7</td><td>69.4</td><td>15.5</td></tr>
 <tr><td>ALBERT (MLM)</td><td>116.8</td><td>36.0</td><td>116.1</td><td>33.8</td><td>67.1</td><td>16.9</td><td>59.2</td><td>17.6</td></tr>
 <tr><td>Electra (RTD)</td><td>109.9</td><td>28.3</td><td>125.1</td><td>46.5</td><td>88.2</td><td>18.5</td><td>62.1</td><td>19.2</td></tr>
 <tr><td>XLNet (PLM)</td><td>430.6</td><td>50.0</td><td>756.3</td><td>91.8</td><td>197.3</td><td>35.5</td><td>205.1</td><td>36.8</td></tr>
 <tr><td>XLNet (CLM)</td><td>137.9</td><td>24.0</td><td>139.8</td><td>57.8</td><td>54.4</td><td>11.9</td><td>62.1</td><td>11.2</td></tr>
 <tr><td>XLNET(RTD)</td><td>188.4</td><td>52.3</td><td>257.7</td><td>106.2</td><td>74.6</td><td>21.6</td><td>69.7</td><td>17.3</td></tr>
 <tr><td>XLNet (MLM)</td><td>104.8</td><td>40.5</td><td>120.8</td><td>42.2</td><td>63.5</td><td>17.2</td><td>63.3</td><td>16.2</td></tr>
 <tr><td rowspan=3>Transformers with side information features</td><td>Concatenation merge</td><td>142.7</td><td>42.8</td><td>-</td><td>-</td><td>66.9</td><td>17.7</td><td>70.1</td><td>20.7</td></tr>
 <tr><td>Concatenation merge with numericals using Soft-One Hot Encoding</td><td>173.9</td><td>46.5</td><td>-</td><td>-</td><td>69.0</td><td>21.4</td><td>80.8</td><td>19.8</td></tr>
 <tr><td>Element-wise merge</td><td>127.7</td><td>26.2</td><td>-</td><td>-</td><td>66.8</td><td>15.4</td><td>56.3</td><td>16.4</td></tr>
</tbody></table>

Notes:
- Each hypertuning trial performs the full incremental training and evaluation pipeline for a number of sliding windows for each dataset, described in the first row of the spreadhseet								
- All experiments were performed in a machine instance type with 8 CPU cores, 50 GB RAM and 1 V100 GPU with 32 GB.								
- The training implementation of V-SkNN, STAN and VSTAN baselines is CPU-based; all other algorithms were trained on GPU. The evaluation of the Session k-NN methods and GRU4Rec was performed using CPU multi-processing, and all other algorithms were evaluated using GPU.								

## Hypertuning Search Space

<!DOCTYPE html>
<html>
<body>
<h3>Table 2. Algorithms using the Transformers4Rec Meta-Architecture - Transformers and GRU baseline - using only the item id feature</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Hyperparameter Name</th><th>Search space</th><th>Sampling Distribution</th></tr><thead><tbody>
 <tr><td rowspan=30>Common parameters </td><td rowspan=18>fixed</td><td>inp_merge</td><td>mlp</td><td><center>-</center></td></tr>
 <tr><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td>loss_type</td><td>cross_entropy</td><td><center>-</center></td></tr>
 <tr><td>model_type</td><td>gpt2, transfoxl, xlnet, albert, electra, gru (baseline)</td><td><center>-</center></td></tr>
 <tr><td>mf_constrained_embeddings</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>per_device_eval_batch_size</td><td>512</td><td><center>-</center></td></tr>
 <tr><td>tf_out_activation</td><td>tanh</td><td><center>-</center></td></tr>
 <tr><td>similarity_type</td><td>concat_mlp)</td><td><center>-</center></td></tr>
 <tr><td>dataloader_drop_last </td><td>False</td><td><center>-</center></td></tr>
 <tr><td>dataloader_drop_last (for large ecommerce)</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>compute_metrics_each_n_steps</td><td>1</td><td><center>-</center></td></tr>
 <tr><td>eval_on_last_item_seq_only</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>learning_rate_schedule</td><td>linear_with_warmup</td><td><center>-</center></td></tr>
 <tr><td>learning_rate_warmup_steps</td><td>0</td><td><center>-</center></td></tr>
 <tr><td>layer_norm_all_features</td><td>False</td><td><center>-</center></td></tr>
 <tr><td>layer_norm_featurewise</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>num_train_epochs</td><td>10</td><td><center>-</center></td></tr>
 <tr><td>session_seq_length_max</td><td>20</td><td><center>-</center></td></tr>
 <tr><td rowspan=12>hypertuning</td><td>d_model</td><td>[64,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>item_embedding_dim</td><td>[64,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>n_layer</td><td>[1,4]</td><td>int_uniform</td></tr>
 <tr><td>n_head</td><td>[1, 2, 4, 8, 16]</td><td>categorical</td></tr>
 <tr><td>input_dropout</td><td>[0, 0.5]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>discrete_uniform (step 0.1)</td><td>[0, 0,5]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>learning_rate</td><td>[0.0001, 0.01]</td><td>log_uniform</td></tr>
 <tr><td>weight_decay</td><td>[0.000001, 0.001]</td><td>log_uniform</td></tr>
 <tr><td>per_device_train_batch_size</td><td>[128, 512]</td><td>int_uniform (steps 64)</td></tr>
 <tr><td>label_smoothing</td><td>[0, 0.9]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>[0.01, 0.15]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>GRU</td><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>GPT2</td><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>TransformerXL</td><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td rowspan=2>XLNet-CausalLM</td><td>fixed</td><td>attn_type</td><td>uni</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td rowspan=4>XLNet-MLM</td><td rowspan=2>fixed</td><td>mlm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>attn_type</td><td>bi</td><td><center>-</center></td></tr>
 <tr><td rowspan=2>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>mlm_probability</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td rowspan=7>XLNet-PLM</td><td rowspan=3>fixed</td><td>plm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>attn_type</td><td>bi</td><td><center>-</center></td></tr>
 <tr><td>plm_mask_input</td><td>False</td><td><center>-</center></td></tr>
 <tr><td rowspan=4>hypertuning</td><td>plm_probability (for ecommerce dataset)</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>plm_max_span_length (for ecommerce datasets)</td><td>[2, 6]</td><td>int_uniform</td></tr>
 <tr><td>plm_probability (for news datasets)</td><td>[0.4, 0.8]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>plm_max_span_length (for news datasets)</td><td>[1, 4]</td><td>int_uniform</td></tr>
 <tr><td rowspan=7>Electra-RTD</td><td rowspan=5>fixed</td><td>rtd</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>mlm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>rtd_tied_generator</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>rtd_use_batch_interaction</td><td>False</td><td><center>-</center></td></tr>
 <tr><td>rtd_sample_from_batch</td><td>True</td><td><center>-</center></td></tr>
 <tr><td rowspan=2>hypertuning</td><td>rtd_discriminator_loss_weight</td><td>[1, 10, 20, 30, 40, 50]</td><td>categorical</td></tr>
 <tr><td>mlm_probability</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td rowspan=5>ALBERT*</td><td rowspan=3>fixed</td><td>mlm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>inner_group_num</td><td>1</td><td><center>-</center></td></tr>
 <tr><td>num_hidden_groups</td><td>-1</td><td><center>-</center></td></tr>
 <tr><td rowspan=2>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>mlm_probability</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
</tbody></table>
*In our experiments, we fixed the parameters “inner_group_num” and “num_hidden_groups” to 1 and -1, respectively. Under this configuration, the layers are not sharing the weights which is equivalent to BERT architecture.<br>
<br>
<h3>Table 3. XLNet (MLM) - Additional hyperparameters when using side information</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Hyperparameter Name</th><th>Search Space</th><th>Sampling Distribution</th></tr></thead><tbody>
 <tr><td rowspan=4>Common hyperparameters</td><td>fixed</td><td>layer_norm_all_features</td><td>FALSE</td><td><center>-</center></td></tr>
 <tr><td>fixed</td><td>layer_norm_featurewise</td><td>TRUE</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>other_embeddings_init_std</td><td>[0.005, 0.10]</td><td>discrete_uniform (step 0.005)</td></tr>
 <tr><td>hypertuning</td><td>embedding_dim_from_cardinality_multiplier</td><td>[1.0, 10.0]</td><td>discrete_uniform (step 1.0)</td></tr>
 <tr><td>Concatenation merge-Numericals features as scalars</td><td>fixed</td><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td rowspan=3>Concatenation merge-Numerical features-Soft One-Hot Encoding</td><td>fixed</td><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>numeric_features_project_to_embedding_dim</td><td>[5, 55]</td><td>discrete_uniform (step 10)</td></tr>
 <tr><td>hypertuning</td><td>numeric_features_soft_one_hot_encoding_num_embeddings</td><td>[5, 55]</td><td>discrete_uniform (step 10)</td></tr>
 <tr><td>Element-wise merge</td><td>fixed</td><td>input_features_aggregation</td><td>elementwise_sum_multiply_item_embedding</td><td><center>-</center></td></tr>
</tbody></table>

<h3>Table 4. Baselines</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Hyperparameter Name</th><th>Search space</th><th>Sampling Distribution</th></tr><thead><tbody>
 <tr><td rowspan=3>Common parameters </td><td rowspan=3>fixed</td><td>model_type</td><td>gru4rec, vsknn, stan, vstan</td><td><center>-</center></td></tr>
<tr><td>eval_on_last_item_seq_only</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>session_seq_length_max</td><td>20</td><td><center>-</center></td></tr>
 <tr><td rowspan=13>GRU4REC</td><td rowspan=4>fixed</td><td>gru4rec-n_epochs</td><td>10</td><td><center>-</center></td></tr>
 <tr><td>no_incremental_training</td><td>True </td><td><center>-</center></td></tr>
 <tr><td>training_time_window_size (full-train)</td><td>0</td><td><center>-</center></td></tr>
 <tr><td>training_time_window_size (sliding 20%)</td><td>20% of the length of the dataset </td><td><center>-</center></td></tr>
 <tr><td rowspan=9>hypertuning</td><td>gru4rec-batch_size</td><td>[128, 512]</td><td>init_uniform(step 64)</td></tr>
 <tr><td>gru4rec-learning_rate</td><td>[0.0001, 0.1]</td><td>log_uniform</td></tr>
 <tr><td>gru4rec-dropout_p_hidden</td><td>[0, 0.5]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>gru4rec-layers</td><td>[64,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>gru4rec-embedding</td><td>[0,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>gru4rec-constrained_embedding</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td>gru4rec-momentum</td><td>[0, 0.5]</td><td>float_uniform (step 0.01)</td></tr>
 <tr><td>gru4rec-final_act</td><td>[elu-0.5, linear, tanh]</td><td>categorical</td></tr>
 <tr><td>gru4rec-loss</td><td>[bpr-max, top1-max]</td><td>categorical</td></tr>
 <tr><td rowspan=9>V-SkNN</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=7>hypertuning </td><td>vsknn-k</td><td>[50, 1500]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>vsknn-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>vsknn-weighting</td><td>[same, div, linear, quadratic, log]</td><td>categorical</td></tr>
 <tr><td>vsknn-weighting_score</td><td>[same, div, linear, quadratic, log]</td><td>categorical</td></tr>
 <tr><td>vsknn-idf_weighting</td><td>[1, 2, 5 ,10]</td><td>categorical</td></tr>
 <tr><td>vsknn-remind</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td>vsknn-push_reminders</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td rowspan=8>STAN</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=6>hypertuning</td><td>stan-k</td><td>[50, 2000]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>stan-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>stan-lambda_spw </td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>stan-lambda_snh</td><td>[2.5, 5, 10, 20, 40, 80,100]</td><td>categorical</td></tr>
 <tr><td>stan-lambda_inh</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>stan-remind</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td rowspan=10>VSTAN</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=8>hypertuning </td><td>vstan-k</td><td>[50, 2000]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>vstan-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>vstan-lambda_spw </td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_snh</td><td>[2.5, 5, 10, 20, 40, 80,100]</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_inh</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_ipw</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_idf</td><td>[1,2,5,10]</td><td>categorical</td></tr>
 <tr><td>vstan-remind</td><td>[True, False]</td><td>categorical</td></tr>
</tbody></table>
</body>
</html>
* Where L is the average session length

## Best Hyperparameters per Algorithm

## Baselines

### GRU4REC-FT

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>gru4rec-batch_size</td><td>192</td><td>128</td><td>512</td><td>320</td></tr>
 <tr><td>gru4rec-learning_rate</td><td>0.02987583</td><td>0.04835963206</td><td>0.003390859922</td><td>0.006776399704</td></tr>
 <tr><td>gru4rec-dropout_p_hidden</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.1</td></tr>
 <tr><td>gru4rec-layers</td><td>384</td><td>320</td><td>448</td><td>448</td></tr>
 <tr><td>gru4rec-embedding</td><td>384</td><td>256</td><td>320</td><td>256</td></tr>
 <tr><td>gru4rec-constrained_embedding</td><td>True</td><td>True</td><td>True</td><td>True</td></tr>
 <tr><td>gru4rec-momentum</td><td>0.0063542217809</td><td>0.0240110233654</td><td>0.0033795343757</td><td>0.0227154672843</td></tr>
 <tr><td>gru4rec-final_act</td><td>linear</td><td>linear</td><td>tanh</td><td>tanh</td></tr>
 <tr><td>gru4rec-loss</td><td>bpr-max</td><td> top1-max</td><td> bpr-max</td><td>top1-max</td></tr>
</tbody></table>

### GRU4REC-SWT

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>gru4rec-batch_size</td><td>256</td><td>128</td><td>192</td><td>512</td></tr>
 <tr><td>gru4rec-learning_rate</td><td>0.09985796371</td><td>0.02551529973</td><td>0.003728175</td><td>0.006604778881</td></tr>
 <tr><td>gru4rec-dropout_p_hidden</td><td>0.0</td><td>0.2</td><td>0.3</td><td>0.1</td></tr>
 <tr><td>gru4rec-layers</td><td>320</td><td>384</td><td>384</td><td>448</td></tr>
 <tr><td>gru4rec-embedding</td><td>256</td><td>320</td><td>64</td><td>320</td></tr>
 <tr><td>gru4rec-constrained_embedding</td><td>True</td><td>True</td><td>True</td><td>True</td></tr>
 <tr><td>gru4rec-momentum</td><td>0.0080778576522</td><td>0.0141954218043</td><td>0.0235705315583</td><td>0.0131644109509</td></tr>
 <tr><td>gru4rec-final_act</td><td>linear</td><td>linear</td><td>tanh</td><td>tanh</td></tr>
 <tr><td>gru4rec-loss</td><td> top1-max</td><td>bpr-max </td><td>top1-max</td><td>top1-max</td></tr>
 <tr><td>training_time_window_size</td><td>6</td><td>36</td><td>72</td><td>72</td></tr>
</tbody></table>

### V-SkNN

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>vsknn-k</td><td>600</td><td>500</td><td>800</td><td>1200</td></tr>
 <tr><td>vsknn-sample_size </td><td>2500</td><td>100</td><td>500</td><td>500</td></tr>
 <tr><td>vsknn-weighting</td><td>same</td><td>quadratic</td><td>quadratic</td><td>quadratic</td></tr>
 <tr><td>vsknn-weighting_score</td><td>linear</td><td>quadratic</td><td>quadratic</td><td>quadratic</td></tr>
 <tr><td>vsknn-idf_weighting</td><td>10</td><td>10</td><td>False</td><td>False</td></tr>
 <tr><td>vsknn-remind</td><td>True</td><td>False</td><td>False</td><td>False</td></tr>
 <tr><td>vsknn-push_reminders</td><td>True</td><td>False</td><td>True</td><td>False</td></tr>
</tbody></table>

### STAN

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>stan-k</td><td>500</td><td>950</td><td>500</td><td>1850</td></tr>
 <tr><td>stan-sample_size </td><td>10000</td><td>8000</td><td>500</td><td>500</td></tr>
 <tr><td>stan-lambda_spw </td><td>5.49</td><td>1.00E-05</td><td>0.6725</td><td>0.355</td></tr>
 <tr><td>stan-lambda_snh</td><td>100</td><td>5</td><td>100</td><td>5</td></tr>
 <tr><td>stan-lambda_inh</td><td>1.3725</td><td>1.915</td><td>0.6725</td><td>0.71</td></tr>
 <tr><td>stan-remind</td><td>True</td><td>False</td><td>False</td><td>False</td></tr>
</tbody></table>

### VSTAN

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>vstan-k</td><td>1300</td><td>450</td><td>1250</td><td>1300</td></tr>
 <tr><td>vstan-sample_size </td><td>8500</td><td>4500</td><td>500</td><td>1000</td></tr>
 <tr><td>vstan-lambda_spw </td><td>5.49</td><td>9.575E-01</td><td>2.69</td><td>0.355</td></tr>
 <tr><td>vstan-lambda_snh</td><td>80</td><td>5</td><td>80</td><td>100</td></tr>
 <tr><td>vstan-lambda_inh</td><td>2.745</td><td>3.83</td><td>1.345</td><td>0.355</td></tr>
 <tr><td>vstan-lambda_ipw</td><td>5.49</td><td>0.47875</td><td>0.33625</td><td>2.84</td></tr>
 <tr><td>vstan-lambda_idf</td><td>5</td><td>1</td><td>False</td><td>False</td></tr>
 <tr><td>vstan-remind</td><td>True</td><td>False</td><td>False</td><td>False</td></tr>
</tbody></table>


## Transformers with only item id feature

### GRU

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.1</td><td>0.0</td><td>0.08</td><td>0.04</td></tr>
 <tr><td>d_model</td><td>128</td><td>192</td><td>128</td><td>320</td></tr>
 <tr><td>item_embedding_dim</td><td>384</td><td>448</td><td>448</td><td>384</td></tr>
 <tr><td>n_layer</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
 <tr><td>input_dropout</td><td>0.2</td><td>0.2</td><td>0.4</td><td>0.1</td></tr>
 <tr><td>dropout</td><td>0.0</td><td>0.3</td><td>0.1</td><td>0.3</td></tr>
 <tr><td>learning_rate</td><td>0.0007107976723</td><td>0.0003469143861</td><td>0.0006494976636</td><td>0.0003253950755</td></tr>
 <tr><td>weight_decay</td><td>4.01E-06</td><td>2.21E-06</td><td>6.17E-05</td><td>7.84E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>448</td><td>384</td><td>192</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.9</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.09</td><td>0.15</td><td>0.11</td><td>0.11</td></tr>
</tbody></table>


### GPT2

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.0</td><td>0.08</td><td>0.06</td><td>0.08</td></tr>
 <tr><td>d_model</td><td>128</td><td>192</td><td>256</td><td>64</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td>448</td><td>448</td><td>448</td></tr>
 <tr><td>n_layer</td><td>1</td><td>2</td><td>1</td><td>1</td></tr>
 <tr><td>n_head</td><td>1</td><td>1</td><td>1</td><td>2</td></tr>
 <tr><td>input_dropout</td><td>0.4</td><td>0.3</td><td>0.0</td><td>0.1</td></tr>
 <tr><td>dropout</td><td>0.2</td><td>0.1</td><td>0.3</td><td>0.4</td></tr>
 <tr><td>learning_rate</td><td>0.0008781937894</td><td>0.0002622314826</td><td>0.0004451168156</td><td>0.000838438163</td></tr>
 <tr><td>weight_decay</td><td>1.49E-05</td><td>2.92E-06</td><td>5.64E-05</td><td>2.09E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>384</td><td>320</td><td>320</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.9</td><td>0.2</td><td>0.2</td><td>0.3</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.03</td><td>0.05</td><td>0.11</td><td>0.07</td></tr>
</tbody></table>

### TransformerXL

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.02</td><td>0.06</td><td>0.08</td><td>0.06</td></tr>
 <tr><td>d_model</td><td>448</td><td>256</td><td>128</td><td>320</td></tr>
 <tr><td>item_embedding_dim</td><td>320</td><td>320</td><td>448</td><td>448</td></tr>
 <tr><td>n_layer</td><td>1</td><td>1</td><td>1</td><td>2</td></tr>
 <tr><td>n_head</td><td>1</td><td>1</td><td>8</td><td>1</td></tr>
 <tr><td>input_dropout</td><td>0.3</td><td>0.0</td><td>0.2</td><td>0.4</td></tr>
 <tr><td>dropout</td><td>0.1</td><td>0.1</td><td>0</td><td>0</td></tr>
 <tr><td>learning_rate</td><td>0.001007765821</td><td>0.0005964244796</td><td>0.0003290060713</td><td>0.0001117800884</td></tr>
 <tr><td>weight_decay</td><td>1.07E-06</td><td>3.96E-06</td><td>1.73E-06</td><td>2.45E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>512</td><td>512</td><td>192</td><td>128</td></tr>
 <tr><td>label_smoothing</td><td>0.2</td><td>0.8</td><td>0.3</td><td>0.1</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.15</td><td>0.09</td><td>0.03</td><td>0.15</td></tr>
</tbody></table>

### XLNet-CausalLM

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>uni</td><td>uni</td><td>uni</td><td>uni</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.08</td><td>0.06</td><td>0.1</td><td>0.0</td></tr>
 <tr><td>d_model</td><td>320</td><td>448</td><td>128</td><td>384</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td>384</td><td>448</td><td>448</td></tr>
 <tr><td>n_layer</td><td>1</td><td>2</td><td>1</td><td>1</td></tr>
 <tr><td>n_head</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
 <tr><td>input_dropout</td><td>0.0</td><td>0.1</td><td>0.1</td><td>0.4</td></tr>
 <tr><td>dropout</td><td>0.3</td><td>0.3</td><td>0.3</td><td>0.1</td></tr>
 <tr><td>learning_rate</td><td>0.002029182148</td><td>0.00117833948</td><td>0.002321720478</td><td>0.0002668717028</td></tr>
 <tr><td>weight_decay</td><td>1.52E-05</td><td>4.13E-06</td><td>8.18E-05</td><td>5.78E-06</td></tr>
 <tr><td>per_device_train_batch_size</td><td>192</td><td>384</td><td>320</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.1</td><td>0.6</td><td>0.3</td><td>0.3</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.13</td><td>0.09</td><td>0.13</td><td>0.13</td></tr>
</tbody></table>

### XLNet-MLM
<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>bi</td><td>bi</td><td>bi</td><td>bi</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.1</td><td>0</td><td>0.08</td><td>0</td></tr>
 <tr><td>d_model</td><td>192</td><td>320</td><td>384</td><td>384</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td>448</td><td>384</td><td>384</td></tr>
 <tr><td>n_layer</td><td>3</td><td>2</td><td>4</td><td>3</td></tr>
 <tr><td>n_head</td><td>16</td><td>8</td><td>8</td><td>1</td></tr>
 <tr><td>input_dropout</td><td>0.1</td><td>0.3</td><td>0</td><td>0</td></tr>
 <tr><td>dropout</td><td>0</td><td>0</td><td>0</td><td>0.5</td></tr>
 <tr><td>learning_rate</td><td>0.0006667377133</td><td>0.0005427417425</td><td>0.0001426544717</td><td>0.000189558907</td></tr>
 <tr><td>weight_decay</td><td>3.91E-05</td><td>5.86E-06</td><td>8.09E-05</td><td>1.31E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>192</td><td>384</td><td>128</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.0</td><td>0.6</td><td>0.3</td><td>0.2</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.11</td><td>0.09</td><td>0.15</td><td>0.15</td></tr>
 <tr><td>mlm_probability</td><td>0.3</td><td>0.3</td><td>0.3</td><td>0.2</td></tr>
</tbody></table>

### XLNet-PLM

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>bi</td><td>bi</td><td>bi</td><td>bi</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.02</td><td>0</td><td>0</td><td>0</td></tr>
 <tr><td>d_model</td><td>384</td><td>320</td><td>256</td><td>256</td></tr>
 <tr><td>item_embedding_dim</td><td>384</td><td>448</td><td>448</td><td>448</td></tr>
 <tr><td>n_layer</td><td>4</td><td>1</td><td>1</td><td>1</td></tr>
 <tr><td>n_head</td><td>16</td><td>2</td><td>1</td><td>1</td></tr>
 <tr><td>input_dropout</td><td>0.2</td><td>0.1</td><td>0.2</td><td>0.3</td></tr>
 <tr><td>dropout</td><td>0</td><td>0</td><td>0.1</td><td>0.1</td></tr>
 <tr><td>learning_rate</td><td>0.0003387925502</td><td>0.0001934212295</td><td>0.0002623729053</td><td>2.32E-04</td></tr>
 <tr><td>weight_decay</td><td>2.18E-05</td><td>7.79E-06</td><td>1.33E-06</td><td>9.32E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>320</td><td>384</td><td>192</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.7</td><td>0.5</td><td>0.8</td><td>0.2</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.13</td><td>0.11</td><td>0.07</td><td>0.11</td></tr>
 <tr><td>plm_max_span_length</td><td>3</td><td>4</td><td>4</td><td>2</td></tr>
 <tr><td>plm_probability</td><td>0.5</td><td>0.7</td><td>0.5</td><td>0.4</td></tr>
</tbody></table>

### XLNet-RTD

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>bi</td><td>bi</td><td>bi</td><td>bi</td></tr>
 <tr><td>d_model</td><td>384</td><td>384</td><td>448</td><td>448</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td>448</td><td>384</td><td>448</td></tr>
 <tr><td>n_layer</td><td>3</td><td>4</td><td>2</td><td>4</td></tr>
 <tr><td>n_head</td><td>16</td><td>4</td><td>8</td><td>1</td></tr>
 <tr><td>input_dropout</td><td>0.2</td><td>0.3</td><td>0.2</td><td>0.4</td></tr>
 <tr><td>dropout</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr>
 <tr><td>learning_rate</td><td>0.0004549311269</td><td>0.0002805236563</td><td>0.0001523085576</td><td>1.76E-04</td></tr>
 <tr><td>weight_decay</td><td>7.70E-06</td><td>3.48E-06</td><td>5.40E-05</td><td>1.20E-06</td></tr>
 <tr><td>per_device_train_batch_size</td><td>384</td><td>320</td><td>192</td><td>256</td></tr>
 <tr><td>label_smoothing</td><td>0.2</td><td>0.3</td><td>0.3</td><td>0.2</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.15</td><td>0.11</td><td>0.15</td><td>0.09</td></tr>
 <tr><td>mlm_probability</td><td>0.5</td><td>0.3</td><td>0.5</td><td>0.3</td></tr>
 <tr><td>rtd_discriminator_loss_weight</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
</tbody></table>

### ELECTRA

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
 <tr><td>d_model</td><td>384</td><td>384</td><td>320</td><td>256</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td>320</td><td>448</td><td>320</td></tr>
 <tr><td>n_layer</td><td>2</td><td>2</td><td>4</td><td>3</td></tr>
 <tr><td>n_head</td><td>2</td><td>16</td><td>2</td><td>8</td></tr>
 <tr><td>input_dropout</td><td>0.1</td><td>0</td><td>0</td><td>0.4</td></tr>
 <tr><td>dropout</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
 <tr><td>learning_rate</td><td>0.0005122969429</td><td>0.0003369550189</td><td>0.0001436547301</td><td>1.76E-04</td></tr>
 <tr><td>weight_decay</td><td>8.20E-06</td><td>3.20E-06</td><td>1.88E-05</td><td>1.20E-06</td></tr>
 <tr><td>per_device_train_batch_size</td><td>320</td><td>320</td><td>128</td><td>256</td></tr>
 <tr><td>label_smoothing</td><td>0.5</td><td>0.8</td><td>0.5</td><td>0.3</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.09</td><td>0.09</td><td>0.15</td><td>0.05</td></tr>
 <tr><td>rtd_discriminator_loss_weight</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
 <tr><td>mlm_probability</td><td>0.4</td><td>0.2</td><td>0.3</td><td>0.3</td></tr>
</tbody></table>

### ALBERT

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0.06</td><td>0.02</td><td>0.06</td><td>0.08</td></tr>
 <tr><td>d_model</td><td>320</td><td>448</td><td>384</td><td>192</td></tr>
 <tr><td>item_embedding_dim</td><td>320</td><td>448</td><td>384</td><td>448</td></tr>
 <tr><td>n_layer</td><td>2</td><td>4</td><td>4</td><td>4</td></tr>
 <tr><td>n_head</td><td>8</td><td>1</td><td>2</td><td>8</td></tr>
 <tr><td>input_dropout</td><td>0.1</td><td>0.1</td><td>0.2</td><td>0.2</td></tr>
 <tr><td>dropout</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr>
 <tr><td>learning_rate</td><td>0.0004904752786</td><td>0.0002907211377</td><td>0.0001896108995</td><td>1.90E-04</td></tr>
 <tr><td>weight_decay</td><td>9.57E-05</td><td>1.85E-06</td><td>1.63E-05</td><td>2.13E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>192</td><td>512</td><td>128</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.2</td><td>0.3</td><td>0.7</td><td>0.2</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.11</td><td>0.07</td><td>0.15</td><td>0.15</td></tr>
 <tr><td>mlm_probability</td><td>0.6</td><td>0.3</td><td>0.2</td><td>0.4</td></tr>
</tbody></table>

## XLNET MLM with side information features

### XLNet-MLM-all-concat

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>bi</td><td><p align="center">-</p></td><td>bi</td><td>bi</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0</td><td><p align="center">-</p></td><td>0.02</td><td>0.02</td></tr>
 <tr><td>d_model</td><td>448.0</td><td><p align="center">-</p></td><td>384</td><td>192</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td><p align="center">-</p></td><td>448</td><td>384</td></tr>
 <tr><td>n_layer</td><td>2</td><td><p align="center">-</p></td><td>1</td><td>2</td></tr>
 <tr><td>n_head</td><td>8</td><td><p align="center">-</p></td><td>8</td><td>4</td></tr>
 <tr><td>input_dropout</td><td>0.0</td><td><p align="center">-</p></td><td>0</td><td>0.3</td></tr>
 <tr><td>dropout</td><td>0</td><td><p align="center">-</p></td><td>0</td><td>0.00E+00</td></tr>
 <tr><td>learning_rate</td><td>2.02E-04</td><td><p align="center">-</p></td><td>2.70E-04</td><td>3.43E-04</td></tr>
 <tr><td>weight_decay</td><td>2.75E-05</td><td><p align="center">-</p></td><td>1.54E-05</td><td>5.88E-06</td></tr>
 <tr><td>per_device_train_batch_size</td><td>256</td><td><p align="center">-</p></td><td>448</td><td>128</td></tr>
 <tr><td>label_smoothing</td><td>0.5</td><td><p align="center">-</p></td><td>0.5</td><td>0.4</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.09</td><td><p align="center">-</p></td><td>0.11</td><td>0.11</td></tr>
 <tr><td>other_embeddings_init_std</td><td>0.015</td><td><p align="center">-</p></td><td>0.01</td><td>0.03</td></tr>
 <tr><td>mlm_probability</td><td>0.1</td><td><p align="center">-</p></td><td>0.3</td><td>0.2</td></tr>
 <tr><td>embedding_dim_from_cardinality_multiplier</td><td>3</td><td><p align="center">-</p></td><td>2</td><td>4</td></tr>
</tbody></table>

### XLNet-MLM-all-concat-numeric_soft_embedding

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>bi</td><td><p align="center">-</p></td><td>bi</td><td>bi</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0</td><td><p align="center">-</p></td><td>0</td><td>0.06</td></tr>
 <tr><td>d_model</td><td>448.0</td><td><p align="center">-</p></td><td>128</td><td>256</td></tr>
 <tr><td>item_embedding_dim</td><td>384</td><td><p align="center">-</p></td><td>384</td><td>320</td></tr>
 <tr><td>n_layer</td><td>2</td><td><p align="center"><p align="center">-</p></p></td><td>4</td><td>1</td></tr>
 <tr><td>n_head</td><td>8</td><td><p align="center"><p align="center">-</p></p></td><td>16</td><td>8</td></tr>
 <tr><td>input_dropout</td><td>0.1</td><td><p align="center">-</p></td><td>0.0</td><td>0.2</td></tr>
 <tr><td>dropout</td><td>0</td><td><p align="center">-</p></td><td>0</td><td>0.00E+00</td></tr>
 <tr><td>learning_rate</td><td>3.40E-04</td><td><p align="center">-</p></td><td>2.21E-04</td><td>4.38E-04</td></tr>
 <tr><td>weight_decay</td><td>3.17E-05</td><td><p align="center">-</p></td><td>6.47E-06</td><td>1.88E-05</td></tr>
 <tr><td>per_device_train_batch_size</td><td>256</td><td><p align="center">-</p></td><td>128</td><td>128</td></tr>
 <tr><td>label_smoothing</td><td>0.6</td><td><p align="center"><p align="center">-</p></p></td><td>0.4</td><td>0.9</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.07</td><td><p align="center">-</p></td><td>0.13</td><td>0.13</td></tr>
 <tr><td>other_embeddings_init_std</td><td>0.085</td><td><p align="center">-</p></td><td>0.08</td><td>0.06</td></tr>
 <tr><td>mlm_probability</td><td>0.3</td><td><p align="center">-</p></td><td>0.2</td><td>0.4</td></tr>
   <tr><td>embedding_dim_from_cardinality_multiplier</td><td>1</td><td><p align="center">-</p></td><td>1</td><td>7</td></tr>
 <tr><td>numeric_features_project_to_embedding_dim</td><td>20</td><td><p align="center">-</p></td><td>10</td><td>20</td></tr>
 <tr><td>numeric_features_soft_one_hot_encoding_num_embeddings</td><td>5</td><td><p align="center">-</p></td><td>20</td><td>20</td></tr>
</tbody></table>


### XLNet-MLM-all-elementwise

<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>REES46 eCommerce</th><th>YOOCHOOSE eCommerce</th><th>G1 news</th><th>ADRESSA news</th></tr></thead><tbody>
 <tr><td>attn_type</td><td>bi</td><td><p align="center">-</p></td><td>bi</td><td>bi</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>0</td><td><p align="center">-</p></td><td>0</td><td>0.08</td></tr>
 <tr><td>d_model</td><td>448.0</td><td><p align="center">-</p></td><td>320</td><td>384</td></tr>
 <tr><td>item_embedding_dim</td><td>448</td><td><p align="center">-</p></td><td>384</td><td>448</td></tr>
 <tr><td>n_layer</td><td>3</td><td><p align="center">-</p></td><td>3</td><td>1</td></tr>
 <tr><td>n_head</td><td>16</td><td><p align="center">-</p></td><td>8</td><td>8</td></tr>
 <tr><td>input_dropout</td><td>0.1</td><td><p align="center">-</p></td><td>0.2</td><td>0.1</td></tr>
 <tr><td>dropout</td><td>0</td><td><p align="center">-</p></td><td>0</td><td>0.00E+00</td></tr>
 <tr><td>learning_rate</td><td>3.81E-04</td><td><p align="center">-</p></td><td>2.90E-04</td><td>2.01E-04</td></tr>
 <tr><td>weight_decay</td><td>3.16E-05</td><td><p align="center">-</p></td><td>4.76E-06</td><td>2.15E-06</td></tr>
 <tr><td>per_device_train_batch_size</td><td>384</td><td><p align="center">-</p></td><td>192</td><td>192</td></tr>
 <tr><td>label_smoothing</td><td>0.1</td><td><p align="center">-</p></td><td>0.5</td><td>0.2</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>0.13</td><td><p align="center">-</p></td><td>0.15</td><td>0.07</td></tr>
 <tr><td>other_embeddings_init_std</td><td>0.06</td><td><p align="center">-</p></td><td>0.065</td><td>0.085</td></tr>
 <tr><td>mlm_probability</td><td>0.5</td><td><p align="center">-</p></td><td>0.5</td><td>0.5</td></tr>
 <tr><td>embedding_dim_from_cardinality_multiplier</td><td>8</td><td><p align="center">-</p></td><td>9</td><td>7</td></tr>
</tbody></table>
