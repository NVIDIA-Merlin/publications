# Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation
# Online Appendices of the Paper

## Paper resources
- [Published paper](https://dl.acm.org/doi/10.1145/3460231.3474255) at *ACM Digital Library*, with the paper **PDF** accompanied by recordings of the paper presentation and a demo
- [Blog post](https://medium.com/nvidia-merlin/transformers4rec-4523cc7d8fa8) with a gentle introduction to the Transformers4Rec library


## Abstract 
Mirroring advancements in Natural Language Processing, much of the recent progress in sequential and session-based recommendation has been driven by advances in model architecture and pretraining techniques originating in the field of NLP.  Transformer architectures in particular have facilitated building higher-capacity models and provided data augmentation and other training techniques which demonstrably improve the effectiveness of sequential and session-based recommendation.  But with a thousandfold more research going on in NLP, the application of transformers for recommendation understandably lags behind.  To remedy this we introduce Transformers4Rec, an open-source library built upon HuggingFace's Transformers library with a similar goal of opening up the advances of NLP based Transformers to the recommender system community and making these advancements immediately accessible for the tasks of sequential and session-based recommendation. Like its core dependency, Transformers4Rec is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments. 

In order to demonstrate the usefulness of the library for research and also to validate the applicability of Transformer architectures to session-based recommendation where shorter sequence lengths do not match those commonly found in NLP, we have performed the first comprehensive empirical analysis comparing many Transformer architectures and training approaches for the task of session-based recommendation.  We demonstrate that the best Transformer architectures have superior performance across two e-commerce datasets while performing similarly to the baselines on two news datasets.  We further evaluate in isolation the effectiveness of the different training techniques used in causal language modeling, masked language modeling, permutation language modeling and replacement token detection for a single Transformer architecture, XLNet.  We establish that training XLNet with replacement token detection performs well across all datasets.  Finally, we explore techniques to include side information such as item and user context features in order to establish best practices and show that the inclusion of side information uniformly improves recommendation performance. 

## Appendices Organization

- [Appendix A - Techniques used in Transformers4Rec Meta-Architecture](Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)
- [Appendix B - Preprocessing and Feature Engineering](Appendices/Appendix_B-Preprocessing_and_Feature_Engineering.md)
- [Appendix C - Hypertuning - Search space and best hyperparameters](Appendices/Appendix_C-Hyperparameters.md)

## Experiments reproducibility
The experiments for the Transformers4Rec paper were performed in a former version of the Transformers4Rec library tagged as [recsys2021](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021), which can be used for paper experiments reproducibility. 

### Pre-processing
We provide scripts for preprocessing the datasets [here](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/recsys2021/datasets), i.e., for creating features and grouping interactions features by sessions. But for your convenience we also provide the [pre-processed version of the datasets](https://drive.google.com/drive/folders/1fxZozQuwd4fieoD0lmcD3mQ2Siu62ilD?usp=sharing) for download, so that you jump directly into running experiments with Transformers4Rec. 

### Training and evaluation
The command lines to run each experiment group (dataset and algorithm) reported in the paper can be found [here](experiments_reproducibility_commands.md), with the corresponding best hyperparameters reported in [Appendix C](Appendices/Appendix_C-Hyperparameters.md).

**IMPORTANT**: It is worthwhile to mention that since our experiments for the paper were finished the library was completely refactored by NVIDIA Merlin team, converting it from a research library into an flexible [open-source library](https://github.com/NVIDIA-Merlin/Transformers4Rec), packaged with PyTorch and TensorFlow APIs. We are working on preparing Python scripts to reproduce our paper results with the refactored version of Transformers4Rec and will release those scripts soon in this [NVIDIA-Merlin/publications](https://github.com/NVIDIA-Merlin/publications/tree/main/2021_acm_recsys_transformers4rec) repo, so that you can more easily understand and extend the library for your own research on sequential and session-based recommendation ;)
