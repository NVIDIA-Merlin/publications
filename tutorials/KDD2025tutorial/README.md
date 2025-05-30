# ACM KDD 2025 Hands-on Tutorial: Boost the Performance of Tabular Data Models with GPU Accelerated Feature Engineering

**Conference Date**: August 3–7, 2025

**Place**: Toronto, ON, USA

**Tutorial Material**: The material will be made publicly available at the time of the our KDD'25 hands-on tutorial. <br>

**Tutorial Date**: TBD

## Abstract

Feature engineering remains a crucial technique for improving the performance of models trained on tabular data. Unlike computer vision and natural language processing, where deep learning models automatically extract hierarchical features from raw data, the most accurate tabular models, such as gradient-boosted decision trees, still benefit significantly from manually crafted features. This is demonstrated in Team NVIDIA’s many first-place data science competition victories [10] [2] [9] [3].

Fast experimentation in feature engineering is essential to quickly discover the most valuable features that improve model performance. In this tutorial, we use NVIDIA cuDF and cuML libraries [11] to accelerate the experimentation pipeline on GPU, allowing us to search for and engineer new features more rapidly and discover more accurate models faster. 

First, participants will learn specific feature engineering techniques — normalization, binning, count encoding, and target encoding [5]. These techniques will be taught using the Amazon product review dataset [6] [7] that contains product reviews from May 1996 through July 2014. Specifically, we will use the electronic category of this dataset that contains 1,689,188 reviews. Next, they will train gradient-boosted decision tree models and support vector classification models with and without feature engineering.

Participants will learn how engineered features can significantly boost the accuracy of ML models, and by the end of the tutorial, they will gain practical techniques that can be immediately applied to their real-world use cases.

## Tutorial Outline

We will start with a brief overview of the experimentation pipeline for tabular datasets and of feature engineering techniques for processing tabular datasets. We will discuss the importance of acceleration in creating data science pipelines.

We will then introduce NVIDIA cuDF and cuML libraries [11] for accelerated end-to-end data science pipelines on GPU. Afterwards, we will teach the material as hands-on labs. The audience will be able to follow all hands-on sessions in their dedicated environment via jupyter notebooks and participate by running the code themselves and solving the exercises.

The tutorial will be 180 minutes long and is designed as a combination of theoretical lectures and practical exercises for the participants. The tutorial is outlined as follows:

The tutorial is outlined as follow:
- **Presentation (30 min)**
  - Section 1 - Experimentation Pipeline for Tabular datasets
      - Why Acceleration is important?
      - Overview of Feature types
  - Section 2 - Accelerating Data Science End-to-end
      - Accelerated computing is critical for modern applications
      - Accelerating pandas with ZERO Code Change
      - Accelerating scikit-learn with ZERO Code Change
- **Hands-on Labs (120 min):**
  - Part 1 - Best practices for data preprocessing and feature engineering
      - Learn and apply Target Encoding technique
      - Learn and apply Count Encoding technique
      - Learn and apply Binning technique
      - Learn and appply Normalization technique
  - **Break (10 min)**
  - Part 2 - Train ML models on GPU
      - Train an XGBoost model on GPU
      - Train a SVC model on GPU
- **Wrap up and Q&A (20 min)**


## PREVIOUS VERSIONS OF TUTORIAL

We have presented similar content as hands-on tutorial in three different events: ACM RecSys 2020 [12], GTC’23 Digital Spring [13] and GTC 2025 conferences [14]. Each tutorial was a big success, attracting a large audience of over 150 participants at each venue, and receiving a lot of positive feedback.

- **(1) ACM RecSys 2020** - Online – After winning RecSys 2020 competition [12], we presented an online workshop demonstrating the various feature engineering techniques that our
XGBoost models utilized to win. This workshop did not include instruction on how to train models and/or incorporate features.
- **(2) GTC’23 Spring - Online** – For NVIDIA GTC’23 Digital Spring online, we adapted our previous RecSys2020 tutorial by shortening the feature engineering content and adding part 2 about training models and incorporating features.
- **(3) GTC 2025** - In Person – For NVIDIA GTC’25, we presented an updated version of our GTC’23 tutorial. Now for KDD conference 2025, we propose adding more content to feature engineering and more content to model training.

## Tutors' Bios

**Chris Deotte** is a Senior Data Scientist at NVIDIA, where he specializes in improving model performance. Chris earned his Ph.D. in Applied Mathematics with a specialization in Computational Science. Chris has competed in 90 international data science competitions and won 60+ medals. He is a quadruple Kaggle Grandmaster
[1]. Chris delivered numerous hands-on tutorials for academic and industry audiences as an NVIDIA DLI instructor.

**Ronay Ak** is a Senior Data Scientist at NVIDIA working on Information Retrieval for RAG applications. She received her Ph.D. in Energy and Power Systems Engineering discipline from CentraleSupelec in France. Ronay was part of the winning team for the Booking.com Challenge WSDM2021 [9] and the Coveo SIGIR e-Commerce Workshop Data Challenge 2021 [8]. She has authored 20+ technical publications published in internationally reputed conferences and journals and delivered numerous hands-on tutorials for academic and industry audience as an NVIDIA DLI certified instructor.

## Contributors

**Benedikt Schifferer** is a manager of an applied research team at NVIDIA working on information retrieval and LLMs. Previously, he researched on Recommender Systems and was one of the engineers building the NVIDIA Merlin framework. Prior to his work at NVIDIA, he developed recommender systems for a German ecommerce company. He holds a Master of Science in Data Science degree from Columbia University, New York. Benedikt was part of the NVIDIA AI team that won the WSDM WebTour Workshop Challenge 2021 by Booking.com, ACM RecSys 2021, KDD Cup 2023 and 2024 competitions. 


## Equipments

To perform hands-on work in the proposed time, our hands-on training platform will be running in the cloud with the support from NVIDIA’s Deep Learning Institute (DLI)[4]. Our educational platform consists of well-designed, interactive and informative Jupyter notebooks running on a GPU-optimized instance reserved to perform each hands-on task mentioned above. Therefore, participants are only expected to bring their own laptops and have an internet connection. 


## Target Audience

The target audience for this hands-on tutorial is data science, machine learning, and AI practitioners with a beginner to intermediate understanding of machine learning pipelines. Basic Python programming experience and knowledge of pandas and scikit-learn libraries is required.

## Prerequisites & Instructions

In order to be able to sign in NVIDIA DLI platform and execute the pre-prepared jupyter notebooks during hands-on tutorial session, please complete these steps prior to getting started:

- Create or log into your [NVIDIA Developer Program](https://developer.nvidia.com/login) account. This account will provide you with access to all of the training materials during the tutorial.
- Visit `websocketstest.courses.nvidia.com` and make sure all three test steps are checked “Yes.” This will test the ability for your system to access and deliver the training contents. If you encounter issues, try updating your browser. Note: Only Chrome and Firefox are supported.
Check your bandwidth. 1 Mbps downstream is required and 5 Mbps is recommended. This will ensure consistent streaming of audio/video during the tutorial to avoid glitches and delays.


## Societal Impacts

Feature engineering is a critical step in building accurate machine learning models, and its proper application can have significant societal impacts. This tutorial equips participants with the skills to enhance model performance on tabular data, which is widely used in domains such as healthcare, finance, and e-commerce. By improving predictive accuracy, these techniques can lead to better decision-making systems, such as more effective fraud detection, improved medical diagnostics, and enhanced user experiences in online platforms. 

## Acknowledgements

We would like to thank NVIDIA KGMON team and NVIDIA Deep Learning Institute (DLI)[4] team for their help and support in preparation of this tutorial.


## References

[1] Chris Deotte. 2025. Chris Deotte’s Kaggle Profile. https://www.kaggle.com/cdeotte Accessed: 2025-03-18. <br>
[2] Chris Deotte et al . 2021. GPU Accelerated Boosted Trees and Deep Neural Networks for Better Recommender Systems. In RecSysChallenge ’21: Proceedings of the Recommender Systems Challenge 2021 (Amsterdam, Netherlands) (RecSysChallenge 2021). Association for Computing Machinery, New York, NY, USA, 7–14. https://doi.org/10.1145/3487572.3487605 <br>
[3] Chris Deotte et al . 2023. Winning Amazon KDD Cup’23. Retrieved March 18, 2025 from https://openreview.net/pdf?id=J3wj55kK5t <br>
[4]NVIDIA Deep Learning Institute. 2025. Retrieved May 30, 2025 from https://www.nvidia.com/en-us/training/<br>
[5] Jiwei Liu. 2021. Target Encoding with RAPIDS cuML: Do More with Your Categorical Data, [blog post](https://medium.com/rapids-ai/target-encoding-with-rapids-cuml-
do-more-with-your-categorical-data-8c762c79e784), Accessed: 2025-03-18. <br>
[6] Julian McAuley et al . 2014. Amazon Review Dataset - Category Electronics. https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html, Accessed: 2025-03-18. <br>
[7] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (Santiago, Chile). Association for Computing Machinery, New York, NY, USA, 43–52. https://doi.org/10.1145/2766462.2767755 <br>
[8] Gabriel de Souza P. Moreira et al. 2021. Transformers with multi-modal features and post-fusion context for e-commerce session-based recommendation. https://arxiv.org/abs/2107.05124 <br>
[9] Benedikt Schifferer et al. 2021. Using Deep Learning to Win the Booking.com WSDMWebTour21 Challenge on Sequential Recommendations. https://www.bookingchallenge.com/. In Proceedings of the ACM WSDM Workshop on WebTourism (WSDM WebTour’21).
[10] Benedikt Schifferer et al. 2020. GPU Accelerated Feature Engineering and Training for Recommender Systems. In Proceedings of the Recommender Systems Challenge 2020 (Virtual Event, Brazil) (RecSysChallenge ’20). Association for Computing Machinery, New York, NY, USA, 16–23. https://doi.org/10.1145/3415959. <br>
[11] RAPIDS Development Team. 2025. RAPIDS cuDF and cuML: GPU-Accelerated Data Science Libraries. https://rapids.ai/ Accessed: 2025-03-18. <br>
[12] Chris Deotte et al. 2020. Feature Engineering for Recommender Systems. Retrieved March 18, 2025 from https://recsys.acm.org/recsys20/tutorials/ <br>
[13] Chris Deotte et al. 2023. Learn How to Create Features from Tabular Data and Accelerate your Data Science Pipeline. Retrieved March 18, 2025 from https://www.nvidia.com/en-us/on-demand/session/gtcspring23-dlit51195/ <br>
[14] Chris Deotte and Ronay Ak. 2025. Best Practices in Feature Engineering for Tabular Data With GPU Acceleration. Retrieved March 21, 2025 from [link](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=ronay%20ak#/session/1731448128537001rimn) <br>
[15] NVIDIA. 2021. Merlin Framework. Retrieved June 7, 2022 from https://developer. nvidia.com/nvidia-merlin <br>
