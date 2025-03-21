# ACM KDD 2025 Hands-on Tutorial: Boost the Performance of Tabular Data Models with Powerful Feature Engineering

**Conference Date**: August 3–7, 2025

**Place**: Toronto, ON, USA

**Tutorial Material**: The material will be made publicly available at the time of the our KDD'25 hands-on tutorial. <br>

**Tutorial Date**: TBD

## Abstract

Feature engineering remains a crucial technique for improving the performance of models trained on tabular data. Unlike computer vision and natural language processing, where deep learning models automatically extract hierarchical features from raw data, the most accurate tabular models—such as gradient-boosted decision trees—still benefit significantly from manually crafted features. This is demonstrated in Team NVIDIA’s many 1st place data science competition victories [13] [3] [5] [14].

This problem solving hands on tutorial will be presented in two parts. The first part will be dedicated to feature engineering. We will teach specific feature engineering techniques using the Amazon product review dataset [8] [9] which contains product reviews from May 1996 thru July 2014. Specifically, we will use the electronic category of this dataset containing 1,689,188 reviews. Using features generated in the first part, participants will learn how to train a gradient boosted decision trees (XGBoost) and support vector machines (SVM) model in the second part. NVIDIA cuDF and cuML libraries will be used to accelerate the experimentation pipeline allowing us to search for and engineer new features much faster and discover more accurate models quicker [12].

We will cover four feature engineering techniques— normalization, binning, count encoding, and target encoding [7] —and demonstrate their impact on classification accuracy. Applying these methods, we will observe a significant boost in cross-validation AUC when predicting user preferences. By the end of the tutorial, participants will gain practical techniques they can immediately apply to their real-world use cases.

## Tutorial Outline

This tutorial is designed as a problem-solving tutorial. We will start with a short overview of experimentation pipeline for tabular datasets and of feature engineering techniques for processing tabular datasets. We will discuss the importance of acceleration in creating data science pipelines. We will then introduce NVIDIA cuDF and cuML libraries [12] for accelerated end-to-end data science pipelines. Afterwards, we will teach the material as hands-on labs. The audience will be able to follow all hands-on sessions in their dedicated environment via jupyter notebooks and participate by running the code themselves and solving the exercises. The tutorial will be 180 min long and is designed as a combination of theoretical lectures and practical exercises for the participants.

The tutorial is outlined as follow:
- **Presentation**
  - Section 1 - Experimentation Pipeline for Tabular datasets
      - Why Acceleration is important?
      - Overview of Feature types
  - Section 2 - Accelerating Data Science End-to-end
      - Accelerated computing is critical for modern applications
      - Accelerating pandas with ZERO Code Change
      - Accelerating scikit-learn with ZERO Code Change
- **Hands-on Labs:**
  - Part 1 - Best practices for data preprocessing and feature engineering
      - Learn and apply Target Encoding technique
      - Learn and apply Count Encoding technique
      - Learn and apply Binning technique
      - Learn and appply Normalization technique
  - **Break (10 min)**
  - Part 2 - Train ML models on GPU
      - Train an XGBoost model on GPU
      - Train a SVC model on GPU
- **Wrap up and Q&A**


## PREVIOUS VERSIONS OF TUTORIAL

We have presented similar content as hands-on tutorial in three different events: ACM RecSys 2020 [ 2], GTC’23 Digital Spring [4] and GTC 2025 conferences [6]. Each tutorial was a big success, attracting a large audience of over 150 participants at each venue, and receiving a lot of positive feedback.

- **(1) ACM RecSys 2020** - Online – After winning RecSys 2020 competition [ 13 ], we presented an online workshop demonstrating the various feature engineering techniques that our
XGBoost models utilized to win. This workshop did not include instruction on how to train models and/or incorporate features.
- **(2) GTC’23 Spring - Online** – For NVIDIA GTC’23 Digital Spring online, we adapted our previous RecSys2020 tutorial by shortening the feature engineering content and adding part 2 about training models and incorporating features.
- **(3) GTC 2025** - In Person – For NVIDIA GTC’25, we presented an updated version of our GTC’23 tutorial. Now for KDD conference 2025, we propose adding more content to feature engineering and more content to model training.

## Tutors

**Chris Deotte** is a Senior Data Scientist at NVIDIA, where he specializes in improving model performance. Chris earned his Ph.D. in Applied Mathematics with specialization in Computational Science. He has competed in 90 international data science competitions and won over 60 medals. Chris is a quadruple Kaggle Grandmaster [1].

**Ronay Ak** is a Senior Data Scientist at NVIDIA working on Information Retrieval for RAG applications. Prior to her current role, she was focusing on deep learning-based recommender systems and was one of the engineers building the NVIDIA Merlin framework [11]. She received her PhD in Energy and Power Systems Engineering discipline from CentraleSupelec in France. Ronay was part of the winning team of the WSDM2021 Booking.com challenge [14] and the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [10]. She has authored 20+ technical publications published in internationally reputed conferences and journals, and delivered numerous hands-on tutorials for academic and industry audience as an NVIDIA DLI certified instructor.

## Contributors

**Benedikt Schifferer** is a manager of an applied research team at NVIDIA working on information retrieval and LLMs. Previously, he researched on Recommender Systems and was one of the engineers building the NVIDIA Merlin framework. Prior to his work at NVIDIA, he developed recommender systems for a German ecommerce company. He holds a Master of Science in Data Science degree from Columbia University, New York. Benedikt was part of the NVIDIA AI team that won the WSDM WebTour Workshop Challenge 2021 by Booking.com, ACM RecSys 2021, KDD Cup 2023 and 2024 competitions. 


## Equipments

To perform hands-on work in the proposed time, our hands-on training platform will be running in the cloud with the support from NVIDIA’s Deep Learning Institute (DLI)[6]. Our educational platform consists of well-designed, interactive and informative Jupyter notebooks running on a GPU-optimized instance reserved to perform each hands-on task mentioned above. Therefore, participants are only expected to bring their own laptops and have an internet connection. 


## Target Audience

The target audience for this hands-on tutorial is the data science, machine learning and/or AI community with a beginner to intermediate understanding of machine learning and deep learning model pipelines. Basic Python programming experience and knowledge of pandas and scikit-learn libraries is required

## Prerequisites & Instructions

In order to be able to sign in NVIDIA DLI platform and execute the pre-prepared jupyter notebooks during hands-on tutorial session, please complete these steps prior to getting started:

- Create or log into your [NVIDIA Developer Program](https://developer.nvidia.com/login) account. This account will provide you with access to all of the training materials during the tutorial.
- Visit `websocketstest.courses.nvidia.com` and make sure all three test steps are checked “Yes.” This will test the ability for your system to access and deliver the training contents. If you encounter issues, try updating your browser. Note: Only Chrome and Firefox are supported.
Check your bandwidth. 1 Mbps downstream is required and 5 Mbps is recommended. This will ensure consistent streaming of audio/video during the tutorial to avoid glitches and delays.


## Societal Impacts

Feature engineering is a critical step in building accurate machine learning models, and its proper application can have significant societal impacts. This tutorial equips participants with the skills to enhance model performance on tabular data, which is widely used in domains such as healthcare, finance, and e-commerce. By improving predictive accuracy, these techniques can lead to better decision-making systems, such as more effective fraud detection, improved medical diagnostics, and enhanced user experiences in online platforms. 
However, the use of feature engineering also presents challenges. Poorly designed features or biased data transformations can inadvertently reinforce societal biases, leading to unfair model predictions. For example, encoding techniques such as target encoding could propagate biases present in historical data, affecting marginalized groups. Therefore, practitioners must apply feature engineering with an awareness of ethical considerations, ensuring that models remain fair and transparent.


## Acknowledgements

We would like to thank NVIDIA KGMON team and DLI team for their help and support in preparation of this tutorial


## References

[1] Chris Deotte. 2025. Chris Deotte’s Kaggle Profile. https://www.kaggle.com/cdeotte Accessed: 2025-03-18. <br>
[2] Chris Deotte et al. 2020. Feature Engineering for Recommender Systems. Retrieved March 18, 2025 from https://recsys.acm.org/recsys20/tutorials/ <br>
[3] Chris Deotte et al . 2021. GPU Accelerated Boosted Trees and Deep Neural Networks for Better Recommender Systems. In RecSysChallenge ’21: Proceedings of the Recommender Systems Challenge 2021 (Amsterdam, Netherlands) (RecSysChallenge 2021). Association for Computing Machinery, New York, NY, USA, 7–14. https://doi.org/10.1145/3487572.3487605 <br>
[4] Chris Deotte et al. 2023. Learn How to Create Features from Tabular Data and Accelerate your Data Science Pipeline. Retrieved March 18, 2025 from https://www.nvidia.com/en-us/on-demand/session/gtcspring23-dlit51195/ <br>
[5] Chris Deotte et al . 2023. Winning Amazon KDD Cup’23. Retrieved March 18, 2025 from https://openreview.net/pdf?id=J3wj55kK5t <br>
[6] Chris Deotte and Ronay Ak. 2025. Best Practices in Feature Engineering for Tabular Data With GPU Acceleration. Retrieved March 21, 2025 from [link](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=ronay%20ak#/session/1731448128537001rimn) <br>
[7] Jiwei Liu. 2021. Target Encoding with RAPIDS cuML: Do More with Your Categorical Data, [blog post](https://medium.com/rapids-ai/target-encoding-with-rapids-cuml-
do-more-with-your-categorical-data-8c762c79e784), Accessed: 2025-03-18. <br>
[8] Julian McAuley et al . 2014. Amazon Review Dataset - Category Electronics. https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html, Accessed: 2025-03-18. <br>
[9] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (Santiago, Chile). Association for Computing Machinery, New York, NY, USA, 43–52. https://doi.org/10.1145/2766462.2767755 <br>
[10] Gabriel de Souza P. Moreira et al. 2021. Transformers with multi-modal features and post-fusion context for e-commerce session-based recommendation. https://arxiv.org/abs/2107.05124 <br>
[11] NVIDIA. 2021. Merlin Framework. Retrieved June 7, 2022 from https://developer. nvidia.com/nvidia-merlin <br>
[12] RAPIDS Development Team. 2025. RAPIDS cuDF and cuML: GPU-Accelerated Data Science Libraries. https://rapids.ai/ Accessed: 2025-03-18. <br>
[13] Benedikt Schifferer et al. 2020. GPU Accelerated Feature Engineering and Training for Recommender Systems. In Proceedings of the Recommender Systems Challenge 2020 (Virtual Event, Brazil) (RecSysChallenge ’20). Association for Computing Machinery, New York, NY, USA, 16–23. https://doi.org/10.1145/3415959. <br>
[14] Benedikt Schifferer et al. 2021. Using Deep Learning to Win the Booking.com WSDMWebTour21 Challenge on Sequential Recommendations. https://www.bookingchallenge.com/. In Proceedings of the ACM WSDM Workshop on WebTourism (WSDM WebTour’21).
