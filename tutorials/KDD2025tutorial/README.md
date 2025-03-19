# ACM KDD 2025 Hands-on Tutorial: Boost the Performance of Tabular Data Models with Powerful Feature Engineering

**Date**: August 3–7, 2025

**Place**: Toronto, ON, USA

**Tutorial Material**: The material will be made publicly available at the time of the our KDD'25 hands-on tutorial.


## Abstract

Feature engineering is a powerful technique to improve the performance of models trained from tabular data. Unlike computer vision and natural language where deep learning models create features for us, the most accurate tabular data models still utilize the process of human created new columns from existing columns [9] [2] [4]. Transforming features or groups of features into new representations help the model detect and utilize patterns to predict targets. These techniques have helped Team NVIDIA win 1st place in many prestigious international data science competitions [9] [2] [3]. 

In this hands-on tutorial, we will teach specific feature engineering techniques using real world data. Together instructor and participants will train gradient boosted decision trees and support vector machines on real Amazon review data [6 ]. NVIDIA cuDF and cuML will be used to accelerate the experimentation pipline allowing us to find and improve model accuracy quicker [8]. We will learn four feature engineering techniques and observe their application improve model accuracy on a classification task. Using normalization, binning, count encoding, and target encoding we will observe a 25% or more improvement in cross validation AUC score when predicting the probability that a user likes a product [5].

## Tutorial Outline

The tutorial will start with a short overview experimentation pipeline for tabular datasets and of feature engineering techniques for processing tabular datasets. Instructors will introduce NVIDIA Cuda-X libraries (cuDF and cuML) for accelerated end-to-end data science pipelines. Afterwards, the instructors will teach the material as hands-on labs. The audience will be able to follow all hands-on sessions in their dedicated environment via jupyter notebooks and participate by running the code themself and solving the exercises.


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

We have presented similar content as hands-on tutorial in three different events: ACM RecSys 2020 [7], GTC’23 Digital Spring [8] and GTC 2025 conferences [9]. Each tutorial was a big success, attracting a large audience of over 150 participants at each venue, and receiving a lot of positive feedback.


## Tutors

**Chris Deotte** is a Senior Data Scientist at NVIDIA, where he specializes in improving model performance. Chris earned his Ph.D. in Applied Mathematics with specialization in Computational Science. He has competed in 90 international data science competitions and won over 60 medals. Chris is a quadruple Kaggle Grandmaster [1].

**Ronay Ak** is a Senior Data Scientist at NVIDIA working on Information Retrieval for RAG applications. Prior to her current role, she was focusing on deep learning-based recommender systems and was one of the engineers building the NVIDIA Merlin framework. She received her PhD in Energy & Power Systems Engineering discipline from CentraleSupelec in France. She was part of the NVIDIA AI team that won the WSDM WebTour Workshop Challenge 2021 by Booking.com [4] and SIGIR’21 Ecommerce data challenge hosted by Coveo [5]. She has authored 20+ technical publications published in internationally reputed conferences and journals, and delivered numerous hands-on tutorials for academic and industry audience as an NVIDIA DLI certified instructor.


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
By fostering an understanding of both the technical and ethical aspects of feature engineering, this tutorial empowers participants to build models that are not only accurate but also socially responsible.


## Acknowledgements

We would like to thank NVIDIA KGMON team and DLI team for their help and support in preparation of this tutorial


## References

- [1] Chris Deotte. 2025. Chris Deotte’s Kaggle Profile. https://www.kaggle.com/cdeotte Accessed: 2025-03-18.
- [2] [Benedikt Schifferer et al. 2020. GPU Accelerated Feature Engineering and Training for Recommender Systems. In Proceedings of the Recommender Systems Challenge 2020 (Virtual Event, Brazil) (RecSysChallenge ’20). Association for Computing Machinery, New York, NY, USA, 16–23. https://doi.org/10.1145/3415959.3415996
- [3] Chris Deotte et al . 2021. GPU Accelerated Boosted Trees and Deep Neural Networks for Better Recommender Systems. In RecSysChallenge ’21: Proceedings of the Recommender Systems Challenge 2021 (Amsterdam, Netherlands) (RecSysChallenge 2021). Association for Computing Machinery, New York, NY, USA, 7–14. https://doi.org/10.1145/3487572.3487605
- [4] Benedikt Schifferer et al. 2021. Using Deep Learning to Win the Booking.com WSDMWebTour21 Challenge on Sequential Recommendations. https://www.bookingchallenge.com/. In Proceedings of the ACM WSDM Workshop on Web Tourism (WSDM WebTour’21).
- [5] Gabriel de Souza P. Moreira et al. 2021. Transformers with multi-modal features and post-fusion context for e-commerce session-based recommendation. Retrieved June 7, 2022 from https://arxiv.org/abs/2107.05124
- [6] [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [7] Chris Deotte and Ronay Ak. 2025. Best Practices in Feature Engineering for Tabular Data With GPU Acceleration. Retrieved March 18, 2025 from https://www.nvidia.com/gtc/session-catalog/?regcode=pa-srch-goog-
157409-prsp&ncid=pa-srch-goog-157409-prsp&tab.catalogallsessionstab=
16566177511100015Kus&search=ronay%20ak#/session/1731448128537001rimn
