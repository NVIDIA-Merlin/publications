# ACM RecSys'22 Hands-on Tutorial: Training and Deploying Multi-Stage Recommender Systems

**Date**: Sept 18, 2022

**Place**: Seattle, USA

**Tutorial Material**: The material will be made publicly available at the time of the our RecSys'22 tutorial.

## Abstract

Industrial recommender systems are made up of complex pipelines requiring multiple steps including feature engineering and
preprocessing, a retrieval model for candidate generation, filtering, a feature store query, a ranking model for scoring, and an ordering
stage. These pipelines need to be carefully deployed as a set, requiring coordination during their development and deployment. Data
scientists, ML engineers, and researchers might focus on different stages of recommender systems, however they share a common
desire to reduce the time and effort searching for and combining boilerplate code coming from different sources or writing custom
code from scratch to create their own RecSys pipelines.

This tutorial introduces the Merlin framework which aims to make the development and deployment of recommender systems
easier, providing methods for evaluating existing approaches, developing new ideas and deploying them to production. There are
many techniques, such as different model architectures (e.g. MF, DLRM, DCN, etc), negative sampling strategies, loss functions or
prediction tasks (binary, multi-class, multi-task) that are commonly used in these pipelines. Merlin provides building blocks that allow
RecSys practitioners to focus on the “what” question in designing their model pipeline instead of “how”. Supporting research into new
ideas within the RecSys spaces is equally important and Merlin supports the addition of custom components and the extension of
existing ones to address gaps.

In this tutorial, participants will learn: (i) how to easily implement common recommender system techniques for comparison, (ii)
how to modify components to evaluate new ideas, and (iii) deploying recommender systems, bringing new ideas to production- using
an open source framework Merlin and its libraries.

## Tutorial Outline

The tutorial will start with a short overview of challenges in training and deploying recommender systems and possible solutions. Afterwards, the instructors will teach the material as hands-on labs. First, we will teach how to apply common recommender systems architectures to a dataset with [Merlin Models](https://github.com/NVIDIA-Merlin/models), followed by customizing [Merlin Models](https://github.com/NVIDIA-Merlin/models) to evaluate new ideas. After the break, we will demonstrate how easy it is to deploy a recommender system to production with [Merlin](https://github.com/NVIDIA-Merlin/Merlin). The audience will be able to follow all hands-on sessions in their dedicated environment via jupyter notebooks and participate by running the code themself and solving the exercises.

The tutorial is outlined as follow:
- **Presentation**
  - Section 1 - Introduction to training and deploying recommender systems
- **Hands-on Labs:**
  - Section 2 - Implementing popular RecSys architectures and algorithms with Merlin Models
  - Section 3 - How to customize and extend Merlin Models for research
  - **Break (10 min)**
  - Section 4 - Training a retrieval and ranking model
  - Section 5 - Deployment of Multi-Stage Recommender Systems
- **Wrap up and Q&A**


## Presenters

**Sara Rabhi** is a Research Scientist at NVIDIA, where she works on optimizing recommender systems as well as on the development of Transformers4Rec and Merlin Models libraries. Sara received her PhD degree from the Institut Polytechnique de Paris. She has seven years of experience in data science and deep learning programming. Sara was part of the winning team of the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [13].

**Ronay Ak** is a Sr. Data Scientist at NVIDIA working on DL-based recommender systems. Before joining NVIDIA, she worked as a Research Associate at the National Institute of Standards and Technology in USA. She received her Ph.D. in Energy and Power Systems (Engineering) discipline from CentraleSupelec in Paris, FR. Ronay was part of the winning team of the WSDM2021 Booking.com challenge [12] and the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [13].

**Benedikt Schifferer** is a DL Engineer at NVIDIA. Before joining NVIDIA, he worked as a data scientist, developed the in-house recommendation engine at Home24. Afterwards, he worked as a data science consultant at McKinsey & Company. He got his MSc degree from Columbia University. Benedikt was part of the winning team of the ACM RecSys2020 [10], ACM RecSys2021 [11], and WSDM2021 Booking.com challenges [12].


**Gabriel de Souza Pereira Moreira** is a Sr. Applied Research Scientist at NVIDIA Merlin team. He had his PhD degree from Instituto Tecnológico de Aeronáutica (ITA), Brazil with a focus on Deep Learning for RecSys and Session-based recommendation. Before joining NVIDIA, he was lead Data Scientist at CI&T for 5 years. He was part of the teams that won the ACM RecSys Challenge 2020, the WSDM WebTour Workshop Challenge 2021 by Booking.com [12] and the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [13].

## Contributors

**Marc Romeyn** is a Sr. Machine Learning Engineer at NVIDIA focussing on developing tools for recommender systems. He is a core contributor to Merlin libraries. Prior to joining NVIDIA, he has been at Spotify building ML infrastructure for personalization. At Spotify he was one of the core contributors to the managed ML-ops platform which is used across the company. 

## Equipments

To perform hands-on work in the proposed time, our hands-on training platform will be running in the cloud with the support from NVIDIA’s Deep Learning Institute (DLI). Our educational platform consists of well-designed, interactive and informative Jupyter notebooks running on a GPU-optimized instance reserved to perform each hands-on task mentioned above. Therefore, participants are only expected to bring their own laptops and have an internet connection.

## Prerequisites & Instructions

In order to be able to sign in NVIDIA DLI platform and execute the pre-prepared jupyter notebooks during hands-on tutorial session, please complete these steps prior to getting started:

- Create or log into your [NVIDIA Developer Program](https://developer.nvidia.com/login) account. This account will provide you with access to all of the training materials during the tutorial.
- Visit `websocketstest.courses.nvidia.com` and make sure all three test steps are checked “Yes.” This will test the ability for your system to access and deliver the training contents. If you encounter issues, try updating your browser. Note: Only Chrome and Firefox are supported.
Check your bandwidth. 1 Mbps downstream is required and 5 Mbps is recommended. This will ensure consistent streaming of audio/video during the tutorial to avoid glitches and delays.

## Acknowledgements

We would like to thank NVIDIA Merlin team and DLI team for their help and support in preparation of this tutorial

## References

- [1] Justin Basilico. 2021. Trends in Recommendation and Personalization at Netflix.
Retrieved June 7, 2022 from https://scale.com/blog/Netflix-Recommendation-
Personalization-TransformX-Scale-AI-Insights
- [2] Even Oldridge and Karl Byleen-Higley. 2022. Recommender Systems, Not Just
Recommender Models. Retrieved June 7, 2022 from https://medium.com/nvidia-
merlin/recommender-systems-not-just-recommender-models-485c161c755e
- [3] Yehuda Koren et al . 2009. Matrix factorization techniques for recommender
systems. Computer 42, 8 (2009), 30–37.
- [4] Steffen Rendle et al. 2009. BPR: Bayesian Personalized Ranking from Implicit
Feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial
Intelligence (Montreal, Quebec, Canada) (UAI ’09). AUAI Press, Arlington, Virginia,
USA, 452–461.
- [5] Paul Covington et al . 2016. Deep Neural Networks for YouTube Recommendations.
In Proceedings of the 10th ACM Conference on Recommender Systems (Boston,
Massachusetts, USA) (RecSys ’16). Association for Computing Machinery, New
York, NY, USA, 191–198. https://doi.org/10.1145/2959100.2959190
- [6] Jiaqi Ma et al . 2020. Off-Policy Learning in Two-Stage Recommender Systems.
Association for Computing Machinery, New York, NY, USA, 463–473. https:
//doi.org/10.1145/3366423.3380130
- [7] Naumov Maxim et al . 2019. Deep Learning Recommendation Model for Per-
sonalization and Recommendation Systems. Retrieved March 23, 2022 from
https://arxiv.org/abs/1906.00091
- [8] Ruoxi Wang et al . 2021. DCN V2: Improved Deep & Cross Network and Practical
Lessons for Web-Scale Learning to Rank Systems. In Proceedings of the Web
Conference 2021 (Ljubljana, Slovenia) (WWW ’21). Association for Computing
Machinery, New York, NY, USA, 1785–1797. https://doi.org/10.1145/3442381.
3450078
- [9] Xinyang Yi et al . 2019. Sampling-Bias-Corrected Neural Modeling for Large
Corpus Item Recommendations. In Proceedings of the 13th ACM Conference on
Recommender Systems (Copenhagen, Denmark) (RecSys ’19). Association for
Computing Machinery, New York, NY, USA, 269–277. https://doi.org/10.1145/
3298689.3346996
 - [10] [Benedikt Schifferer et al. 2020. GPU Accelerated Feature Engineering and Train-
ing for Recommender Systems. In Proceedings of the Recommender Systems Chal-
lenge 2020 (Virtual Event, Brazil) (RecSysChallenge ’20). Association for Com-
puting Machinery, New York, NY, USA, 16–23. https://doi.org/10.1145/3415959.
3415996
- [11] Chris Deotte et al . 2021. GPU Accelerated Boosted Trees and Deep Neural Net-
works for Better Recommender Systems. In RecSysChallenge ’21: Proceedings of
the Recommender Systems Challenge 2021 (Amsterdam, Netherlands) (RecSysChal-
lenge 2021). Association for Computing Machinery, New York, NY, USA, 7–14.
https://doi.org/10.1145/3487572.3487605
- [12] Benedikt Schifferer et al. 2021. Using Deep Learning to Win the Booking.com
WSDMWebTour21 Challenge on Sequential Recommendations (to be published).
https://www.bookingchallenge.com/. In Proceedings of the ACM WSDM Workshop
on Web Tourism (WSDM WebTour’21).
- [13] Gabriel de Souza P. Moreira et al. 2021. Transformers with multi-modal features
and post-fusion context for e-commerce session-based recommendation. Retrieved
June 7, 2022 from https://arxiv.org/abs/2107.05124.