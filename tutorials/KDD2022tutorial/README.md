# KDD'22 Hands-on Tutorial: Reducing the Friction for Building Recommender Systems with Merlin

**Date**: Wednesday, Aug 17 2022 (US Eastern Time Zone)

**Place**: Washington DC Convention Center, USA

**Tutorial Material**: The material will be made publicly available at the time of the our KDD'22 tutorial.

## Abstract

Recommender Systems (RecSys) are the engine of the modern internet and the catalyst for human decisions [1][2]. The goal of a recommender system is to generate relevant recommendations for users from a collection of items or services that might interest them. Building a recommendation system is challenging because it requires multiple stages (item retrieval, filtering, ranking, ordering) to work together seamlessly and efficiently during training and inference. The biggest challenges faced by new practitioners are the lack of understanding around what RecSys look like in the real world and the difficulty in transitioning from the simple Matrix Factorization (MF) [3] to more complex deep learning architectures with multiple input features, neural components and prediction heads. 
  
To address these challenges on building recommender systems, NVIDIA developed an open source framework, called [Merlin](https://github.com/NVIDIA-Merlin). Merlin consists of a set of libraries and tools to help RecSys practitioners build models and pipelines easily and more efficiently. [Merlin Models](https://github.com/NVIDIA-Merlin/models) provides modularized building blocks that can be easily connected to build classic and state-of-the-art models. It offers flexibility at each stage: multiple input processing/representation modules, different layers for designing the model’s architecture, prediction heads, loss functions, negative sampling techniques [4], among others. 
  
In this hands-on tutorial, participants will start with data preparation using [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular)- an open-source feature engineering and preprocessing library designed to quickly and easily manipulate large scale datasets. Participants will then work on modeling with Merlin Models library, building the fundamental recommendation models such as MF [3] and then transitioning to more complex deep learning-based models for candidate retrieval [5][6] and ranking [7][8]. In each iteration, we will demonstrate the seamless integration between data preparation and model training. Over the span of this tutorial, participants will learn the fundamentals of recommender systems modeling and how to build a two-stage recommender system easily using open source Merlin libraries.

## Tutorial Outline

The tutorial will cover how to iterate over performing GPU accelerated feature engineering and preprocessing using Merlin [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular), and building and training  a two-stage recommender system with [Merlin Models](https://github.com/NVIDIA-Merlin/models). Two-stage recommender system will consist of a candidate retrieval stage with MF [3] and Two-Tower models [9], and a ranking stage including deep learning-based ranking models [7], [8]. We will use a publicly available dataset to teach these concepts on a small scale, but participants can apply the same code to large scale data tasks. Participants will gain hands-on experience in this tutorial, and take this knowledge back to their organizations to build custom accelerated recommender systems models.

The tutorial is outlined as follow:
- **Presentation**
  - Introduction to Two-stage Recommender Systems
- **Hands-on Lab 1:**
  - Candidate Retrieval Models
    - GPU-accelerated ETL with NVTabular
    - Training a MF model using Merlin Models
    - Generating more features with NVTabular
    - Training a Two-Tower model using Merlin Models
- **Break (15 min)**
- **Hands-on Lab 2:**
  - Two-stage Recommender Models
    - Training ranking models using Merlin Models
  - Two-stage recommender model evaluation
- **Wrap up and Q&A**


## Presenters

**Sara Rabhi** is a Research Scientist at NVIDIA, where she works on optimizing recommender systems as well as on the development of Transformers4Rec and Merlin Models libraries. Sara received her PhD degree from the Institut Polytechnique de Paris. She has seven years of experience in data science and deep learning programming. Sara was part of the winning team of the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [13].

**Ronay Ak** is a Sr. Data Scientist at NVIDIA working on DL-based recommender systems. Before joining NVIDIA, she worked as a Research Associate at the National Institute of Standards and Technology in USA. She received her Ph.D. in Energy and Power Systems (Engineering) discipline from CentraleSupelec in Paris, FR. Ronay was part of the winning team of the WSDM2021 Booking.com challenge [12] and the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [13].


## Contributors

**Gabriel de Souza Pereira Moreira** is a Sr. Applied Research Scientist at NVIDIA Merlin team. He had his PhD degree from Instituto Tecnológico de Aeronáutica (ITA), Brazil with a focus on Deep Learning for RecSys and Session-based recommendation. Before joining NVIDIA, he was lead Data Scientist at CI&T for 5 years. He was part of the teams that won the ACM RecSys Challenge 2020, the WSDM WebTour Workshop Challenge 2021 by Booking.com [12] and the SIGIR eCommerceWorkshop Data Challenge 2021 by Coveo [13].

**Marc Romeyn** is a Sr. Machine Learning Engineer at NVIDIA focussing on developing tools for recommender systems. He is a core contributor to Merlin libraries. Prior to joining NVIDIA, he has been at Spotify building ML infrastructure for personalization. At Spotify he was one of the core contributors to the managed ML-ops platform which is used across the company. 

**Benedikt Schifferer** is a DL Engineer at NVIDIA. Before joining NVIDIA, he worked as a data scientist, developed the in-house recommendation engine at Home24. Afterwards, he worked as a data science consultant at McKinsey & Company. He got his MSc degree from Columbia University. Benedikt was part of the winning team of the ACM RecSys2020 [10], ACM RecSys2021 [11], and WSDM2021 Booking.com challenges [12].


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
 - [10] [Benedikt Schifferer et al. 2020. GPU Accelerated Feature Engineering and Training for Recommender Systems. In Proceedings of the Recommender Systems Challenge 2020 (Virtual Event, Brazil) (RecSysChallenge ’20). Association for Computing Machinery, New York, NY, USA, 16–23. https://doi.org/10.1145/3415959.
3415996
- [11] Chris Deotte et al . 2021. GPU Accelerated Boosted Trees and Deep Neural Networks for Better Recommender Systems. In RecSysChallenge ’21: Proceedings of the Recommender Systems Challenge 2021 (Amsterdam, Netherlands) (RecSysChallenge 2021). Association for Computing Machinery, New York, NY, USA, 7–14. https://doi.org/10.1145/3487572.3487605
- [12] Benedikt Schifferer et al. 2021. Using Deep Learning to Win the Booking.com
WSDMWebTour21 Challenge on Sequential Recommendations (to be published).
https://www.bookingchallenge.com/. In Proceedings of the ACM WSDM Workshop
on Web Tourism (WSDM WebTour’21).
- [13] Gabriel de Souza P. Moreira et al. 2021. Transformers with multi-modal features
and post-fusion context for e-commerce session-based recommendation. Retrieved
June 7, 2022 from https://arxiv.org/abs/2107.05124.
