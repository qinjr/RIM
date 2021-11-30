# Retrieval & Interaction Machine for Tabular Data Prediction (RIM)
A `tensorflow` implementation of our KDD 2021 paper:
[Retrieval & Interaction Machine for Tabular Data Prediction](https://arxiv.org/abs/2108.05252)
If you have any questions, please contact the author: [Jiarui Qin](http://jiaruiqin.me).

## Abstract
> Click-through rate (CTR) prediction plays a key role in modern online personalization services.
  In practice, it is necessary to capture user's drifting interests by modeling sequential user behaviors to build an accurate CTR prediction model. 
  However, as the users accumulate more and more behavioral data on the platform, it becomes non-trivial for the sequential models to make use of the whole behavior history of each user. First, directly feeding the long behavior sequence will make online inference time and system load infeasible. Second, there is much noise in such long histories to fail the sequential model learning.
  The current industrial solutions mainly truncate the sequences and just feed recent behaviors to the prediction model, which leads to a problem that sequential patterns such as periodicity or long-term dependency are not embedded in the recent several behaviors but in far back history.
  To tackle these issues, in this paper we consider it from the data perspective instead of just designing more sophisticated yet complicated models and propose User Behavior Retrieval for CTR prediction (UBR4CTR) framework. In UBR4CTR, the most relevant and appropriate user behaviors will be firstly retrieved from the entire user history sequence using a learnable search method. These retrieved behaviors are then fed into a deep model to make the final prediction instead of simply using the most recent ones. It is highly feasible to deploy UBR4CTR into industrial model pipeline with low cost. Experiments on three real-world large-scale datasets demonstrate the superiority and efficacy of our proposed framework and models.

## Citation
```
@inproceedings{qin2021retrieval,
  title={Retrieval \& Interaction Machine for Tabular Data Prediction},
  author={Qin, Jiarui and Zhang, Weinan and Su, Rong and Liu, Zhirong and Liu, Weiwen and Tang, Ruiming and He, Xiuqiang and Yu, Yong},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={1379--1389},
  year={2021}
}
```
