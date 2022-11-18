# Retrieval & Interaction Machine for Tabular Data Prediction (RIM)
A `tensorflow` implementation of our KDD 2021 paper:
[Retrieval & Interaction Machine for Tabular Data Prediction](https://arxiv.org/abs/2108.05252).

If you have any questions, please contact the author: [Jiarui Qin](http://jiaruiqin.me).

## Abstract
> Prediction over tabular data is an essential task in many data science applications such as recommender systems, online advertising, medical treatment, etc. Tabular data is structured into rows and columns, with each row as a data sample and each column as a feature attribute. Both the columns and rows of the tabular data carry useful patterns that could improve the model prediction performance. However, most existing models focus on the cross-column patterns yet overlook the cross-row patterns as they deal with single samples independently. In this work, we propose a general learning framework named Retrieval & Interaction Machine (RIM) that fully exploits both cross-row and cross-column patterns among tabular data. Specifically, RIM first leverages search engine techniques to efficiently retrieve useful rows of the table to assist the label prediction of the target row, then uses feature interaction networks to capture the cross-column patterns among the target row and the retrieved rows so as to make the final label prediction. We conduct extensive experiments on 11 datasets of three important tasks, i.e., CTR prediction (classification), top-n recommendation (ranking) and rating prediction (regression). Experimental results show that RIM achieves significant improvements over the state-of-the-art and various baselines, demonstrating the superiority and efficacy of RIM.

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

## Raw file of the Tmall & ML-1M & LastFM Dataset
Link: 链接:https://pan.baidu.com/s/1ZCs8n3X1iXWFglIiYQ9fGg
Password:bq7j

Other raw data can be downloaded from the links in the paper (Table 9).
