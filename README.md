# DGL-GNNExplainer
GNNExplainer implementation using Deep Graph Library (DGL)

This is a initial version of the GNNExplainer module from  "GNNExplainer: Generating Explanations for Graph Neural Networks", by Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec, 2019
https://arxiv.org/abs/1903.03894

This module supports prediction explanation using edge and feature masking. A basic visualization tool is added for convenience.

Usage examples can be found under the 'examples' folder. Current examples include GCN, GAT and GIN models operating on node classification and graph classification tasks. 

Example result using the visualization:
![Alt text](images/gcn_example.jpg?raw=true "Example")

This module is under development and so the support is limited to simple DGL graphs using pytorch, with only node features.
The tasks supported are Node-Classification and Graph-classification.
The GNNExplainer is model-agnostic and so the only requirement is that the message passing is done using DGL "update_all" primitive.

Here are some additions that are currently not supported and hopefully will be in the future:
- edge features
- link prediction tasks
- heterogeneous graphs

