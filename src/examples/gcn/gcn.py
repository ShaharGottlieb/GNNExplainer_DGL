import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import dgl.data
from train import train
from GNNExplainer import GNNExplainer
from GNNExplainer import ExplainerTags


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def main(dataset, node_id, edge_size=0.05):
    print('Number of categories:', dataset.num_classes)
    g = dataset[0]

    print('Node features')
    print(g.ndata)
    print('Edge features')
    print(g.edata)

    model_name = "gcn_model_" + dataset.name + ".p"

    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)

    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
    else:
        train(g, model, num_epochs=100)
        torch.save(model.state_dict(), model_name)
    model.eval()

    num_hops = 2
    g.ndata[ExplainerTags.NODE_FEATURES] = g.ndata['feat'].float().to(torch.device("cpu"))
    explainer = GNNExplainer(g, model, num_hops, epochs=100, edge_size=edge_size)
    subgraph, feat_mask = explainer.explain_node(node_id)
    explainer.test_explanation(node_id, subgraph, feat_mask)
    explainer.visualize(subgraph, node_id)


main(dgl.data.CoraGraphDataset(), 11, edge_size=0.15)
