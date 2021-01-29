from functools import partial
import copy
from math import sqrt
from tqdm import tqdm
import numpy as np
import dgl
import torch
import torch.nn as nn
from model import GCN
from visualize import display_graph


# UDF for message passing. this will execute after all the previous messages, and mask the final massage.
def mask_message(edges):
    return {'m': edges.data['m'] * edges.data['edge_mask'].sigmoid().view(-1, 1)}


# workaround to hijack the "update_all" function of DGL
class ExplainGraph(dgl.DGLGraph):
    def update_all(self, message_func, reduce_func, apply_node_func=None, etype=None):
        super().apply_edges(message_func)
        super().update_all(mask_message, reduce_func, apply_node_func, etype)


class GNNExplainer:
    """
        TODO - add info
    """
    # hyper parameters, taken from the original paper
    params = {
        'edge_size': 0.3,  # 0.005,
        'feat_size': 1.0,
        'edge_ent': 1.0,
        'feat_ent': 0.1,
        'eps': 1e-15
    }

    def __init__(self, graph, model, num_hops, epochs: int = 100, lr: float = 0.01):
        self.g: dgl.DGLGraph = graph
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.num_hops = num_hops
        self.feature_mask = None
        for module in self.model.modules():
            if hasattr(module, '_allow_zero_in_degree'):
                module._allow_zero_in_degree = True

    def __set_masks__(self, g: dgl.DGLGraph):
        """ set masks for edges and features """
        num_feat = g.ndata['feat'].shape[1:]
        self.feature_mask = nn.Parameter(torch.randn(num_feat) * 0.1)

        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * g.num_nodes()))
        g.edata['edge_mask'] = nn.Parameter(torch.randn(g.num_edges()) * std)

    def __loss__(self, g, node_idx, log_logits, pred_label):
        # prediction loss
        loss = -log_logits[node_idx, pred_label[node_idx]]

        # edge loss
        me = g.edata['edge_mask'].sigmoid()
        loss = loss + torch.sum(me) * self.params['edge_size']  # edge regularization - subgraph size
        entropy = -me * torch.log(me + self.params['eps']) - (1 - me) * torch.log(1 - me + self.params['eps'])
        loss = loss + self.params['edge_ent'] * entropy.mean()  # edge los: entropy + regularization

        # node features loss
        mn = self.feature_mask.sigmoid()
        loss = loss + torch.mean(mn) * self.params['feat_size']  # node feature regularization
        entropy = -mn * torch.log(mn + self.params['eps']) - (1 - mn) * torch.log(1 - mn + self.params['eps'])
        loss = loss + self.params['feat_ent'] * entropy.mean()  # node feature los: entropy + regularization

        return loss

    @staticmethod
    def _predict(graph, model, node_id, feat_mask=None):
        model.eval()
        feat = graph.ndata['feat']
        if feat_mask is not None:
            feat = feat * feat_mask
        with torch.no_grad():
            log_logits = model(graph, feat)
            pred_label = log_logits.argmax(dim=-1)
        return log_logits[node_id], pred_label[node_id]

    def _create_subgraph(self, node_idx):
        """ get all nodes that contribute to the computation of node's embedding """
        nodes = torch.tensor([node_idx])
        eid_list = []
        for _ in range(self.num_hops):
            predecessors, _, eid = self.g.in_edges(nodes, form='all')
            eid_list.extend(eid)
            predecessors = torch.flatten(predecessors).unique()
            nodes = torch.cat([nodes, predecessors])
            nodes = torch.unique(nodes)
        eid_list = list(np.unique(np.array([eid_list])))
        # sub_g = dgl.node_subgraph(self.g, nodes)
        sub_g = dgl.edge_subgraph(self.g, eid_list)  # TODO - handle heterogeneous graphs
        return sub_g

    def explain_node(self, node_idx):
        """ main function - calculate explanation """
        # get prediction label
        self.model.eval()
        feat = self.g.ndata['feat']
        device = feat.device
        with torch.no_grad():
            log_logits = self.model(self.g, feat)
            pred_label = log_logits.argmax(dim=-1)

        # create initial subgraph (all nodes and edges that contribute to the explanation)
        subgraph = self._create_subgraph(node_idx)
        new_node_id = np.where(subgraph.ndata[dgl.NID] == node_idx)[0][0]
        pred_label = pred_label[subgraph.ndata[dgl.NID]]

        # "trick" the graph so we can hijack its calls
        original_graph_class = subgraph.__class__
        subgraph.__class__ = ExplainGraph  # super hacky, but i find it elegant in it's own way.

        # set feature and edge masks
        self.__set_masks__(subgraph)
        feat = subgraph.ndata['feat']
        # move to device
        self.feature_mask.to(device)
        subgraph.to(device)

        # start optimizing
        optimizer = torch.optim.Adam([self.feature_mask, subgraph.edata['edge_mask']], lr=self.lr)

        pbar = tqdm(total=self.epochs)
        pbar.set_description('Explaining node {}'.format(node_idx))
        # training loop
        for epoch in range(1, self.epochs + 1):
            h = feat * self.feature_mask.view(1, -1).sigmoid()  # soft mask features
            log_logits = self.model(subgraph, h)         # get prediction (will mask edges inside dgl.graph.update_all)
            loss = self.__loss__(subgraph, new_node_id, log_logits, pred_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.detach())
            pbar.update(1)
        pbar.close()

        node_feat_mask = self.feature_mask.detach().sigmoid()
        edge_mask = subgraph.edata['edge_mask'].detach().sigmoid()
        subgraph.__class__ = original_graph_class
        return subgraph, node_feat_mask, edge_mask

    def test_explanation(self, node_id, subgraph, feat_mask, edge_mask, threshold=None):
        log_logit_original, label_original = self._predict(self.g, self.model, node_id)
        num_edges_original = subgraph.num_edges()
        num_feat_original = torch.numel(feat_mask)
        new_node_id = np.where(subgraph.ndata[dgl.NID] == node_id)[0][0]

        if threshold is None:
            original_graph_class = subgraph.__class__
            subgraph.__class__ = ExplainGraph
            subgraph.edata['edge_mask'] = edge_mask.logit()  # need inverse of sigmoid to work with the soft mask
            log_logit, label = self._predict(subgraph, self.model, new_node_id, feat_mask)
            subgraph.__class__ = original_graph_class
            num_feat = num_feat_original
        else:
            subgraph.remove_edges(np.where(edge_mask < threshold)[0])
            for module in self.model.modules():
                if hasattr(module, '_allow_zero_in_degree'):
                    module._allow_zero_in_degree = True
            feat_mask_hard = feat_mask > threshold
            log_logit, label = self._predict(subgraph, self.model, new_node_id, feat_mask_hard)
            num_feat = feat_mask_hard.sum()

        print("num edges before masking: {}".format(num_edges_original))
        print("num edges after masking: {}".format(subgraph.num_edges()))
        print("num features before masking: {}".format(num_feat_original))
        print("num features after masking: {}".format(num_feat))
        print("log_logits before masking: {}".format(log_logit_original))
        print("log_logits after masking: {}".format(log_logit))
        print("label before masking: {}".format(label_original))
        print("label after masking: {}".format(label))


def f1():
    dataset = dgl.data.CoraGraphDataset()
    print('Number of categories:', dataset.num_classes)
    g = dataset[0]

    model = torch.load("model.p")
    print(model)
    num_hops = 2
    node_id = 11
    threshold = 0.5

    explainer = GNNExplainer(g, model, num_hops, epochs=200)
    subgraph, feat_mask, edge_mask = explainer.explain_node(node_id)
    explainer.test_explanation(node_id, copy.deepcopy(subgraph), feat_mask, edge_mask, threshold)
    display_graph(subgraph)
    subgraph.remove_edges(np.where(edge_mask < threshold)[0])
    display_graph(subgraph)


f1()
