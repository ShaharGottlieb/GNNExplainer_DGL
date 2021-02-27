"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CiteseerGraphDataset
from gat_model import GAT
from GNNExplainer import GNNExplainer
from GNNExplainer import ExplainerTags


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(g, data, args):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = evaluate(model, g, features, labels, val_mask)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, g, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    return model


class GATArgs:
    epochs = 200
    num_heads = 8
    num_out_heads = 1
    num_layers = 1
    num_hidden = 8
    residual = False
    in_drop = 0.6
    attn_drop = 0.6
    lr = 0.005
    weight_decay = 5e-4
    negative_slope = 0.2


if __name__ == '__main__':
    data = CiteseerGraphDataset()
    g = data[0]
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    args = GATArgs()
    if not os.path.isfile("gat_model.p"):
        model = main(g, data, args)
        torch.save(model, "gat_model.p")
    else:
        model = torch.load("gat_model.p")
    model.eval()
    num_hops = args.num_layers + 1
    node_id = 136
    g.ndata[ExplainerTags.NODE_FEATURES] = g.ndata['feat'].float().to(torch.device("cpu"))
    explainer = GNNExplainer(g, model, num_hops, epochs=200, edge_size=0.015)
    subgraph, feat_mask = explainer.explain_node(node_id)
    explainer.test_explanation(node_id, subgraph, feat_mask)
    explainer.visualize(subgraph, node_id)
