import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dgl.data.gindt import GINDataset
from dataloader import GraphDataLoader, collate
from gin_model import GIN

from GNNExplainer import GNNExplainer
from GNNExplainer import ExplainerTags

"""
This code was taken from DGL public examples

How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, (graphs, labels) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr').to(args.device)
        graphs = graphs.to(args.device)
        outputs = net(graphs, feat)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        feat = graphs.ndata.pop('attr').to(args.device)
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        total += len(labels)
        outputs = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc


def main(args, dataset):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    trainloader, validloader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()
    # or split_name='rand', split_ratio=0.7

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        dataset.dim_nfeats, args.hidden_dim, dataset.gclasses,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)

    criterion = nn.CrossEntropyLoss()  # default reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)

    # it's not cost-effective to handle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    best_valid_acc = 0
    best_train_acc = 0
    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, trainloader, optimizer, criterion, epoch)

        train_loss, train_acc = eval_net(
            args, model, trainloader, criterion)
        tbar.set_description(
            'train set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(train_loss, 100. * train_acc))

        valid_loss, valid_acc = eval_net(
            args, model, validloader, criterion)
        vbar.set_description(
            'valid set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(valid_loss, 100. * valid_acc))

        lrbar.set_description(
            "Learning eps with learn_eps={}: {}".format(
                args.learn_eps, [layer.eps.data.item() for layer in model.ginlayers]))
        if valid_acc > best_valid_acc and train_acc > best_train_acc:
            torch.save(model.state_dict(), "gin_model.p")
            best_valid_acc = valid_acc
            best_train_acc = train_acc

    tbar.close()
    vbar.close()
    lrbar.close()
    print(best_train_acc, best_valid_acc)
    return model


class Args:
    dataset = 'MUTAG'
    batch_size = 32
    fold_idx = 0
    num_layers = 4
    num_mlp_layers = 2
    hidden_dim = 64
    graph_pooling_type = "mean"
    neighbor_pooling_type = "sum"
    learn_eps = False
    seed = 0
    epochs = 100
    lr = 0.001
    final_dropout = 0.5
    disable_cuda = True
    device = 0


if __name__ == '__main__':
    args = Args()
    dataset = GINDataset(args.dataset, False)

    if not os.path.isfile("gin_model.p"):
        model = main(args, dataset)
    else:
        model = GIN(
            args.num_layers, args.num_mlp_layers,
            dataset.dim_nfeats, args.hidden_dim, dataset.gclasses,
            args.final_dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type)
        model.load_state_dict(torch.load("gin_model.p"))
    model.eval()
    num_hops = args.num_layers - 1
    graph_label = 178
    g = dataset.graphs[graph_label]
    g.ndata[ExplainerTags.NODE_FEATURES] = g.ndata['attr'].float().to(torch.device("cpu"))
    explainer = GNNExplainer(g, model, num_hops, epochs=100, edge_size=0.05, feat_size=0)
    subgraph, feat_mask = explainer.explain_node(None)
    explainer.test_explanation(None, subgraph, feat_mask)
    label_mapping = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    explainer.visualize(subgraph, None, label_mapping, "Mutagenic label {}".format(dataset.labels[graph_label]))
