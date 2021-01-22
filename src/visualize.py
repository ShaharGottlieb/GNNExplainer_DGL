import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt

dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)
g = dataset[0]

print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)

seeds = torch.tensor([53])
for _ in range(2):
    _, succ = g.out_edges(seeds)
    pred, _ = g.in_edges(seeds)
    pred = torch.flatten(pred).unique()
    seeds = torch.cat([seeds, pred])
    seeds = torch.unique(seeds)

sub_g = dgl.node_subgraph(g, seeds)
print(sub_g.ndata['label'].tolist())

nx_g = dgl.to_networkx(sub_g, node_attrs=['label']).to_undirected()
pos = nx.spring_layout(nx_g)
labels = {i: nx_g.nodes[i]['label'].item() for i in range(nx_g.number_of_nodes())}

cmap = plt.get_cmap('jet', 8)
cmap.set_under('gray')

node_labels = [nx_g.nodes[i]['label'].item() for i in range(nx_g.number_of_nodes())]
ec = nx.draw_networkx_edges(nx_g, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(nx_g, pos, node_size=25, node_color=node_labels, cmap=cmap)
plt.colorbar(nc)
plt.show()

