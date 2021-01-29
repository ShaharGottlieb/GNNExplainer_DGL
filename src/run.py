import dgl
import torch
from GNNExplainer import GNNExplainer
from model import GCN

dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)
g = dataset[0]

model = torch.load("model.p")
print(model)
num_hops = 2
node_id = 11
threshold = 0.5

explainer = GNNExplainer(g, model, num_hops, epochs=300)
subgraph, feat_mask = explainer.explain_node(node_id)
explainer.test_explanation(node_id, subgraph, feat_mask)
explainer.visualize(subgraph, node_id)