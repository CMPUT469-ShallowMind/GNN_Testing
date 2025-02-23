'''
    Based on this tutorial: https://colab.research.google.com/drive/1DIQm9rOx2mT1bZETEeVUThxcrP1RKqAn#scrollTo=ymy1pgN5oNQG
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as torchgeo_nn
import torch_geometric.utils as torchgeo_utils

import time
from datetime import datetime
from collections import OrderedDict

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, ZINC, PPI
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.dtype, "#"*10)
        return x


class TestGNN(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim):
        super(TestGNN, self).__init__()

        self.dropout = 0.25
        self.num_convs = 3
        self.edge_dim = edge_dim

        self.model = torchgeo_nn.Sequential('x, edge_index, edge_attr', [
            (PrintLayer(), "x -> x"),
            (torchgeo_nn.NNConv(input_dim, 32, self.edge_feat_net(input_dim, 32)), 'x, edge_index, edge_attr -> x'),
            PrintLayer(),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(32),
            (torchgeo_nn.NNConv(32, 64, self.edge_feat_net(32, 64)), 'x, edge_index, edge_attr -> x'),
            PrintLayer(),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(64),
            (torchgeo_nn.NNConv(64, 128, self.edge_feat_net(64, 128)), 'x, edge_index, edge_attr -> x'),
            PrintLayer(),
            
        ])

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 64), 
            nn.Dropout(self.dropout),
            nn.Linear(64, output_dim),
        )

        
    def edge_feat_net(self, input_dims, output_dims):
        return nn.Sequential(
            nn.Linear(self.edge_dim, self.edge_dim, dtype=torch.float32), 
            nn.ReLU(), 
            nn.Linear(self.edge_dim, input_dims * output_dims, dtype=torch.float32)
        )
        
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)
        x = x.type(torch.FloatTensor)
        # edge_index = edge_index.to(torch.float32)
        edge_attr = edge_attr.to(torch.float32)
        # print(x, edge_attr)
        # print(type(x), type(edge_attr))
        # split network in two, so we can get node embbedings for visualization
        emb = self.model(x, edge_index, edge_attr)
        x = self.post_mp(emb)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
def train(dataset, writer):
    test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(dataset.num_edge_features, dataset.num_node_features)

    # build model
    learner = TestGNN(max(dataset.num_node_features, 1), dataset.num_classes, max(dataset.num_edge_features, 1))
    opt = optim.Adam(learner.parameters(), lr=0.01)
    
    # train
    for epoch in range(EPOCHS):
        total_loss = 0
        learner.train()
        for batch in loader:
            #print(batch.train_mask, '----')
            # print(batch)
            opt.zero_grad()
            embedding, pred = learner(batch)
            label = batch.y

            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = learner.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, learner)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return learner

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()
    return correct / total

EPOCHS = 200

writer = SummaryWriter("./runs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

dataset = ZINC(root='/tmp/zinc', subset=True)

model = train(dataset, writer)

writer.close()

color_list = ["red", "orange", "green", "blue", "purple", "brown", "yellow"]

loader = DataLoader(dataset, batch_size=64, shuffle=True)
embs = []
colors = []
model.eval()
for batch in loader:
    # print(batch.edge_index.size())
    emb, pred = model(batch)
    embs.append(emb)
    colors += [color_list[y] for y in batch.y]
embs = torch.cat(embs, dim=0)

xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
plt.scatter(xs, ys, color=colors)
plt.savefig("TNSE Plot of Embeddings")