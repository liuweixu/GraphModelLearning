import os.path as osp

import torch
import torch.nn as nn

from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score

from models import GeniePath as Net

dataset_pubmed = Planetoid(root='./datasets/pubmed', name='Pubmed')
device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')

data_pubmed = dataset_pubmed[0].to(device)

train_mask = data_pubmed.train_mask
val_mask = data_pubmed.val_mask
test_mask = data_pubmed.test_mask

train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

model = Net(data_pubmed.num_features, dataset_pubmed.num_classes, device).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()


    out = model(data_pubmed.x, data_pubmed.edge_index)
    total_loss = loss_fn(out[train_idx], data_pubmed.y[train_idx])
    total_acc = accuracy_score(data_pubmed.y[train_idx].cpu(), out[train_idx].argmax(dim=1).cpu())
    valid_loss = loss_fn(out[val_idx], data_pubmed.y[val_idx])
    valid_acc = accuracy_score(data_pubmed.y[val_idx].cpu(), out[val_idx].argmax(dim=1).cpu())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_acc, valid_acc


if __name__ == "__main__":
    for epochs in range(300):
        total_acc, valid_acc = train()
        if (epochs + 1) % 10 == 0:
            print('{:.2f}====={:.2f}'.format(total_acc, valid_acc))

