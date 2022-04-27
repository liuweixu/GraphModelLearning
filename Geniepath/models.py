from turtle import forward
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv



class Breadth(nn.Module):
    def __init__(self, in_dim, out_dim, heads = 1):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=heads)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index)) #TODO 注意 edge_index的建立
        return x
    
class Depth(nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, bias=False)
    
    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)

class GeniePathLayer(nn.Module):
    def __init__(self, in_dim):
        super(GeniePathLayer, self).__init__()
        self.dim = 256
        self.lstm_hidden = 256
        self.breadth_func = Breadth(in_dim, self.dim)
        self.depth_func = Depth(self.dim, self.lstm_hidden)
    
    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)

class GeniePath(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(GeniePath, self).__init__()
        self.device = device
        self.dim = 256
        self.lstm_hidden = 256
        self.layer_num = 4
        self.lin1 = nn.Linear(in_features=in_dim, out_features=self.dim)
        self.gplayers = nn.ModuleList([GeniePathLayer(self.dim) for i in range(self.layer_num)])
        self.lin2 = nn.Linear(in_features=self.dim, out_features=out_dim)
    
    def forward(self, x, edge_index):
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], self.lstm_hidden).to(self.device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden).to(self.device)

        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
        
        x = self.lin2(x)
        return x
    