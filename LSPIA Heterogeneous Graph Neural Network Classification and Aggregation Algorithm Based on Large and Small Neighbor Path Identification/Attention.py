import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch


class Intra_order(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(Intra_order, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim)).data
        if bias:
            self.Bias = Parameter(torch.FloatTensor(hidden_dim)).data
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)
        if self.Bias is not None:
            self.Bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        output = torch.spmm(adj, torch.spmm(inputs, self.Weight))
        if self.Bias is not None:
            out = output + self.Bias
        else:
            out = output
        return out


class GCN(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(GCN, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim)).data
        if bias:
            self.Bias = Parameter(torch.FloatTensor(hidden_dim)).data
        else:
            self.register_parameter('bias', True)
        self.reset_parameters()
        self.non_linear = nn.Tanh()
    def reset_parameters(self,):
        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)
        if self.Bias is not None:
            self.Bias.data.uniform_(-stdv, stdv)
    def forward(self, inputs, adj):
        output = self.non_linear(torch.spmm(adj, self.non_linear(torch.spmm(inputs, self.Weight))))
        if self.Bias is not None:
            output = output + self.Bias
        return output