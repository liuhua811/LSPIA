import torch
import torch.nn as nn
import numpy as np
import copy
from PageRank import PageRank
from Small_AGG import Small_AGG

class AggAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(AggAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class models(nn.Module):
    def __init__(self,input_dim,hid_dim,out_dim,drop=0.5):
        super(models, self).__init__()
        self.input_dim = input_dim
        self.no_linear = nn.Tanh()
        self.fc_list = nn.ModuleList([
            nn.Linear(feat_dim, hid_dim) for feat_dim in input_dim
        ])
        self.page_rank=PageRank(hid_dim)
        self.small_agg = Small_AGG(hid_dim)
        self.semantic_attention = AggAttention(in_size=hid_dim)
        self.dropout = nn.Dropout(p=drop)
        self.project = nn.Linear(hid_dim,out_dim)
        self.project1 = nn.Linear(hid_dim,hid_dim)
    def degree(self,adj):
        adj_degree = []
        for i in range(len(adj)):
            adj_degree.append(adj[i].sum(dim=0))    #得到每个邻接矩阵的度向量
            adj_degree[i] = adj_degree[i].sum(dim=0)    #对每个度向量求和，得到每个邻接矩阵的总度值
        return adj_degree
    def degree_analyze(self,degree):
        min_value = np.min(degree)
        relative_diffs = [(value - min_value) / min_value * 100 for value in degree]
        return relative_diffs

    def forward(self,features,adj,PRO,simlar):
        h_all = [self.no_linear(self.fc_list[i](features[i])) for i in range(len(features))]
        adj_degree = self.degree(adj)   #度统计
        #对总度值进行分析，选择应用策略--->比较不同值之间的差异情况
        relative_diffs = self.degree_analyze(adj_degree)
        big_degree_adj = []
        small_degree_adj = copy.deepcopy(adj)
        j=0
        for i in range(len(relative_diffs)):
            if relative_diffs[i] >= 100:
                big_degree_adj.append(adj[i])
                small_degree_adj.pop(i-j)
                j+=1
        #邻接矩阵分类完毕  对大度矩阵与小度矩阵分别运算
        feat = []
        big_degree_feat = [self.dropout(self.page_rank(h_all[0],PRO[i],simlar)) for i in range(len(PRO))]
        small_degree_feat = [self.small_agg(h_all[0],small_degree_adj[i]) for i in range(len(small_degree_adj))]
        feat.extend(big_degree_feat)
        feat.extend(small_degree_feat)
        h = self.semantic_attention(torch.stack(feat, dim=1))
        # h = self.no_linear(self.project1(h))
        return self.project(h),h







