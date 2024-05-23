import torch
import torch.nn as nn
import numpy as np
from Attention import Intra_order,GCN


class PageRank(nn.Module):
    def __init__(self,hid_dim):
        super(PageRank, self).__init__()
        self.hid_dim = hid_dim
        self.gcn = GCN(hid_dim)
        self.project = nn.Linear(hid_dim,hid_dim)
        self.project2 = nn.Linear(hid_dim,hid_dim)
        self.non_linear = nn.Tanh()
        self.att = Intra_order(hid_dim)


    # def forward(self,h,adj,simlar):
    #     #similarity_matrix = adj * simlar
    #     similarity_matrix = self.non_linear(adj * simlar)
    #     top_t_similarities = torch.zeros_like(similarity_matrix)
    #     t=500
    #     for i in range(similarity_matrix.size(0)):
    #         # 对于每一行，找到相似性最高的前t个节点的值和索引
    #         top_values, top_indices = torch.topk(similarity_matrix[i], t)
    #
    #         # 将这些最高值填充到结果矩阵的对应位置
    #         top_t_similarities[i, top_indices] = top_values
    #     top_t_similarities = (top_t_similarities>0).float()
    #     h =self.non_linear(self.project(h))
    #     # h =self.non_linear(self.project2(h))
    #     feat = self.att(h,top_t_similarities)
    #     return feat

    def forward(self, h, adj, simlar):
        # Step 1: 计算相似性矩阵，考虑邻接矩阵中的连接
        similarity_matrix = self.non_linear(adj * simlar)
        # Step 2: 仅考虑adj矩阵中存在的连接
        filtered_similarity_matrix = similarity_matrix * (adj > 0).float()
        # 初始化新的连接矩阵
        new_connection_matrix = torch.zeros_like(filtered_similarity_matrix)
        t = 700 # 设定t值，根据需要调整
        for i in range(filtered_similarity_matrix.size(0)):
            # Step 3: 对于每一行，找到相似性最高的前t个节点的值和索引
            top_values, top_indices = torch.topk(filtered_similarity_matrix[i],
                                                 min(t, filtered_similarity_matrix[i].nonzero().size(0)))
            # Step 4: 将这些最高值填充到新的连接矩阵的对应位置
            new_connection_matrix[i, top_indices] = top_values
        # 这里可以继续后续处理，例如利用new_connection_matrix更新节点特征
        # 示例：使用新的连接矩阵进行特征聚合
        h = self.non_linear(self.project(h))
        feat = self.gcn(h, new_connection_matrix)
        return feat
