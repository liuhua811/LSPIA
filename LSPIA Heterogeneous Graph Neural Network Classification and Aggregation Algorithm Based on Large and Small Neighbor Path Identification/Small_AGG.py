import torch
import torch.nn as nn
import numpy as np
from Attention import GCN
from sklearn.metrics.pairwise import cosine_similarity

def sim_process(feature_matrix,adjacency_matrix):

    # 假设 feature_matrix 是 n × d 的二维数组，n为节点数，d为特征维度
    # adjacency_matrix 是 n × n 的邻接矩阵

    # 计算特征向量间的相似度矩阵
    similarity_matrix = cosine_similarity(feature_matrix.detach())

    # 初始化新的邻接矩阵A和B
    adjacency_matrix_A = torch.zeros_like(adjacency_matrix)
    adjacency_matrix_B = adjacency_matrix.clone()

    # 对于每个节点，找到其与邻居中相似度最高的前t个邻居
    t = 5  # 设置你想要保留的相似邻居数量
    for node in range(feature_matrix.shape[0]):
        # 获取该节点的所有邻居
        neighbors = np.flatnonzero(adjacency_matrix[node])

        # 计算该节点与其邻居的相似度
        node_neighbors_similarity = similarity_matrix[node, neighbors]

        # 按照相似度从高到低排序，并选择前t个最相似的邻居
        top_t_indices = np.argsort(-node_neighbors_similarity)[:t]

        # 将这些相似邻居在新邻接矩阵A中标记为1
        adjacency_matrix_A[node, neighbors[top_t_indices]] = 1

        # 在原邻接矩阵B中移除这t个相似邻居
        adjacency_matrix_B[node, neighbors[top_t_indices]] = 0
    # 现在 adjacency_matrix_A 包含了对每个节点来说相似度最高的前t个邻居关系
    # 而 adjacency_matrix_B 包含了除去相似度最高的t个邻居后剩下的连接关系
    return adjacency_matrix_A, adjacency_matrix_B


class Small_AGG(nn.Module):
    def __init__(self,hid_dim,drop=0.5):
        super(Small_AGG, self).__init__()
        self.hid_dim = hid_dim
        self.gcn = GCN(hid_dim)
        self.dropout = nn.Dropout(p=drop)
        self.no_linear = nn.Tanh()

    def forward(self,feature,adj):
        # adj1,adj2 = sim_process(feature,adj)
        # hight_sim_feat = self.dropout(self.gcn(feature,adj1))
        # low_sim_feat = self.dropout(self.att(feature,adj2))
        # return [hight_sim_feat,low_sim_feat]

        hight_sim = self.dropout(self.gcn(feature,adj))
        hight_sim = self.no_linear(hight_sim)
        # hight_sim = self.no_linear(self.gcn(feature,adj))
        # hight_sim = self.dropout(hight_sim)
        return hight_sim