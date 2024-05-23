import torch
import networkx as nx
import scipy
import numpy as np
import torch.nn.functional as F

prefix=r"./ACM_processed"
PAP =( scipy.sparse.load_npz(prefix + '/pspsp.npz').toarray() >0)*1
adj_matrix = F.normalize(torch.from_numpy(PAP).type(torch.FloatTensor), dim=1, p=2)

adjacency_matrix = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 1],
                             [1, 1, 0, 1],
                             [0, 1, 1, 0]])

# 构建图
G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)

# 计算PageRank
pagerank_scores = nx.pagerank(G)

# 排序节点
sorted_nodes = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)

# 选择每个节点最重要的前t个节点
t = 2
top_t_nodes = {node: pagerank_scores[node] for node in sorted_nodes[:t]}

print("Top t nodes according to PageRank:")
print(top_t_nodes)
