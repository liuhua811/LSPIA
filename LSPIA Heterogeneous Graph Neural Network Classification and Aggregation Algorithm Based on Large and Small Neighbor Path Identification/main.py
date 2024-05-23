import argparse
import torch
from tools import evaluate_results_nc, EarlyStopping
from load_data import load_acm, load_dblp, load_imdb, load_Yelp
import numpy as np
import random
from Model import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    ADJ, PRO,simlar, features, labels, num_classes, train_idx, val_idx, test_idx = load_Yelp()
    in_dims = [feature.shape[1] for feature in features]
    features = [feature.to(args['device']) for feature in features]
    labels = labels.to(args['device'])

    model = models(in_dims, args["hidden_units"], num_classes)
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format('ACM'))
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits, h = model(features, ADJ, PRO,simlar)
        loss = loss_fcn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        logits, h = model(features, ADJ, PRO,simlar)
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])
        print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(),
                                                                                    val_loss.item(), test_loss.item()))
        early_stopping(val_loss.data.item(), model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format('ACM')))
    model.eval()
    logits, h = model(features, ADJ, PRO,simlar)
    evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)
    Y = labels[test_idx].numpy()
    ml = TSNE(n_components=2)
    node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
    color_idx = {}
    for i in range(len(h[test_idx].detach().cpu().numpy())):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)
    for c, idx in color_idx.items():  # c是类型数，idx是索引
        if str(c) == '1':
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
        elif str(c) == '2':
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
        elif str(c) == '0':
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
        elif str(c) == '3':
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
    plt.legend()
    plt.savefig( str(args['dataset']) + "分类图.png", dpi=1000,
                bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--weight_decay', default=0.0006, help='权重衰减')
    # parser.add_argument('--dataset', default='imdb')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--feat_drop', default=0.55, help='特征丢弃率')
    parser.add_argument('--nei_num', default=2, help='邻居数量')
    parser.add_argument('--alpha', default=0.5, help='alpha')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=6, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--sample_rate', default=[5, 5], help='属性节点数量')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)
