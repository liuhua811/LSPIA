import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp
from sklearn.manifold import TSNE


def load_acm(prefix=r".\ACM_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_p.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features = [features_0]
    similarity_matrix = np.load(prefix+"/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)

    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    PAP = scipy.sparse.load_npz(prefix + '/PAP.npz').toarray()
    norm_PAP = F.normalize(torch.from_numpy(PAP).type(torch.FloatTensor), dim=1, p=2)
    PSP = scipy.sparse.load_npz(prefix + '/PSP.npz').toarray()
    norm_PSP = F.normalize(torch.from_numpy(PSP).type(torch.FloatTensor), dim=1, p=2)
    PSPSP = scipy.sparse.load_npz(prefix + '/pspsp.npz').toarray()
    norm_PSPSP = F.normalize(torch.from_numpy(PSPSP).type(torch.FloatTensor), dim=1, p=2)
    PAPAP = scipy.sparse.load_npz(prefix + '/papap.npz').toarray()
    norm_PAPAP = F.normalize(torch.from_numpy(PAPAP).type(torch.FloatTensor), dim=1, p=2)
    # PSPAP = scipy.sparse.load_npz(prefix + '/pspap.npz').toarray()
    # norm_PSPAP = F.normalize(torch.from_numpy(PSPAP).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_PAP,norm_PAPAP,norm_PSPSP]

    pro_PSP = (scipy.sparse.load_npz(prefix + '/pro-psp.npz').toarray())
    pro_PSP = F.normalize(torch.from_numpy(pro_PSP).type(torch.FloatTensor), dim=1, p=2)
    pro_PAPAP = (scipy.sparse.load_npz(prefix + '/pro-papap.npz').toarray())
    pro_PAPAP = F.normalize(torch.from_numpy(pro_PAPAP).type(torch.FloatTensor), dim=1, p=2)
    pro_PSPSP = (scipy.sparse.load_npz(prefix + '/pro-pspsp.npz').toarray() )
    pro_PSPSP = F.normalize(torch.from_numpy(pro_PSPSP).type(torch.FloatTensor), dim=1, p=2)
    pro_PAP = (scipy.sparse.load_npz(prefix + '/pro-pap.npz').toarray() )
    pro_PAP = F.normalize(torch.from_numpy(pro_PAP).type(torch.FloatTensor), dim=1, p=2)
    PRO = [pro_PAPAP,pro_PSPSP]

    return ADJ, PRO,similarity_matrix, features, labels, num_classes, train_idx, val_idx, test_idx


def load_dblp(prefix=r".\DBLP_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_A.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features = [features_0]  # 因为只有p是属性，所以只传P的特征即可
    similarity_matrix = np.load(prefix + "/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)
    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 4

    APA = scipy.sparse.load_npz(prefix + '/apa.npz').toarray()
    APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').toarray()
    APCPA = scipy.sparse.load_npz(prefix + '/apcpa.npz').toarray()
    APAPA = np.load(prefix + '/apapa.npy')

    norm_APA = F.normalize(torch.from_numpy(APA).type(torch.FloatTensor), dim=1, p=2)
    norm_APTPA = F.normalize(torch.from_numpy(APTPA).type(torch.FloatTensor), dim=1, p=2)
    norm_APCPA = F.normalize(torch.from_numpy(APCPA).type(torch.FloatTensor), dim=1, p=2)
    norm_APAPA = F.normalize(torch.from_numpy(APAPA).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_APA, norm_APCPA, norm_APTPA, norm_APAPA]


    pro_APCPA = (scipy.sparse.load_npz(prefix + '/pro-apcpa.npz').toarray() > 0) * 1
    pro_APCPA = F.normalize(torch.from_numpy(pro_APCPA).type(torch.FloatTensor), dim=1, p=2)
    pro_APTPA = (scipy.sparse.load_npz(prefix + '/pro-aptpa.npz').toarray() > 0) * 1
    pro_APTPA = F.normalize(torch.from_numpy(pro_APTPA).type(torch.FloatTensor), dim=1, p=2)

    PRO = [pro_APCPA,pro_APTPA]

    return ADJ, PRO,similarity_matrix, features, labels, num_classes, train_idx, val_idx, test_idx


def load_imdb(prefix=r".\IMDB_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_M.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features = [features_0]
    similarity_matrix = np.load(prefix + "/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)
    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    print(labels)
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    MAM = sp.load_npz(prefix + '/MAM.npz').toarray()
    norm_MAM = F.normalize(torch.from_numpy(MAM).type(torch.FloatTensor), dim=1, p=2)
    MDM = scipy.sparse.load_npz(prefix + '/MDM.npz').toarray()
    norm_MDM = F.normalize(torch.from_numpy(MDM).type(torch.FloatTensor), dim=1, p=2)
    MAMAM = sp.load_npz(prefix + '/MAMAM.npz').toarray()
    norm_MAMAM = F.normalize(torch.from_numpy(MAMAM).type(torch.FloatTensor), dim=1, p=2)
    # MAMDM = sp.load_npz(prefix + '/MAMDM.npz').toarray()
    # norm_MAMDM = F.normalize(torch.from_numpy(MAMDM).type(torch.FloatTensor), dim=1, p=2)
    MDMDM = sp.load_npz(prefix + '/MDMDM.npz').toarray()
    norm_MDMDM = F.normalize(torch.from_numpy(MDMDM).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_MAM,norm_MAMAM,norm_MDMDM]

    pro_MDM = (scipy.sparse.load_npz(prefix + '/pro-mdm.npz').toarray() > 0) * 1
    pro_MDM = F.normalize(torch.from_numpy(pro_MDM).type(torch.FloatTensor), dim=1, p=2)
    pro_MAM = (scipy.sparse.load_npz(prefix + '/pro-mam.npz').toarray() > 0) * 1
    pro_MAM = F.normalize(torch.from_numpy(pro_MAM).type(torch.FloatTensor), dim=1, p=2)
    pro_MAMAM = (scipy.sparse.load_npz(prefix + '/pro-mamam.npz').toarray() > 0) * 1
    pro_MAMAM = F.normalize(torch.from_numpy(pro_MAMAM).type(torch.FloatTensor), dim=1, p=2)
    # pro_MAMDM = (scipy.sparse.load_npz(prefix + '/pro-mamdm.npz').toarray() > 0) * 1
    # pro_MAMDM = F.normalize(torch.from_numpy(pro_MAMDM).type(torch.FloatTensor), dim=1, p=2)
    pro_MDMDM = (scipy.sparse.load_npz(prefix + '/pro-mdmdm.npz').toarray() > 0) * 1
    pro_MDMDM = F.normalize(torch.from_numpy(pro_MDMDM).type(torch.FloatTensor), dim=1, p=2)
    PRO = [pro_MAMAM]

    return ADJ, PRO,similarity_matrix, features, labels, num_classes, train_idx, val_idx, test_idx


def load_Yelp(prefix=r"\4_Yelp"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()  # B:2614
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_u.npz').toarray()  # U:1286
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()  # S:4
    features_3 = scipy.sparse.load_npz(prefix + '/features_3_l.npz').toarray()  # L:9
    features_0 = torch.FloatTensor(features_0)
    # features_1 = torch.FloatTensor(features_1)
    # features_2 = torch.FloatTensor(features_2)
    # features_3 = torch.FloatTensor(features_3)
    features = [features_0]

    similarity_matrix = np.load(prefix + "/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)

    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy', allow_pickle=True)
    train_idx = train_val_test_idx.item()['train_idx'].astype(int)
    val_idx = train_val_test_idx.item()['val_idx']
    test_idx = train_val_test_idx.item()['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    adj = torch.LongTensor(sp.load_npz(prefix + "/adjM.npz").toarray())
    BU = (adj[:2614, 2614:2614 + 1286] > 0) * 1
    BS = (adj[:2614, 2614 + 1286:2614 + 1286 + 4] > 0) * 1
    BL = (adj[:2614, 2614 + 1286 + 4:2614 + 1286 + 4 + 9] > 0) * 1
    NS = [BU, BS]

    BUB = sp.load_npz(prefix + '/adj_bub_one.npz').toarray()
    norm_BUB = F.normalize(torch.from_numpy(BUB).type(torch.FloatTensor), dim=1, p=2)
    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb_one.npz').toarray()
    norm_BSB = F.normalize(torch.from_numpy(BSB).type(torch.FloatTensor), dim=1, p=2)
    BLB = scipy.sparse.load_npz(prefix + '/adj_blb_one.npz').toarray()
    norm_BLB = F.normalize(torch.from_numpy(BLB).type(torch.FloatTensor), dim=1, p=2)
    BLBLB = scipy.sparse.load_npz(prefix + '/blblb.npz').toarray()
    norm_BLBLB = F.normalize(torch.from_numpy(BLBLB).type(torch.FloatTensor), dim=1, p=2)
    # BSBLB = scipy.sparse.load_npz(prefix + '/bsblb.npz').toarray()
    # norm_BSBLB = F.normalize(torch.from_numpy(BSBLB).type(torch.FloatTensor), dim=1, p=2)
    BSBSB = scipy.sparse.load_npz(prefix + '/bsbsb.npz').toarray()
    norm_BSBSB = F.normalize(torch.from_numpy(BSBSB).type(torch.FloatTensor), dim=1, p=2)
    # BSBUB = scipy.sparse.load_npz(prefix + '/bsbub.npz').toarray()
    # norm_BSBUB = F.normalize(torch.from_numpy(BSBUB).type(torch.FloatTensor), dim=1, p=2)
    BUBUB = scipy.sparse.load_npz(prefix + '/bubub.npz').toarray()
    norm_BUBUB = F.normalize(torch.from_numpy(BUBUB).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_BUB,norm_BUBUB,norm_BSBSB,norm_BLBLB]

    pro_BLB = (scipy.sparse.load_npz(prefix + '/pro-blb.npz').toarray() > 0) * 1
    pro_BLB = F.normalize(torch.from_numpy(pro_BLB).type(torch.FloatTensor), dim=1, p=2)
    pro_BUB = (scipy.sparse.load_npz(prefix + '/pro-bub.npz').toarray() > 0) * 1
    pro_BUB = F.normalize(torch.from_numpy(pro_BUB).type(torch.FloatTensor), dim=1, p=2)
    pro_BSB = (scipy.sparse.load_npz(prefix + '/pro-bsb.npz').toarray() > 0) * 1
    pro_BSB = F.normalize(torch.from_numpy(pro_BSB).type(torch.FloatTensor), dim=1, p=2)
    pro_BLBLB = (scipy.sparse.load_npz(prefix + '/pro-blblb.npz').toarray() > 0) * 1
    pro_BLBLB = F.normalize(torch.from_numpy(pro_BLBLB).type(torch.FloatTensor), dim=1, p=2)
    # pro_BSBLB = (scipy.sparse.load_npz(prefix + '/pro-bsblb.npz').toarray() > 0) * 1
    # pro_BSBLB = F.normalize(torch.from_numpy(pro_BSBLB).type(torch.FloatTensor), dim=1, p=2)
    pro_BSBSB = (scipy.sparse.load_npz(prefix + '/pro-bsbsb.npz').toarray() > 0) * 1
    pro_BSBSB = F.normalize(torch.from_numpy(pro_BSBSB).type(torch.FloatTensor), dim=1, p=2)
    # pro_BSBUB = (scipy.sparse.load_npz(prefix + '/pro-bsbub.npz').toarray() > 0) * 1
    # pro_BSBUB = F.normalize(torch.from_numpy(pro_BSBUB).type(torch.FloatTensor), dim=1, p=2)
    pro_BUBUB = (scipy.sparse.load_npz(prefix + '/pro-bubub.npz').toarray() > 0) * 1
    pro_BUBUB = F.normalize(torch.from_numpy(pro_BUBUB).type(torch.FloatTensor), dim=1, p=2)
    PRO = [pro_BUBUB,pro_BSBSB]

    return ADJ, PRO,similarity_matrix, features, labels, num_classes, train_idx, val_idx, test_idx


if __name__ == "__main__":
    load_imdb()
