import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp


def load_dblp(prefix=r".\DBLP_processed"):
    adj = sp.load_npz(prefix+"/adjM.npz")
    AP=adj[:4057,4057:4057+14328]
    PC = adj[4057:4057+14328,4057+14328:4057+14328+7723]
    PA = AP.T
    CP = PC.T
    D_AP = AP.sum(axis=1)
    D_PA = PA.sum(axis=1)
    D_PC = PC.sum(axis=1)
    D_CP = CP.sum(axis=1)

    D_AP[D_AP==0] = 1e-10
    D_PA[D_PA==0] = 1e-10
    D_PC[D_PC==0] = 1e-10
    D_CP[D_CP==0] = 1e-10

    D_AP = (1/D_AP).A
    D_PA = (1/D_PA).A
    D_PC = (1/D_PC).A
    D_CP = (1/D_CP).A

    normalized_AP = AP.toarray() * D_AP[:,np.newaxis]
    normalized_PA = PA.toarray() * D_PA[:,np.newaxis]
    normalized_PC = PC.toarray() * D_PC[:,np.newaxis]
    normalized_CP = CP.toarray() * D_CP[:,np.newaxis]
    normalized_APC = normalized_AP * normalized_PC
    # normalized_CPA = normalized_CP * normalized_PA
    # normalized_APCPA = normalized_APC * normalized_CPA
    print(normalized_APC.shape)
    print(type(normalized_APC))
    np.save(prefix+"\\normalized_APC.npy",normalized_APC)






if __name__ == "__main__":
    load_dblp()
