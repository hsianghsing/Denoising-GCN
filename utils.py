import os
from time import time
from datetime import datetime
import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score
from torch import optim
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import matplotlib as mpl

def get_model_attribute(model, attr_name):
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)



def get_logs_name():
    now = datetime.now()
    tmp = str(now.strftime('%Y%m%d%H%M'))
    file = f"{tmp}"
    return file


def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class SelfSupervisedLoss(nn.Module):
    def __init__(self, device):
        super(SelfSupervisedLoss, self).__init__()
        self.device = device

    def forward(self, emb_1, emb_2):
        emb_1 = F.normalize(emb_1, dim=1)
        emb_2 = F.normalize(emb_2, dim=1)

        similarity = torch.matmul(emb_1, emb_2.t()).to(self.device)
        n, m = similarity.shape
        label = - torch.ones_like(similarity) + torch.eye(n, m).to(self.device) * 2

        pos = (1 - similarity) * (label == 1).float()
        neg = torch.clamp(similarity, min=0) * (label == -1).float()
        return (pos + neg).mean()
class CombineLoss:
    def __init__(self, rec_model, config):
        self.model = rec_model
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.cl_rate = config['cl_rate']
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.cl_loss_function = SelfSupervisedLoss(config['device'])
        self.flag = None

    def stage(self, users, pos, neg, epoch):
        # bpr_loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        bpr_loss, reg_loss = get_model_attribute(self.model, 'bpr_loss')(users, pos, neg)
        # all_users_emb_1, all_items_emb_1 = self.model.computer(cal_type="view1")
        # all_users_emb_2, all_items_emb_2 = self.model.computer(cal_type="view2")
        all_users_emb_1, all_items_emb_1 = get_model_attribute(self.model, 'computer')("view1")
        all_users_emb_2, all_items_emb_2 = get_model_attribute(self.model, 'computer')("view2")
        # all_users_emb_1, all_items_emb_1 = get_model_attribute(self.model, 'computer')("view2")
        # all_users_emb_2, all_items_emb_2 = get_model_attribute(self.model, 'computer')("view3")
        cl_loss = self.cl_loss_function(all_users_emb_1[users], all_users_emb_2[users]) \
                  + self.cl_loss_function(all_items_emb_1[pos], all_items_emb_2[pos])
        total_loss = bpr_loss + self.weight_decay * reg_loss + self.cl_rate * cl_loss
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        # if epoch != 0 and epoch % 20 == 0 and self.flag != epoch :
        #     for p in self.opt.param_groups:
        #         p['lr'] *= 0.9
        #     self.flag = epoch
        #     print("lr:{} ".format(self.opt.state_dict()['param_groups'][0]['lr']))
        return total_loss.cpu().item()

def UniformSample_original(dataset):
    user_num = dataset.trainSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def get_new_R(R, embeddings_users, embeddings_items, threshold=50, recompute=True):
    if not recompute:
        try:
            print("Loading new R from file")
            return torch.load(f'new_R_{threshold}.pt')
        except:
            print("File not found, recompute new R")
            # recompute = True
    print("Using CPU to get new R")
    _device = torch.device('cpu')
    embeddings_users = embeddings_users.to(_device)
    embeddings_items = embeddings_items.to(_device)
    R = R.to(_device)
    weights = torch.mm(embeddings_users, embeddings_items.T)
    attention_weights = torch.softmax(weights, dim=1)
    # torch.save(attention_weights, 'attention_weights.pt')
    flat_scores = attention_weights.cpu().detach().flatten()
    percentile = np.percentile(flat_scores, threshold) # 50% of the scores are below this value

    print("threshold:{}".format(percentile))
    pad_0 = torch.zeros_like(attention_weights)
    pad_1 = torch.ones_like(attention_weights)
    result = torch.where(attention_weights > percentile, pad_1, pad_0)
    result = result.to(_device)
    R_ = torch.mul(R, result)
    # save the new R
    # torch.save(R_, f'new_R_{threshold}.pt')
    return R_


def get_ADJ(R):
    num_users, num_items = R.shape
    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    adj_mat[:num_users, num_users:] = R
    adj_mat[num_users:, :num_users] = R.T
    adj_mat = adj_mat.todok()

    row_sum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    return convert_sp_mat_to_sp_tensor(norm_adj)

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================