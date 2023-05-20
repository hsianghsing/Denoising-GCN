import os
from time import time
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
import scipy.sparse as sp
import world
import torch.nn as nn
import torch.nn.functional as F
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import matplotlib as mpl


# def attention_fuse(A, B):
#     weights_1 = torch.matmul(A, B.transpose(0, 1))
#     weights_1 = torch.softmax(weights_1, dim=-1)
#
#     weights_2 = torch.matmul(B, A.transpose(0, 1))
#     weights_2 = torch.softmax(weights_2, dim=-1)
#
#     attention = torch.matmul(weights_1, B) + torch.matmul(weights_2, A)
#     return attention / attention.norm(2)


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


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


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


class MyLoss:
    def __init__(self, rec_model, config):
        self.model = rec_model
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.cl_rate = config['cl_rate']
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.cl_loss_function = MyLossV2()
        self.flag = None

    def stage(self, users, pos, neg, epoch):
        bpr_loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        all_users_emb_1, all_items_emb_1 = self.model.computer(cal_type="view1")
        all_users_emb_2, all_items_emb_2 = self.model.computer(cal_type="view2")
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


def getFileName():
    file = None
    now = datetime.now()
    tmp = str(now.strftime('%Y%m%d%H%M'))
    if world.model_name == 'bpr':
        file = f"{tmp}-mf-{world.dataset}-{world.config['latent_dim']}.pth.tar"
    elif world.model_name in ['LightGCN', 'PAGCN', "TEST"]:
        file = f"{tmp}-{world.model_name}-{world.dataset}-{world.config['layers']}layer-" \
               f"{world.config['latent_dim']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


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


def get_new_R(R, embeddings_users, embeddings_items):
    print("Using CPU to get new R")
    _device = torch.device('cpu')
    embeddings_users = embeddings_users.to(_device)
    embeddings_items = embeddings_items.to(_device)
    R = R.to(_device)
    weights = torch.mm(embeddings_users, embeddings_items.T)
    attention_weights = torch.softmax(weights, dim=1)
    # torch.save(attention_weights, 'attention_weights.pt')
    flat_scores = attention_weights.cpu().detach().flatten()
    threshold = np.percentile(flat_scores, 50)
    # 25 和 15 的区别并不大
    print("threshold:{}".format(threshold))
    pad_0 = torch.zeros_like(attention_weights)
    pad_1 = torch.ones_like(attention_weights)
    result = torch.where(attention_weights > threshold, pad_1, pad_0)
    result = result.to(_device)
    R_ = torch.mul(R, result)
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


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


# class ContrastiveLossInfoNCE(nn.Module):
#     def __init__(self, temperature=world.config["temperature"]):
#         super(ContrastiveLossInfoNCE, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, view1, view2):
#         view1 = F.normalize(view1, dim=1)
#         view2 = F.normalize(view2, dim=1)
#         pos_score = (view1 * view2).sum(dim=-1)
#         pos_score = torch.exp(pos_score / self.temperature)
#         ttl_score = torch.matmul(view1, view2.transpose(0, 1))
#         ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
#         _cl_loss = -torch.log(pos_score / ttl_score + 1e-6)
#         return torch.mean(_cl_loss)


class MyLossV2(nn.Module):
    def __init__(self):
        super(MyLossV2, self).__init__()

    def forward(self, emb_1, emb_2):
        emb_1 = F.normalize(emb_1, dim=1)
        emb_2 = F.normalize(emb_2, dim=1)

        similarity = torch.matmul(emb_1, emb_2.t()).to(world.device)
        n, m = similarity.shape
        label = - torch.ones_like(similarity) + torch.eye(n, m).to(world.device) * 2

        pos = (1 - similarity) * (label == 1).float()
        neg = torch.clamp(similarity, min=0) * (label == -1).float()
        return (pos + neg).mean()


def print_result(_res, _cold=False, _finally=False):
    ndcg_5 = _res['ndcg'][0]
    ndcg_10 = _res['ndcg'][1]
    ndcg_20 = _res['ndcg'][2]

    recall_5 = _res['recall'][0]
    recall_10 = _res['recall'][1]
    recall_20 = _res['recall'][2]

    pre_5 = _res['precision'][0]
    pre_10 = _res['precision'][1]
    pre_20 = _res['precision'][2]
    if _finally:
        print(recall_5, ndcg_5, pre_5,
              recall_10, ndcg_10, pre_10,
              recall_20, ndcg_20, pre_20)
    else:
        if _cold:
            print(f"cold_recall_5:{recall_5}  cold_ndcg_5:{ndcg_5}  cold_pre_5:{pre_5}")
            print(f"cold_recall_10:{recall_10}  cold_ndcg_10:{ndcg_10}  cold_pre_10:{pre_10}")
            print(f"cold_recall_20:{recall_20}  cold_ndcg_20:{ndcg_20}  cold_pre_20:{pre_20}")
        else:
            print(f"recall_5:{recall_5}  ndcg_5:{ndcg_5}  pre_5:{pre_5}")
            print(f"recall_10:{recall_10}  ndcg_10:{ndcg_10}  pre_10:{pre_10}")
            print(f"recall_20:{recall_20}  ndcg_20:{ndcg_20}  pre_20:{pre_20}")


def plot_two_graphs(best_data, best_epoch):
    X = np.array(best_data.cpu().detach())
    # plt.figure()
    plt.subplot(211)
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X)

    # 将嵌入归一化到单位圆上
    X_embedded_norm = X_embedded / np.linalg.norm(X_embedded, axis=1).reshape(-1, 1)
    plt.scatter(X_embedded_norm[:, 0], X_embedded_norm[:, 1])
    plt.axis("equal")
    plt.subplot(212)

    # 归一化向量到S1上
    vecs = best_data.cpu().detach()
    norms = np.linalg.norm(vecs, axis=1)
    vecs_norm = vecs / norms[:, None]

    # 将向量转换为角度
    theta = np.arctan2(vecs_norm[:, 1], vecs_norm[:, 0])

    # 使用KernelDensity估计密度
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(theta[:, np.newaxis])
    theta_range = np.linspace(-np.pi, np.pi, 100)
    log_density = kde.score_samples(theta_range[:, np.newaxis])
    density = np.exp(log_density)

    # 可视化密度估计结果
    plt.plot(theta_range, density)
    #     plt.hist(theta, density=True, bins=50, alpha=0.5)
    plt.xlabel('Angle')
    plt.ylabel('Density')
    plt.savefig("{}.png".format(str(best_epoch)))
    # plt.show()
    plt.close()


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
