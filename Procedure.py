import numpy as np
import torch

import utils
import world


def BPR_train_original(dataset, recommend_model, loss_class):
    _Recmodel = recommend_model
    _Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    S = utils.UniformSample_original(dataset)
    # print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    pos_items = pos_items.to(world.device)
    neg_items = neg_items.to(world.device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // world.config['bpr_batch'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in \
            enumerate(utils.minibatch(users, pos_items, neg_items, batch_size=world.config['bpr_batch'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return aver_loss


def My_train(dataset, recommend_model, loss_class, epoch):
    _Recmodel = recommend_model
    _Recmodel.train()
    all_loss: utils.MyLoss = loss_class
    S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    pos_items = pos_items.to(world.device)
    neg_items = neg_items.to(world.device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // world.config['bpr_batch'] + 1
    loss = 0.0
    for (batch_i, (batch_users, batch_pos, batch_neg)) in \
            enumerate(utils.minibatch(users, pos_items, neg_items, batch_size=world.config['bpr_batch'])):
        cri = all_loss.stage(users=batch_users, pos=batch_pos, neg=batch_neg, epoch=epoch)
        loss += cri
    loss = loss / total_batch
    return loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    ground_true = X[1]
    r = utils.getLabel(ground_true, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(ground_true, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(ground_true, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, cold=False):
    u_batch_size = world.config['test_u_batch']
    if cold:
        test_dict: dict = dataset.coldTestDict
    else:
        test_dict: dict = dataset.testDict
    recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(test_dict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [test_dict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            # batch_users_gpu = batch_users_gpu.to(world.device)

            rating = recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print(results)
        return results
