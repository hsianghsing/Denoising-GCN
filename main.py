import numpy as np
import torch
import dataloader
import optuna
import utils
from utils import fix_randomness
from model import Denoise
from utils import get_model_attribute
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

fix_randomness(2022)

def train_model(dataset, recommend_model, loss_class, epoch, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        recommend_model = torch.nn.DataParallel(recommend_model)
    recommend_model = recommend_model.to(device)
    recommend_model.train()

    combined_loss: utils.CombineLoss = loss_class
    S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()

    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)

    total_batch = len(users) // config['bpr_batch'] + 1
    loss = 0.0
    for (batch_i, (batch_users, batch_pos, batch_neg)) in \
            enumerate(utils.minibatch(users, pos_items, neg_items, batch_size=config['bpr_batch'])):
        cri = combined_loss.stage(users=batch_users, pos=batch_pos, neg=batch_neg, epoch=epoch)
        loss += cri
    loss = loss / total_batch
    return loss

def test_model(dataset, recommend_model, config):
    u_batch_size = config['test_u_batch']
    test_dict: dict = dataset.testDict
    recmodel = recommend_model.eval()
    max_K = max(config['topk'])
    results = {'precision': np.zeros(len(config['topk'])),
               'recall': np.zeros(len(config['topk'])),
               'ndcg': np.zeros(len(config['topk']))}

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

            # rating = recmodel.getUsersRating(batch_users_gpu)
            rating = get_model_attribute(recmodel, 'getUsersRating')(batch_users_gpu)
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
            pre_results.append(test_one_batch(x, config['topk']))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        return results

def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    ground_true = X[1]
    r = utils.getLabel(ground_true, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(ground_true, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(ground_true, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}




dataset = dataloader.SocialGraphDataset('ciao')

# test model

config = {
    'layers': 3,
    'bpr_batch': 2048,
    'latent_dim': 64,
    'test_u_batch': 100,
    'decay': 0.001,
    'lr': 0.001,
    'cl_rate': 0.01,
    'topk': [5, 10, 20],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'threshold': [75]
}

model = Denoise(config, dataset)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)
myloss = utils.CombineLoss(model, config)

scheduler = ExponentialLR(myloss.opt, gamma=0.99, verbose=True)

best_recall_20 = 0
best_ndcg_20 = 0
best_precision_20 = 0


for epoch in range(2000):
    loss = train_model(dataset, model, myloss, epoch, config)
    print(f"epoch {epoch}, loss {loss}")
    if epoch % 10 == 0:
        print("[TEST]")
        results = test_model(dataset, model, config)
        print(f"epoch {epoch}, results {results}")
        
        if results['recall'][2] > best_recall_20:
            best_recall_20 = results['recall'][2]
            torch.save(model.state_dict(), 'best_recall_20_ciao_view12.pth')
        
        if results['ndcg'][2] > best_ndcg_20:
            best_ndcg_20 = results['ndcg'][2]
            torch.save(model.state_dict(), 'best_ndcg_20_ciao_view12.pth')
        
        if results['precision'][2] > best_precision_20:
            best_precision_20 = results['precision'][2]
            torch.save(model.state_dict(), 'best_precision_20_ciao_view12.pth')
        
        print(f"Current best recall@20 {best_recall_20}, ndcg@20 {best_ndcg_20}, precision@20 {best_precision_20}")
    
    scheduler.step()

torch.save(model.state_dict(), 'final_ciao_view12.pth')






