import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils import *

class Denoise(nn.Module):
    def __init__(self, configs, datasets):
        super(Denoise, self).__init__()

        self.new_A = None
        self.new_A_2 = None
        self.config = configs
        self.dataset = datasets

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self._init_weight()

    def _init_weight(self):

        self.latent_dim = self.config['latent_dim']
        self.n_layers = self.config['layers']

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim).to(self.device)

        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.leaky = torch.nn.LeakyReLU().to(self.device)

        self.f = nn.Sigmoid()

        self.A = self.dataset.getSparseGraph().to(self.device)
        self.R = utils.convert_sp_mat_to_sp_tensor(self.dataset.UserItemNet).to_dense()
        self.socialGraph = self.dataset.getSocialGraph().to(self.device)
        self.User_Embedding_Combine = ANewFusionModel(self.latent_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.01)
                torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out")
                torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

    def computer(self, cal_type="view1"):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if cal_type == "view1":
            ego_emb = torch.cat([users_emb, items_emb]).to(self.device)
            social_emb = users_emb
            all_emb = [ego_emb]

            for k in range(self.n_layers):
                ego_emb = torch.sparse.mm(self.A, ego_emb)
                social_emb = torch.sparse.mm(self.socialGraph, social_emb)
                _user_emb, _item_emb = torch.split(ego_emb, [self.num_users, self.num_items])
                social_emb = self.User_Embedding_Combine(_user_emb, social_emb)
                ego_emb = torch.cat([social_emb, _item_emb])
                all_emb.append(ego_emb)
            all_emb = torch.stack(all_emb, dim=1)
            all_emb = torch.mean(all_emb, dim=1)
            _users, _items = torch.split(all_emb, [self.num_users, self.num_items])
            return _users, _items

        elif cal_type == "view2":
            ego_emb = torch.cat([users_emb, items_emb]).to(self.device)
            social_emb = users_emb
            all_emb = [ego_emb]

            eu1, ei1 = self.computer()
            if self.new_A is None:
                self.new_A = utils.get_ADJ(utils.get_new_R(self.R, eu1, ei1, threshold=self.config['threshold'][0], recompute=False))
            self.new_A = self.new_A.to(self.device)

            for k in range(self.n_layers):
                ego_emb = torch.sparse.mm(self.new_A, ego_emb)
                social_emb = torch.sparse.mm(self.socialGraph, social_emb)
                _user_emb, _item_emb = torch.split(ego_emb, [self.num_users, self.num_items])
                social_emb = self.User_Embedding_Combine(_user_emb, social_emb)
                ego_emb = torch.cat([social_emb, _item_emb])
                all_emb.append(ego_emb)

            all_emb = torch.stack(all_emb, dim=1)
            all_emb = torch.mean(all_emb, dim=1)
            _users, _items = torch.split(all_emb, [self.num_users, self.num_items])
            return _users, _items
        
        elif cal_type == "view3":
            ego_emb = torch.cat([users_emb, items_emb]).to(self.device)
            social_emb = users_emb
            all_emb = [ego_emb]

            eu1, ei1 = self.computer()
            if self.new_A_2 is None:
                self.new_A_2 = utils.get_ADJ(utils.get_new_R(self.R, eu1, ei1, threshold=self.config['threshold'][1], recompute=False))
            self.new_A_2 = self.new_A_2.to(self.device)

            for k in range(self.n_layers):
                ego_emb = torch.sparse.mm(self.new_A_2, ego_emb)
                social_emb = torch.sparse.mm(self.socialGraph, social_emb)
                _user_emb, _item_emb = torch.split(ego_emb, [self.num_users, self.num_items])
                social_emb = self.User_Embedding_Combine(_user_emb, social_emb)
                ego_emb = torch.cat([social_emb, _item_emb])
                all_emb.append(ego_emb)

            all_emb = torch.stack(all_emb, dim=1)
            all_emb = torch.mean(all_emb, dim=1)
            _users, _items = torch.split(all_emb, [self.num_users, self.num_items])
            return _users, _items

    def bpr_loss(self, _users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         users_emb0, pos_emb0, neg_emb0) = self.getEmbedding(_users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)) / float(len(_users))

        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_scores - neg_scores))
        loss = torch.mean(loss)

        return loss, reg_loss

    def getUsersRating(self, _users):
        all_users, all_items = self.computer()
        users_emb = all_users[_users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, _users, pos_items, neg_items):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        all_users, all_items = self.computer()
        users_emb = all_users[_users].to(device)
        pos_emb = all_items[pos_items].to(device)
        neg_emb = all_items[neg_items].to(device)
        users_emb_ego = self.embedding_user(_users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


class ANewFusionModel(nn.Module):
    def __init__(self, embed_dim):
        super(ANewFusionModel, self).__init__()
        # Idea Comes from "Graph Attention Networks for Neural Social Recommendation"
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.mish = nn.Mish()
        self.fc1 = nn.Linear(embed_dim * 3, embed_dim * 3).to(self.device)
        self.fc2 = nn.Linear(embed_dim * 3, embed_dim * 3).to(self.device)
        self.fc3 = nn.Linear(embed_dim * 3, embed_dim).to(self.device)

    def forward(self, x, y):
        x_y = torch.addcmul(torch.zeros_like(x), x, y)
        _cat = torch.cat((x, y, x_y), dim=1)
        tmp1 = self.mish(self.fc1(_cat))
        tmp2 = self.mish(self.fc2(tmp1))
        tmp3 = self.fc3(tmp2)
        out = tmp3 / tmp3.norm(2)
        return out