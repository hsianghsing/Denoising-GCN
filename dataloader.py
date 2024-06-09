import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from os.path import join
from collections import defaultdict
from sklearn.model_selection import train_test_split


def cprint(words):
    print(f"\033[0;30;43m{words}\033[0m")

class GraphDataset:
    def __init__(self, src):
        super(GraphDataset, self).__init__()
        cprint("loading [{}]".format(src))
        path = "./data/{0}/".format(src)
        self.path = path
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if (str(src) == "ciao") or (str(src) == "lastfm") or (str(src) == "Test"):
            self.trainData = pd.read_csv(join(path, "train_set.txt"))
            self.testData = pd.read_csv(join(path, "test_set.txt"))

        elif str(src) == "Kuaishou":
            _Data_path = join(path, "ratings.txt")

            _Data = pd.read_csv(_Data_path)
            _trainData, _testData = train_test_split(_Data, test_size=0.4, random_state=2022, shuffle=True)
            self.Data = _Data
            self.trainData = _trainData
            self.testData = _testData

        elif str(src) == "douban":
            _Data_path = join(path, 'douban.txt')
            _Data = pd.read_csv(_Data_path)
            _trainData, _testData = train_test_split(_Data, test_size=0.4, random_state=2022, shuffle=True)
            self.Data = _Data
            self.trainData = _trainData
            self.testData = _testData

        elif str(src) == "yelp":
            _Data_path = join(path, 'yelp-ratings.txt')
            _Data = pd.read_csv(_Data_path)
            _trainData, _testData = train_test_split(_Data, test_size=0.4, random_state=2022, shuffle=True)
            self.Data = _Data
            self.trainData = _trainData
            self.testData = _testData

        elif str(src) == "hetrec2011-delicious-2k":
            _Data_path = join(path, 'hetrec2011-interaction.txt')
            _Data = pd.read_csv(_Data_path)
            _trainData, _testData = train_test_split(_Data, test_size=0.4, random_state=2022, shuffle=True)
            self.Data = _Data
            self.trainData = _trainData
            self.testData = _testData

        elif str(src) == "Epinions":
            _Data_path = join(path, 'Epinions-ratings.txt')
            _Data = pd.read_csv(_Data_path)
            _trainData, _testData = train_test_split(_Data, test_size=0.4, random_state=2022, shuffle=True)
            self.Data = _Data
            self.trainData = _trainData
            self.testData = _testData

        self.trainUsers = self.trainData['user'].to_numpy()
        self.trainUniqueUsers = np.unique(self.trainUsers)
        self.trainItems = self.trainData['item'].to_numpy()

        self.testUsers = self.testData['user'].to_numpy()
        self.testUniqueUsers = pd.unique(self.testUsers)
        self.testItems = self.testData['item'].to_numpy()

        self.n_users = pd.concat([self.trainData, self.testData])['user'].nunique()
        self.m_items = pd.concat([self.trainData, self.testData])['item'].nunique()

        self.trainSize = len(self.trainData)
        self.testSize = len(self.testData)

        print(f"{self.trainSize} interactions for training")
        print(f"{self.testSize} interactions for testing")
        print(f"Density : {(self.trainSize + self.testSize) / self.n_users / self.m_items}")

        self.interactionGraph = None
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUsers)),
                                       (self.trainUsers, self.trainItems)),
                                      shape=(self.n_users, self.m_items),
                                      dtype=np.int8)

        self.allPos = self.getUserPosItems(list(range(self.n_users)))
        self.testDict = self.__build_test()
        self.coldTestDict = self.__build_cold_test()
        print(f"Ready to go")

    def __build_test(self):
        test_data = defaultdict(set)
        for user, item in zip(self.testUsers, self.testItems):
            test_data[user].add(item)
        return test_data

    def __build_cold_test(self):
        cold_data = defaultdict(set)
        for user, item in zip(self.testUsers, self.testItems):
            cold_data[user].add(item)
        for key in list(cold_data.keys()):
            if self.trainData['user'].value_counts()[key] > 10:
                del cold_data[key]
        return cold_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getSubGraph(self, _u_list):
        sub_items = []
        sub_users = []
        print("generating subgraph....")
        print(f"_u_list: {len(_u_list)}")
        for _ in range(len(self.trainUsers)):
            if self.trainUsers[_] in _u_list:
                sub_users.append(self.trainUsers[_])
                sub_items.append(self.trainItems[_])
        sub_users = np.array(sub_users)
        sub_items = np.array(sub_items)
        # print(len(sub_users), len(sub_items))
        SubUserItemNet = csr_matrix((np.ones(len(sub_users)),
                                     (sub_users, sub_items)),
                                    shape=(self.n_users, self.m_items))
        adj = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj = adj.tolil()
        R = SubUserItemNet.tolil()
        adj[:self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        adj = adj.todok()

        row_sum = np.array(adj.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        subGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        subGraph = subGraph.coalesce().to(self.device)
        return subGraph

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.interactionGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + "interaction_adj_mat.npz")
                print("successfully loaded ....")
                norm_adj = pre_adj_mat
                self.interactionGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                # self.interactionGraph = self.interactionGraph.coalesce()
                self.interactionGraph = self.interactionGraph.coalesce().to(self.device)
                return self.interactionGraph
            except IOError or FileNotFoundError:
                print("generating adjacency matrix")
                start_ = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                row_sum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(row_sum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end_ = time()
                sp.save_npz(self.path + "interaction_adj_mat.npz", norm_adj)
                print(f"cost {end_ - start_}s, saved norm_mat...")

                self.interactionGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.interactionGraph = self.interactionGraph.coalesce().to(self.device)
                return self.interactionGraph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float16)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class SocialGraphDataset(GraphDataset):
    def __init__(self, src):
        super(SocialGraphDataset, self).__init__(src)
        self.src = src
        if (src == "ciao") or (src == "lastfm") or (src == "Test"):
            self.friendNet = pd.read_csv((self.path + "/trust.txt"))
            self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'],
                                                                        self.friendNet['friend'])),
                                        shape=(self.n_users, self.n_users))
        elif src == "Kuaishou":
            try:
                self.friendNet = pd.read_csv((self.path + "/trust.txt"))
                self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'],
                                                                            self.friendNet['friend'])),
                                            shape=(self.n_users, self.n_users))
            except IOError:
                print("Kuaishou trust.txt need to be processed manually")
        elif src == "douban":
            self.friendNet = pd.read_csv((self.path + "/douban-friend.txt"))
            self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'],
                                                                        self.friendNet['friend'])),
                                        shape=(self.n_users, self.n_users))

        elif src == "yelp":
            self.friendNet = pd.read_csv((self.path + "/yelp-trusts.txt"))
            self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'],
                                                                        self.friendNet['friend'])),
                                        shape=(self.n_users, self.n_users))
        elif src == "hetrec2011-delicious-2k":
            self.friendNet = pd.read_csv((self.path + "/hetrec2011-friend.txt"))
            self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'],
                                                                        self.friendNet['friend'])),
                                        shape=(self.n_users, self.n_users))
        elif src == "Epinions":
            self.friendNet = pd.read_csv((self.path + "/Epinions-trusts.txt"))
            self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'],
                                                                        self.friendNet['friend'])),
                                        shape=(self.n_users, self.n_users))
        self.socialGraph = None
        print("Number of links: {}".format(len(self.friendNet)))
        print("{} Link Density: {}".format(self.src, len(self.friendNet) / self.n_users / self.n_users))

    def getSocialGraph(self):
        if self.socialGraph is None:
            try:
                pre_adj_mat = sp.load_npz("./data/{}/social_adj_mat.npz".format(self.src))
                print("successfully loaded ...")
                norm_adj = pre_adj_mat
                self.socialGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                # self.socialGraph = self.socialGraph.coalesce()
                self.socialGraph = self.socialGraph.coalesce().to(self.device)
                return self.socialGraph
            except IOError:
                print("generating adjacency matrix")
                start = time()
                adj_mat = self.socialNet.tolil()

                row_sum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(row_sum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz("./data/{}/social_adj_mat.npz".format(self.src), norm_adj)

            self.socialGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.socialGraph = self.socialGraph.coalesce().to(self.device)
        return self.socialGraph

    def getDenseSocialGraph(self):
        if self.socialGraph is None:
            self.socialGraph = self.getSocialGraph().to_dense()
        else:
            pass
        return self.socialGraph
# if __name__ == "__main__":
#     # dataset = SocialGraphDataset("lastfm")
#     # dataset.getSocialGraph()
#     # print(dataset.n_users, dataset.m_items)
#     # print(dataset.trainUsers)
#     # print(dataset.trainItems)
#     dataset = SocialGraphDataset('lastfm')
#     A = dataset.getSparseGraph()