"""
Design Dataset here
Every dataset's index has to start at 0
"""
import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from tools import world
from tools.world import cprint
# import cppimport

from scipy.sparse import coo_matrix


class Loader(Dataset):
    """
    Dataset type for pytorch
    Incldue graph information
    """

    def __init__(self, config=world.config, path="./data/gowalla"):
        # train or test
        cprint(f"loading [{path}]")
        self.n_user = 0
        self.m_item = 0
        self.config = config

        self.path = path
        
        self.read_data(path)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.trainDataSize + self.testDataSize} interactions in total")
        print(f"{world.config['dataset']} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()  # 字典testDict[uid] = [pos_iid1, pos_iid2, ...]
        print(f"{world.config['dataset']} is ready to go")

        if world.model_name != "mf":
            if world.model_name == "LightGCL":
                self.get_SVD_matrix()
            else:
                self.getSparseGraph()

        self.train_df = self._sample_pos_neg()
        # self.pos_dic = self._sample_pos_temp()
        self.interaction_tensor = self.create_interaction_tensor().cuda()
        
    def create_interaction_tensor(self):
        interaction_tensor = torch.zeros(self.n_user, self.m_item, dtype=torch.bool)
        for row in self.train_df.itertuples():
            user, pos_set = row.userId, row.pos_items
            interaction_tensor[user, list(pos_set)] = 1
        return interaction_tensor

    def get_item_popularity_group(self, num=10):
        item_counts = torch.bincount(self.trainItem_tensor)
        items = torch.arange(len(item_counts))
        sorted_indices = torch.argsort(item_counts, descending=True)

        # 确定每组应该有多少个item
        num_items = len(sorted_indices)
        num_items_per_group = num_items // num

        group_labels = torch.full((num_items,), -1, dtype=torch.long)

        # 首先将items分配到10个基本组
        for i in range(num_items_per_group * num):
            group_index = i // num_items_per_group
            group_labels[sorted_indices[i]] = group_index

        # 创建一个额外的小组11
        extra_group_index = num  # 因为基本组是从0到9，所以额外的小组索引是10

        # 将剩余的items分配到额外的小组中
        for i in range(num_items_per_group * num, num_items):
            group_labels[sorted_indices[i]] = extra_group_index

        # 输出分组标签
        return group_labels
    
    def get_item_popularity_group_V2(self):
        train_item_array, item_num = self.trainItem, self.m_item
        unique, counts = np.unique(train_item_array, return_counts=True)
        item_counts_tensor = torch.zeros(item_num, dtype=torch.int64)
        item_counts_tensor[unique] = torch.from_numpy(counts)
        item_frequencies = item_counts_tensor.cuda()
        unique_freqs, inverse_indices = torch.unique(item_frequencies, return_inverse=True)
        groupID = inverse_indices
        print("unique_freqs.shape[0]", unique_freqs.shape[0])
        
        return groupID
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def read_data(self, path):
        train_file = path + "/train.txt"
        test_file = path + "/test.txt"
        self.path = path

        trainItem, trainUser = [], []
        testItem, testUser = [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split()  # l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    if len(items) == 0:
                        continue
                    uid = int(l[0])
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.traindataSize += len(items)
        self.m_item = max(trainItem)
        self.n_user = max(trainUser)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split()  # l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    if len(items) == 0:
                        continue
                    uid = int(l[0])
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.testDataSize += len(items)
        self.m_item = max(self.m_item, max(testItem))
        self.n_user = max(self.n_user, max(testUser))
        self.m_item += 1
        self.n_user += 1
        print("Num_item: ", self.m_item)
        print("Nume_user: ", self.n_user)
        self.trainItem = np.array(trainItem)
        self.trainUser = np.array(trainUser)
        self.testItem = np.array(testItem)
        self.testUser = np.array(testUser)

        # new add
        self.trainUser_tensor = torch.LongTensor(self.trainUser)
        self.trainItem_tensor = torch.LongTensor(self.trainItem)

    def _sample_pos_neg(self):
        train_df = pd.DataFrame({"userId": self.trainUser, "itemId": self.trainItem})
        train_df = train_df.groupby("userId")["itemId"].apply(set).reset_index().rename(columns={"itemId": "pos_items"})
        return train_df[["userId", "pos_items"]]

    def _sample_pos_temp(self):
        """
        本函数返回字典，key为userId，value为该user的正样本的idx组成的集合；
        用来test Top负采样的样本中正样本的比例；
        """
        pos_df = pd.DataFrame(
            {
                "userId": np.concatenate((self.trainUser, self.testUser)),
                "itemId": np.concatenate((self.trainItem, self.testItem)),
            }
        )
        pos_df = pos_df.groupby("userId")["itemId"].apply(set).reset_index().rename(columns={"itemId": "pos_items"})
        pos_df = pos_df[["userId", "pos_items"]]
        pos_items_dict = pos_df.set_index("userId")["pos_items"].to_dict()
        return pos_items_dict

    def judge_pair(self, users, items):
        result = torch.ones_like(users)
        for i, (user, item) in enumerate(zip(users, items)):
            if item.item() in self.pos_dic[user.item()]:
                result[i] = 0
        return result.sum()

    def choose_items(self, exclude_set, num, method="method1"):
        """从item pool中选取不在exclude_set中的num个item"""
        if method == "method1":
            sample_neg = cppimport.imp("sample1")
            choose_set = list(sample_neg.choose_items(self.m_item - 1, num, exclude_set))
        elif method == "method2":
            sample_neg = cppimport.imp("sample2")
            choose_set = list(sample_neg.choose_items(self.m_item - 1, num, exclude_set))
        elif method == "method3":
            choose_set = random.choices(range(0, self.m_item), k=num)
            for i in range(len(choose_set)):
                while choose_set[i] in exclude_set:
                    choose_set[i] = random.randint(0, self.m_item - 1)
        elif method == "method4":
            choose_set = set(range(0, self.m_item)) - set(exclude_set)
            choose_set = random.choices(tuple(choose_set), k=num)
        return choose_set

    def get_train_neg_items(self, num_negatives=4):
        users, pos_items, neg_items = [], [], []
        if self.m_item < 5000:
            # sample_method = "method4"
            sample_method = "method1"  # cpp
        else:
            # sample_method = "method3"
            sample_method = "method2"  # cpp
        for row in self.train_df.itertuples():
            user, pos_set = row.userId, row.pos_items
            len_pos_set = len(pos_set)
            if self.config["sample_noise"] != 0:
                random_neg_num = int(num_negatives * len_pos_set * (1 - self.config["sample_noise"]))
                noise_pos_num = num_negatives * len_pos_set - random_neg_num
                random_neg_list = self.choose_items(pos_set, random_neg_num, method=sample_method)
                noise_pos_list = random.choices(list(pos_set), k=noise_pos_num)
                final_neg_list = random_neg_list + noise_pos_list
                # random.shuffle(final_neg_list)
                neg_items += final_neg_list
            else:
                if self.config["sample_method"] == "negative":
                    random_neg_set = self.choose_items(pos_set, num_negatives * len_pos_set, method=sample_method)
                elif self.config["sample_method"] == "random":
                    random_neg_set = random.choices(range(self.m_item), k=num_negatives * len_pos_set)
                neg_items.extend(random_neg_set)
            pos_items.extend(list(pos_set))
            users.extend([int(user)] * len(pos_set))

        users, pos_items, neg_items = torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)
        return users, pos_items, neg_items

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def edges(self):
        """返回2 x datasize的tensor"""
        return torch.stack((torch.tensor(self.trainUser), torch.tensor(self.trainItem)), dim=0)

    def sample_edges(self, num):
        """返回S为num x 2的tensor，每一行为一条边，第一列为user，第二列为item"""
        index = np.random.randint(0, self.traindataSize, num)  # 随机采num条边
        S = np.zeros((num, 2))
        S[:, 0] = self.trainUser[index]
        S[:, 1] = self.trainItem[index]
        return S

    def getSparseGraph(self):
        print("loading adjacency matrix")
        edges = self.edges()
        U = edges[0]
        I = edges[1]
        uI = I + self.n_users  # 偏移n_users个indices

        ind = torch.stack((torch.cat((U, uI)), torch.cat((uI, U))), dim=0)
        tempg = torch.sparse_coo_tensor(
            ind, torch.ones(ind.shape[1]), (self.n_users + self.m_items, self.n_users + self.m_items)
        ).coalesce()  # 创建邻接矩阵的稀疏tensor；coalesce()用于对相同索引的多个值求和
        deg = torch.sparse.sum(tempg, dim=1).to_dense()  # 度向量
        deg = torch.where(deg == 0, torch.tensor(1.0), deg)
        muldeg = torch.pow(deg, -0.5)

        val = tempg.values()
        val = val * muldeg[ind[0]]
        val = val * muldeg[ind[1]]

        deg = deg.cuda()
        G = torch.sparse_coo_tensor(ind, val, (self.n_users + self.m_items, self.n_users + self.m_items))
        G = G.coalesce().cuda()

        # Graph为一个稀疏矩阵、graphdeg为一个向量
        self.Graph, self.graphdeg = G, deg
        self.graphdeg_cpu = self.graphdeg.clone().cpu()
        return True

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        """给定user的list，返回多个list，每个list为一个user的正样本的idx"""
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    # 下面是给LightGCL使用的函数
    def read_txt2coo(self):
        path = os.path.join(self.path, "train.txt")
        data = pd.read_csv(path, delimiter=" ", header=None)
        values = np.ones(len(data))
        user_ids = data[0].values
        item_ids = data[1].values
        coo = coo_matrix((values, (user_ids, item_ids)))
        
        return coo

    def get_SVD_matrix(self):
        
        def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)
        
        train = self.read_txt2coo()  # scipy.sparse.coo.coo_matrix
        train_csr = (train != 0).astype(np.float32)  # 转换为CSR矩阵
        
        # normalizing the adj matrix
        rowD = np.array(train.sum(1)).squeeze()
        colD = np.array(train.sum(0)).squeeze()
        for i in range(len(train.data)):
            train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)
        
        adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)  # torch.sparse.FloatTensor
        adj_norm = adj_norm.coalesce().cuda()  # 合并重复位置的值并排序
        self.adj_norm = adj_norm
        print("Adj matrix normalized.")
        
        train = train.tocoo()
        
        # perform svd reconstruction
        adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda()
        print("Performing SVD...")
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=self.config['q'])  # 对矩阵SVD分解

        self.u_mul_s = svd_u @ (torch.diag(s))  # user_num x 5
        self.v_mul_s = svd_v @ (torch.diag(s))  # item_num x 5
        
        self.ut = svd_u.T
        self.vt = svd_v.T
        
        self.train_csr = train_csr
        
        del s
        print("SVD done.")
        
        