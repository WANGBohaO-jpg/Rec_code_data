import os
import pdb
import random
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from tqdm import tqdm
import world
import torch
from dataloader import Loader
import dataloader
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch.distributions.beta import Beta


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def compute_ssm_loss(self, y_pred, temp):
        pos_logits = torch.exp(y_pred[:, 0] / temp)
        neg_logits = torch.exp(y_pred[:, 1:] / temp)
        neg_logits = torch.sum(neg_logits, dim=-1)
        loss = -torch.log(pos_logits).mean() + torch.log(neg_logits).mean()
        
        return loss

    def getUsersRating(self, users):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        items_emb = embedding_item

        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        return loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_ssm_loss(y_pred, self.config["ssm_temp"])

        return loss

    def bce_loss(self, users, pos, neg):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        label = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(scores, label)

        return loss


class PureMF(BasicModel):
    def __init__(self, config: dict, dataset: Loader):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.dataset = dataset
        self.latent_dim = config["latent_dim_rec"]
        self.config = config
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

    def compute(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.config["norm_emb"]:
            users_emb = F.normalize(users_emb, p=2, dim=1)
            items_emb = F.normalize(items_emb, p=2, dim=1)

        return users_emb, items_emb


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: Loader):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.Loader = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint("use NORMAL distribution initilizer")

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        print(f"lgn is already to go(dropout:{self.config['enable_dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def compute(self):
        """
        propagate methods for lightGCN；
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config["enable_dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)

        return users, items


class LightGCL(BasicModel):
    def __init__(self, config: dict, dataset: Loader):
        super(LightGCL, self).__init__()

        self.config = config
        self.dataset: dataloader.Loader = dataset

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items

        self.E_u_0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dataset.n_user, self.config["latent_dim_rec"]))
        )
        self.E_i_0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dataset.m_item, self.config["latent_dim_rec"]))
        )

        self.train_csr = dataset.train_csr  # user_num x item_num的CSR矩阵
        self.adj_norm = dataset.adj_norm  # 邻接矩阵 torch.sparse.FloatTensor
        self.l = self.config["n_layers"]  # 层数
        self.E_u_list = [None] * (self.l + 1)  # 存储每一层聚合得到的embedding
        self.E_i_list = [None] * (self.l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (self.l + 1)  # 存储每一层聚合得到的embedding 感觉没啥用
        self.Z_i_list = [None] * (self.l + 1)
        self.G_u_list = [None] * (self.l + 1)  # 存储每一层用SVD邻接矩阵聚合得到的embedding
        self.G_i_list = [None] * (self.l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = self.config["cl_temp"]
        self.lambda_1 = self.config["cl_rate"]
        # self.lambda_2 = lambda_2
        self.dropout = self.config["keep_prob"]
        self.act = nn.LeakyReLU(0.5)

        self.E_u = None
        self.E_i = None

        self.u_mul_s = dataset.u_mul_s
        self.v_mul_s = dataset.v_mul_s

        self.ut = dataset.ut
        self.vt = dataset.vt

        self.f = nn.Sigmoid()

    def compute(self):
        def sparse_dropout(mat, dropout):
            if dropout == 0.0:
                return mat
            indices = mat.indices()
            values = nn.functional.dropout(mat.values(), p=dropout)
            size = mat.size()
            return torch.sparse.FloatTensor(indices, values, size)

        for layer in range(1, self.l + 1):
            # GNN propagation
            self.Z_u_list[layer] = torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.spmm(
                sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]
            )

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        # 原来的邻接矩阵获得的embedding
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        user_embedding = self.E_u / len(self.E_u_list)
        item_embedding = self.E_i / len(self.E_i_list)

        if self.config["norm_emb"]:
            # self.E_u = F.normalize(self.E_u, p=2, dim=1)
            # self.E_i = F.normalize(self.E_i, p=2, dim=1)
            # self.G_u = F.normalize(self.G_u, p=2, dim=1)
            # self.G_i = F.normalize(self.G_i, p=2, dim=1)
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            item_embedding = F.normalize(item_embedding, p=2, dim=1)

        return user_embedding, item_embedding

    def cal_cl_loss(self, uids, iids):
        self.compute()

        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i

        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
            torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)
        ).mean()
        loss = -pos_score + neg_score

        return loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(users, pos)

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        loss = self.compute_ssm_loss(y_pred, self.config["ssm_temp"], epoch)

        loss = loss + cl_loss

        return loss


class XSimGCL(LightGCN):
    def __init__(self, config: dict, dataset: Loader):
        super(XSimGCL, self).__init__(config, dataset)

    def getUsersRating(self, users):
        embedding_user, embedding_item, _, _ = self.compute()
        users_emb = embedding_user[users.long()]
        items_emb = embedding_item

        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def compute(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config["enable_dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for k in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            random_noise = torch.rand_like(all_emb, device="cuda")
            all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.config["eps"]
            embs.append(all_emb)

            if k == self.config["cl_layer"]:
                all_embeddings_cl = all_emb

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(
            all_embeddings_cl, [self.num_users, self.num_items]
        )

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
            user_all_embeddings_cl = F.normalize(user_all_embeddings_cl, p=2, dim=1)
            item_all_embeddings_cl = F.normalize(item_all_embeddings_cl, p=2, dim=1)

        return users, items, user_all_embeddings_cl, item_all_embeddings_cl

    def InfoNCE(self, view1, view2, temperature: float):
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.config["cl_temp"])
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.config["cl_temp"])

        return user_cl_loss + item_cl_loss

    def bpr_loss(self, users, pos, neg):
        # print(neg.shape)
        neg = neg.squeeze()

        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        loss = loss + cl_loss

        return loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_ssm_loss(y_pred, self.config["ssm_temp"], epoch)

        loss = loss + cl_loss

        return loss

    def bce_loss(self, users, pos, neg):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        label = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(scores, label)

        loss = loss + cl_loss

        return loss


class SimGCL(LightGCN):
    def __init__(self, config: dict, dataset: Loader):
        super(SimGCL, self).__init__(config, dataset)

    def compute(self, perturbed=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config["enable_dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb, device="cuda")
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.config["eps"]
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)

        return users, items

    def InfoNCE(self, view1, view2, temperature: float):
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.compute(perturbed=True)
        user_view_2, item_view_2 = self.compute(perturbed=True)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.config["cl_temp"])
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.config["cl_temp"])
        return user_cl_loss + item_cl_loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss([users, pos])

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_ssm_loss(y_pred, self.config["ssm_temp"], epoch)

        loss = loss + cl_loss

        return loss
