from model.model_Base import IRModel
from torch import nn
import torch.nn.functional as F
import torch


class LightGCNModel(IRModel):
    def __init__(self, config: dict, num_users: int, num_items: int, Graph):
        super().__init__(config, num_users, num_items)

        self.Graph = Graph
        self.n_layers = config["n_layers"]
        self.keep_prob = config["keep_prob"]
        self.enable_dropout: bool = config["enable_dropout"]
        self._init_weight()


        print(f"lgn is already to go(dropout:{config['enable_dropout']})")


    def _init_weight(self):
        self.embedding_user = nn.init.xavier_uniform_(torch.empty(self.num_users,self.latent_dim))
        self.embedding_item = nn.init.xavier_uniform_(torch.empty(self.num_items,self.latent_dim))

        self.embedding_user = nn.Parameter(self.embedding_user)
        self.embedding_item = nn.Parameter(self.embedding_item)


    def compute(self):
        """
        propagate methods for lightGCN；
        """
        users_emb = self.embedding_user
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]


        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if self.norm:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)


        return users, items

    def additional_loss(*args, **kwargs):
        return 0