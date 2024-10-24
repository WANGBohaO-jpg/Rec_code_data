from model.model_Base import IRModel
from torch import nn

import torch
import torch.nn.functional as F

class MFModel(IRModel):
    def __init__(self, config: dict, num_users: int, num_items: int, Graph = None):
        super().__init__(config, num_users, num_items)
        self._init_weight()
    def _init_weight(self):
        self.embedding_user = nn.init.xavier_uniform_(torch.empty(self.num_users,self.latent_dim))
        self.embedding_item = nn.init.xavier_uniform_(torch.empty(self.num_items,self.latent_dim))
        self.embedding_user = nn.Parameter(self.embedding_user)
        self.embedding_item = nn.Parameter(self.embedding_item)


    def compute(self):
        users_emb = self.embedding_user
        items_emb = self.embedding_item

        if self.norm:
            users_emb = F.normalize(input = users_emb, p = 2, dim = 1)
            items_emb = F.normalize(input = items_emb, p = 2, dim = 1)

        return users_emb, items_emb
    
    def additional_loss(*args, **kwargs):
        return 0