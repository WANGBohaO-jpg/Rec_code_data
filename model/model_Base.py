from torch import nn
from abc import ABC, abstractmethod
from dataset.dataloader import Loader
import torch

class IRModel(nn.Module,ABC):
    def __init__(self, config: dict, num_users: int, num_items: int):
        super().__init__()
        self.norm: bool         = config['norm_emb']
        self.latent_dim: int    = config["latent_dim_rec"]

        self.num_users: int = num_users
        self.num_items: int = num_items

        self.f = nn.Sigmoid()

    @abstractmethod
    def _init_weight(self):
        raise NotImplemented

    @abstractmethod
    def compute(self):
        return NotImplemented
    
    @abstractmethod
    def additional_loss(*args, **kwargs):
        return NotImplemented


    def getUsersRating(self, users: torch.Tensor) -> torch.Tensor:
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        items_emb = embedding_item

        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)