from model.model_Base import IRModel
from torch import nn
import torch.nn.functional as F
import torch


class XSimGCLModel(IRModel):
    def __init__(self, config: dict, num_users: int, num_items: int, Graph):
        super().__init__(config, num_users, num_items)

        self.Graph = Graph
        self.n_layers = config["n_layers"]
        
        # === Model Parameter ===
        self.contrastive_weight     = config["cl_rate"]
        self.noise_modulus          = config["eps"]
        self.contrastive_temp       = config["cl_temp"]
        self.contrastive_layer:int  = config["cl_layer"]
        
        self._init_weight()


        print(f"XSimGCL is already to go(dropout:{config['enable_dropout']})")


    def _init_weight(self):
        self.embedding_user = nn.init.xavier_uniform_(torch.empty(self.num_users,self.latent_dim))
        self.embedding_item = nn.init.xavier_uniform_(torch.empty(self.num_items,self.latent_dim))

        self.embedding_user = nn.Parameter(self.embedding_user)
        self.embedding_item = nn.Parameter(self.embedding_item)



    def compute(self):
        users_emb = self.embedding_user
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])
        embs = []


        for k in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            random_noise = torch.rand_like(all_emb, device="cuda")
            all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.noise_modulus
            embs.append(all_emb)

            if k == self.contrastive_layer:
                self.all_embeddings_cl = all_emb

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])


        if self.norm:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)


        return users, items

    def InfoNCE(self, view1, view2, temperature: float):
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()


    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(  (idx[0].type(torch.long)).clone().detach()   )
        i_idx = torch.unique(  (idx[1].type(torch.long)).clone().detach()   )
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.contrastive_temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.contrastive_temp)

        return user_cl_loss + item_cl_loss
    
    def compute_contrastive_emb(self):
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(
            self.all_embeddings_cl, [self.num_users, self.num_items]
        )
        if self.norm:
            user_all_embeddings_cl = F.normalize(user_all_embeddings_cl, p=2, dim=1)
            item_all_embeddings_cl = F.normalize(item_all_embeddings_cl, p=2, dim=1)
        return user_all_embeddings_cl, item_all_embeddings_cl
    
    def additional_loss(self, usr_idx, pos_idx, embedding_user, embedding_item):
        cl_user_emb, cl_item_emb = self.compute_contrastive_emb()
        return self.contrastive_weight * self.cal_cl_loss(
            [usr_idx, pos_idx], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )