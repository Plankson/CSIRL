import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model.sac_models import SquashedNormal
def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn=dropout(attn)
    res = torch.matmul(attn, v)
    return res, attn

class Ego_attention_state_action(nn.Module):
    def __init__(self, feature_size, n_head=2, dropout=None):
        super(Ego_attention_state_action, self).__init__()
        self.feature_per_head=int(feature_size/n_head)
        self.feature_size=feature_size
        self.n_head=n_head
        self.dropout=dropout
        self.ego_Q = nn.Linear(feature_size, feature_size, bias=False)
        self.ego_K = nn.Linear(feature_size, feature_size, bias=False)
        self.ego_V = nn.Linear(feature_size, feature_size, bias=False)
        self.other_K = nn.Linear(feature_size, feature_size, bias=False)
        self.other_V = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_comb = nn.Linear(feature_size,feature_size,bias=False)

    def forward(self,ego,others,mask=None):
        batch_num = others.shape[0]
        other_entity = others.shape[1]
        # Dimension: batch entity head a
        ego_q = self.ego_Q(ego).view(batch_num, 1, self.n_head, self.feature_per_head)
        ego_k = self.ego_K(ego).view(batch_num, 1, self.n_head, self.feature_per_head)
        ego_v = self.ego_V(ego).view(batch_num, 1, self.n_head, self.feature_per_head)
        other_k = self.other_K(others).view(batch_num, other_entity, self.n_head, self.feature_per_head)
        other_v = self.other_V(others).view(batch_num, other_entity, self.n_head, self.feature_per_head)
        all_k=torch.cat((ego_k, other_k), dim=1)
        all_v=torch.cat((ego_v, other_v), dim=1)
        # Dimension: batch head entity a
        ego_q = ego_q.permute(0, 2, 1, 3)
        all_k = all_k.permute(0, 2, 1, 3)
        all_v = all_v.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view(batch_num, 1, 1, other_entity+1).repeat((1, self.n_head, 1, 1))
        value, attn = attention(ego_q, all_k, all_v, mask, self.dropout)
        res = (self.attention_comb(value.reshape((batch_num,self.feature_size)))+ego.squeeze(1))/2
        return res, attn

class Ego_attention_state(nn.Module):
    def __init__(self, feature_size, n_head=2, dropout=None):
        super(Ego_attention_state, self).__init__()
        self.feature_per_head=int(feature_size/n_head)
        self.feature_size=feature_size
        self.n_head=n_head
        self.dropout=dropout
        self.ego_Q = nn.Linear(feature_size, feature_size, bias=False)
        self.all_K = nn.Linear(feature_size, feature_size, bias=False)
        self.all_V = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_comb = nn.Linear(feature_size,feature_size,bias=False)

    def forward(self,ego,others,mask=None):
        batch_num = others.shape[0]
        all_entity = others.shape[1] + 1
        all_input = torch.cat((ego.view(batch_num,1,self.feature_size),others), dim=1)
        # Dimension: batch entity head a
        ego_q = self.ego_Q(ego).view(batch_num, 1, self.n_head, self.feature_per_head)
        all_k = self.all_K(all_input).view(batch_num, all_entity, self.n_head, self.feature_per_head)
        all_v = self.all_V(all_input).view(batch_num, all_entity, self.n_head, self.feature_per_head)
        # Dimension: batch head entity a
        ego_q = ego_q.permute(0, 2, 1, 3)
        all_k = all_k.permute(0, 2, 1, 3)
        all_v = all_v.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view(batch_num, 1, 1, all_entity).repeat((1, self.n_head, 1, 1))
        value, attn = attention(ego_q, all_k, all_v, mask, self.dropout)
        res = (self.attention_comb(value.reshape((batch_num,self.feature_size)))+ego.squeeze(1))/2
        return res, attn

class Attention_crtic(nn.Module):
    def __init__(self,action_dim,feature_per_entity=7,feature_size=64, presence_feature_idx=0):
        super(Attention_crtic, self).__init__()
        self.presence_feature_idx=presence_feature_idx
        self.feature_per_entity=feature_per_entity
        self.ego_embed=nn.Sequential(
            nn.Linear(feature_per_entity+action_dim,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU()
        )
        self.other_embed=nn.Sequential(
            nn.Linear(feature_per_entity,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU()
        )
        self.attention_layer=Ego_attention_state(feature_size)
        self.output_layer = nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,1)
        )

    def split(self, obs):
        ## Dimension: batch, entities, features
        batch_num=obs.size(0)
        obs_contain=obs.size(1)
        work_obs=obs.view(batch_num,(int)(obs_contain/self.feature_per_entity),self.feature_per_entity)
        ego=work_obs[:, 0:1, :]
        other=work_obs[:, 1:, :]
        mask = work_obs[:, :, self.presence_feature_idx:self.presence_feature_idx + 1] < 0.5
        return ego, other, mask

    def forward(self, obs, act):
        ego, other, mask = self.split(obs)
        act=act.unsqueeze(1)
        weight_value, _ = self.attention_layer(self.ego_embed(torch.cat((ego,act),dim=-1)), self.other_embed(other), mask)
        weight_value = self.output_layer(weight_value)
        return weight_value

    def get_attention(self, obs, act):
        ego, other, mask = self.split(obs)
        torch.cat((obs,act),dim=-1)
        _, attn = self.attention_layer(self.ego_embed(torch.cat((ego,act),dim=-1)), self.other_embed(other), mask)
        return attn

class Attention_actor(nn.Module):
    def __init__(self,action_dim,feature_per_entity=7,feature_size=64, presence_feature_idx=0):
        super(Attention_actor, self).__init__()
        self.feature_per_entity=feature_per_entity
        self.presence_feature_idx=presence_feature_idx
        self.ego_embed=nn.Sequential(
            nn.Linear(feature_per_entity,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU()
        )
        self.other_embed=nn.Sequential(
            nn.Linear(feature_per_entity,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU()
        )
        self.attention_layer=Ego_attention_state(feature_size)
        self.output_layer_mean = nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,action_dim)
        )
        self.output_layer_logstd = nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,action_dim)
        )

    def split(self, obs):
        ## Dimension: batch, entities, features
        batch_num=obs.size(0)
        obs_contain=obs.size(1)
        work_obs=obs.view(batch_num,(int)(obs_contain/self.feature_per_entity),self.feature_per_entity)
        ego=work_obs[:, 0:1, :]
        other=work_obs[:, 1:, :]
        mask = work_obs[:, :, self.presence_feature_idx:self.presence_feature_idx + 1] < 0.5
        return ego, other, mask

    def forward(self,obs):
        ego, other, mask = self.split(obs)
        weight_value, _ = self.attention_layer(self.ego_embed(ego), self.other_embed(other), mask)
        mean = self.output_layer_mean(weight_value)
        log_std = self.output_layer_logstd(weight_value)
        std=log_std.exp()
        dist = SquashedNormal(mean, std)
        return dist

    def sample(self,obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, dist.mean