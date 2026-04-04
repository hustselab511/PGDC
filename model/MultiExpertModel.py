from typing import Optional, List

import torch
import torch.nn as nn

from model.ExpertEncoder import ExpertEncoder


class MultiExpertModel(nn.Module):
    def __init__(self, experts: Optional[List[nn.Module]] = None,dim = 256):
        super(MultiExpertModel, self).__init__()
        if experts is None:
            self.experts = nn.ModuleList([
                ExpertEncoder(),
                ExpertEncoder()
            ])
        else:
            self.experts = nn.ModuleList(experts)
            
        for expert in self.experts:
            for param in expert.encoder.parameters():
                param.requires_grad = False
        self.num_experts = len(self.experts)
        self.feat_dim_sum = dim * self.num_experts

        self.fc = self.net = nn.Sequential(
            nn.Linear(self.feat_dim_sum, dim*2, bias=False),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim,bias = True),
        )
        
    def forward(self, expert_inputs):
        expert_feats = []
        for expert in self.experts:
            feat = expert.encoder(expert_inputs)['embedding']
            flat_feat = feat.flatten(1)
            expert_feats.append(flat_feat)
        feat = torch.cat(expert_feats, dim=1)
        feat = self.fc(feat)
        return {'embedding': feat}
