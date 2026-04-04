# -*- coding: UTF-8 -*-
'''
@Project ：ECG identify 
@File    ：ExpertEncoder.py
@Author  ：yankangli
@Date    ：2025/10/17 16:02 
'''
import os
import sys
import torch
from torch import nn

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from model.before_input import init_x
from model.prejection_head import Pro_Classify

class ExpertEncoder(nn.Module):
    def __init__(self, encoder = Pro_Classify, seg ='seg', **kwargs):
        super(ExpertEncoder, self).__init__()
        self.encoder = encoder
        self.seg = seg
        self.output_dim = encoder.output_dim

    def init_x(self, x):
        def process(t):
            t = init_x(t, seg=self.seg)
            return t
        if isinstance(x, torch.Tensor):
            return process(x)
        x = process(x)
        return x

    def forward(self, x):
        x = self.init_x(x)
        out = self.encoder(x)
        return out

class MultiExpert(nn.Module):
    def __init__(self, experts,**kwargs):
        super(MultiExpert, self).__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)
        self.output_dim = experts[0].output_dim
        self.seg = 'seg_list'
    
    def getExperts(self):
        experts = []
        for expert in self.experts:
            expert.encoder = expert.encoder.encoder
            experts.append(expert)
        return experts
    
    def forward(self, x):
        expert_feats = {}
        for expert in self.experts:
            feat = expert(x)
            expert_feats[expert.seg] = feat
        return expert_feats
