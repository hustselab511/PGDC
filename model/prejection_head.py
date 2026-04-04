# -*- coding: UTF-8 -*-
'''
@Project ：ECG identify 
@File    ：prejection_head.py
@Author  ：yankangli
@Date    ：2025/11/19 19:01 
'''
from torch import nn


class Pro_Classify(nn.Module):
    def __init__(self, encoder=None, num_classes=63, dropout=0.5, output_dim=128,):
        super().__init__()

        self.encoder = encoder
        self.input_dim = encoder.output_dim
        self.output_dim = output_dim
        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim, bias=False),
            nn.LayerNorm(self.input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, output_dim,bias = True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim, bias=False),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.input_dim, num_classes)
        )
    def forward(self, x,**kwargs):
        feat = self.encoder(x,**kwargs)['embedding']
        result = {}
        result['view_feat'] = feat
        result['logit'] = self.classifier(feat)
        result['embedding'] = self.projection_head(feat)
        return result
