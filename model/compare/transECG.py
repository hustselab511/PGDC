import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.ops import StochasticDepth

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.ops import StochasticDepth


class TransECGBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, p_survival=0.8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.drop_path = StochasticDepth(p=1 - p_survival, mode="batch")

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.drop_path(ffn_out)
        x = self.norm2(x)
        return x


class TransECG(nn.Module):
    def __init__(self,
                 seq_len=1000,
                 patch_size=10,
                 in_channels=1,
                 hidden_size=256,
                 depth=6,
                 heads=8,
                 num_classes=87,
                 dropout=0.1,
                 p_survival=0.8):
        super().__init__()

        assert seq_len % patch_size == 0, \
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})"

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = seq_len // patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.to_patch_embedding = nn.Linear(in_channels * patch_size, hidden_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_size) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList([
            TransECGBlock(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                p_survival=p_survival
            )
            for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def modify_num_classes(self, num_classes):
        self.num_classes = num_classes
        device = next(self.parameters()).device
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        ).to(device)

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.shape

        assert c == self.in_channels, \
            f"Expected {self.in_channels} channels, got {c}"
        assert l == self.seq_len, \
            f"Expected input length {self.seq_len}, got {l}"
        assert l % self.patch_size == 0, \
            f"Input length {l} must be divisible by patch_size {self.patch_size}"

        x = rearrange(x, 'b c (n p) -> b n (c p)', p=self.patch_size)
        x = self.to_patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        assert x.size(1) == self.pos_embedding.size(1), \
            f"Token length mismatch: got {x.size(1)}, expected {self.pos_embedding.size(1)}"

        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.transformer:
            x = block(x)

        cls_output = x[:, 0]
        logits = self.mlp_head(cls_output)
        return logits