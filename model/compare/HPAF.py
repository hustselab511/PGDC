import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Conv1d 需要 (B, C, L)
    支持:
    - (B, L) -> (B, 1, L)
    - (B, 1, L) 保持不变
    - (B, 1, 1, L) -> (B, 1, L)
    """
    if x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    elif x.ndim != 3:
        raise ValueError(f"Unsupported input shape for Conv1d: {x.shape}")
    return x

# =========================================================
# 1. Learnable Gabor Conv1d
# =========================================================
class LearnableGaborConv1d(nn.Module):
    """
    Learnable 1D Gabor convolution.
    输入:  x -> (B, C_in, L)
    输出:  y -> (B, C_out, L_out)

    论文对应:
        g_k(t) = exp(-t^2 / (2 sigma_k^2)) * cos(2*pi*f_k*t + psi_k)
        再做 zero-mean 约束
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 31,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 建议取奇数"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding

        # 可学习 Gabor 参数
        self.log_sigma = nn.Parameter(torch.zeros(out_channels))
        self.freq = nn.Parameter(torch.rand(out_channels) * 0.25)
        self.psi = nn.Parameter(torch.zeros(out_channels))

        # 输出通道对输入通道的缩放
        self.channel_scale = nn.Parameter(
            torch.randn(out_channels, in_channels) * 0.1
        )

    def _build_gabor_kernels(self, device, dtype):
        t = torch.linspace(-1.0, 1.0, steps=self.kernel_size, device=device, dtype=dtype)
        t = t.unsqueeze(0)  # (1, K)

        sigma = torch.exp(self.log_sigma).unsqueeze(1) + 1e-4
        freq = self.freq.unsqueeze(1)
        psi = self.psi.unsqueeze(1)

        g = torch.exp(-(t ** 2) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * freq * t + psi)
        g = g - g.mean(dim=1, keepdim=True)  # zero-mean

        # (C_out, C_in, K)
        g = g.unsqueeze(1) * self.channel_scale.unsqueeze(-1)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._build_gabor_kernels(x.device, x.dtype)
        return F.conv1d(x, weight, bias=None, stride=self.stride, padding=self.padding)


# =========================================================
# 2. Multi-Scale Feature Block
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleFeatureBlock(nn.Module):
    """
    论文 MSFB:
    kernel sizes = {7, 15, 17, 39, 41}
    concat 后用 1x1 conv 融合，再下采样/映射
    """
    def __init__(
        self,
        in_ch: int,
        branch_ch: int = 32,
        out_ch: int = 64,
        kernels: Tuple[int, ...] = (7, 15, 17, 39, 41),
        downsample: bool = True,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNAct(in_ch, branch_ch, kernel_size=k) for k in kernels]
        )

        fuse_layers = [
            nn.Conv1d(branch_ch * len(kernels), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if downsample:
            fuse_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.fuse = nn.Sequential(*fuse_layers)

    def forward(self, x):
        xs = [branch(x) for branch in self.branches]
        x = torch.cat(xs, dim=1)
        x = self.fuse(x)
        return x


# =========================================================
# 3. MFEB / VFEB
# =========================================================
class MFEB(nn.Module):
    """
    Morphology Feature Extraction Branch
    """
    def __init__(
        self,
        in_ch: int = 1,
        stem_ch: int = 32,
        ms_out: int = 64,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.stem = ConvBNAct(in_ch, stem_ch, kernel_size=7)
        self.msfb = MultiScaleFeatureBlock(stem_ch, branch_ch=32, out_ch=ms_out)
        self.head = nn.Sequential(
            ConvBNAct(ms_out, ms_out, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(ms_out, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.msfb(x)
        x = self.head(x).squeeze(-1)
        z_m = self.proj(x)
        return z_m


class VFEB(nn.Module):
    """
    Variation Feature Extraction Branch
    """
    def __init__(
        self,
        in_ch: int = 1,
        gabor_ch: int = 32,
        ms_out: int = 64,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.gabor = LearnableGaborConv1d(in_channels=in_ch, out_channels=gabor_ch, kernel_size=31)
        self.bn = nn.BatchNorm1d(gabor_ch)
        self.act = nn.ReLU(inplace=True)

        self.msfb = MultiScaleFeatureBlock(gabor_ch, branch_ch=32, out_ch=ms_out)
        self.head = nn.Sequential(
            ConvBNAct(ms_out, ms_out, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(ms_out, embed_dim)

    def forward(self, x):
        x = self.gabor(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.msfb(x)
        x = self.head(x).squeeze(-1)
        z_v = self.proj(x)
        return z_v


# =========================================================
# 4. PR-GAT
# =========================================================
class ScoreMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, N, D)
        return self.net(x).squeeze(-1)  # (B, N)


class PRGAT(nn.Module):
    """
    论文里的 PR-GAT 近似实现:
    - stack two nodes
    - Q/K/V
    - softmax(LeakyReLU(QK^T / sqrt(dk)))
    - residual + norm + FFN
    - attention pooling
    """
    def __init__(self, dim: int, attn_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        attn_dim = dim if attn_dim is None else attn_dim

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.score_mlp = ScoreMLP(dim)

    def forward(self, x):
        """
        x: (B, 2, D) or (B, N, D)
        return:
            fused: (B, D)
            alpha: (B, N)
            refined: (B, N, D)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale = math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-1, -2)) / (scale + 1e-8)
        attn = F.leaky_relu(attn, negative_slope=0.2)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out))
        ffn_out = self.ffn(x)
        refined = self.norm2(x + self.dropout(ffn_out))

        scores = self.score_mlp(refined)
        alpha = F.softmax(scores, dim=-1)
        fused = (alpha.unsqueeze(-1) * refined).sum(dim=1)
        fused = F.layer_norm(fused, fused.shape[-1:])

        return fused, alpha, refined


# =========================================================
# 5. MVFE
# =========================================================
class MVFE(nn.Module):
    """
    一个 phase-specific MVFE:
        MFEB + VFEB + PR-GAT
    """
    def __init__(
        self,
        in_ch: int = 1,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mfeb = MFEB(in_ch=in_ch, embed_dim=embed_dim)
        self.vfeb = VFEB(in_ch=in_ch, embed_dim=embed_dim)
        self.prgat = PRGAT(dim=embed_dim, dropout=dropout)

    def forward(self, x):
        z_m = self.mfeb(x)
        z_v = self.vfeb(x)

        nodes = torch.stack([z_v, z_m], dim=1)  # (B, 2, D)
        h_phase, alpha, refined = self.prgat(nodes)

        aux = {
            "z_m": z_m,
            "z_v": z_v,
            "alpha_vm": alpha,       # [variation, morphology]
            "refined_vm": refined,
        }
        return h_phase, aux


# =========================================================
# 6. PGHF
# =========================================================
class PGHF(nn.Module):
    """
    Phase-Grouped Hierarchical Fusion
    slow group: P + TU
    fast group: QRS + ST
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.slow_fuser = PRGAT(dim=dim, dropout=dropout)
        self.fast_fuser = PRGAT(dim=dim, dropout=dropout)

    def forward(self, h_p, h_qrs, h_st, h_tu):
        slow_nodes = torch.stack([h_p, h_tu], dim=1)
        fast_nodes = torch.stack([h_qrs, h_st], dim=1)

        h_slow, alpha_slow, refined_slow = self.slow_fuser(slow_nodes)
        h_fast, alpha_fast, refined_fast = self.fast_fuser(fast_nodes)

        aux = {
            "alpha_slow": alpha_slow,   # [P, TU]
            "alpha_fast": alpha_fast,   # [QRS, ST]
            "refined_slow": refined_slow,
            "refined_fast": refined_fast,
        }
        return h_slow, h_fast, aux


# =========================================================
# 7. GRF
# =========================================================
class GRF(nn.Module):
    """
    Global Representation Fusion

    论文中写法更强调:
    - 将两组特征映射到共享 scoring space
    - 计算 scalar attention-like gate
    - 做加权混合
    - 再线性投影得到最终 global feature

    这里给一个工程化实现:
    - 先用 PRGAT 建模 slow / fast 的关系
    - 再显式算一个 scalar gate
    """
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        hidden_dim = dim if hidden_dim is None else hidden_dim

        self.global_prgat = PRGAT(dim=dim, dropout=dropout)

        self.slow_proj = nn.Linear(dim, hidden_dim)
        self.fast_proj = nn.Linear(dim, hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, h_slow, h_fast):
        # 先做图交互
        nodes = torch.stack([h_slow, h_fast], dim=1)
        h_global_graph, alpha_global, refined_global = self.global_prgat(nodes)

        # 再显式 gate
        s = self.slow_proj(h_slow)
        f = self.fast_proj(h_fast)

        gate_fast = torch.sigmoid(self.gate_net(torch.cat([s, f], dim=-1)))  # (B, 1)
        mixed = gate_fast * f + (1.0 - gate_fast) * s

        # 与 graph feature 融合
        mixed = mixed + self.slow_proj(h_global_graph)
        u = self.out_proj(mixed)

        aux = {
            "alpha_global": alpha_global,    # [slow, fast]
            "refined_global": refined_global,
            "gate_fast": gate_fast,
            "gate_slow": 1.0 - gate_fast,
        }
        return u, aux


# =========================================================
# 8. HPAF Backbone
# =========================================================
class HPAFModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        embed_dim: int = 256,
        final_dim: int = 256,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output

        # 每个 phase 一个独立 MVFE
        self.mvfe_p = MVFE(in_ch=in_ch, embed_dim=embed_dim, dropout=dropout)
        self.mvfe_qrs = MVFE(in_ch=in_ch, embed_dim=embed_dim, dropout=dropout)
        self.mvfe_st = MVFE(in_ch=in_ch, embed_dim=embed_dim, dropout=dropout)
        self.mvfe_tu = MVFE(in_ch=in_ch, embed_dim=embed_dim, dropout=dropout)

        self.pghf = PGHF(dim=embed_dim, dropout=dropout)
        self.grf = GRF(dim=embed_dim, out_dim=final_dim, dropout=dropout)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        p,qrs,st,tu = _ensure_3d(x['p']),_ensure_3d(x['qrs']),_ensure_3d(x['st']),_ensure_3d(x['tu'])
        h_p, aux_p = self.mvfe_p(p)
        h_qrs, aux_qrs = self.mvfe_qrs(qrs)
        h_st, aux_st = self.mvfe_st(st)
        h_tu, aux_tu = self.mvfe_tu(tu)

        h_slow, h_fast, aux_group = self.pghf(h_p, h_qrs, h_st, h_tu)
        u, aux_global = self.grf(h_slow, h_fast)

        if self.normalize_output:
            u = F.normalize(u, dim=-1)

        return {
            "embedding": u,          # 最终 beat-level embedding
            # "h_p": h_p,
            # "h_qrs": h_qrs,
            # "h_st": h_st,
            # "h_tu": h_tu,
            # "h_slow": h_slow,
            # "h_fast": h_fast,
            # "aux": {
            #     "phase": {
            #         "p": aux_p,
            #         "qrs": aux_qrs,
            #         "st": aux_st,
            #         "tu": aux_tu,
            #     },
            #     "group": aux_group,
            #     "global": aux_global,
            # }
        }

if __name__ == "__main__":
    model = HPAFModel()
    ## 参数量
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")