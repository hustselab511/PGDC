from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

def _to_2d_grid(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    B, L, C = x.shape
    H, W = hw
    assert L == H * W, f"Token length {L} != H*W {H*W}"
    return x.transpose(1, 2).contiguous().view(B, C, H, W)

def _from_2d_grid(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, H, W) -> (B, L, C)
    """
    B, C, H, W = x.shape
    return x.view(B, C, H * W).transpose(1, 2).contiguous()

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: (B, H, W, C) -> windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H/W must be divisible by window_size"
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    windows: (num_windows*B, window_size, window_size, C) -> x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)

        if attn_mask is not None:
            # attn_mask: (nW, N, N); B_ = batch*nW
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj_drop(self.proj(out))
        return out


def build_shifted_window_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> torch.Tensor:
    """
    Standard Swin trick: build attention mask for SW-MSA.
    Returns mask with shape (nW, N, N), values in {0, -inf}.
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)
    cnt = 0
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class SwinBlock(nn.Module):
    """
    One Swin block. Use shift_size=0 for W-MSA and shift_size=window_size//2 for SW-MSA.
    """
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

        self._attn_mask_cache = {}  # (H,W,device)->mask

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        x: (B, L, C) with L=H*W
        """
        H, W = hw
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x_grid = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x_grid = torch.roll(x_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            key = (H, W, str(x.device))
            if key not in self._attn_mask_cache:
                self._attn_mask_cache[key] = build_shifted_window_mask(H, W, self.window_size, self.shift_size, x.device)
            attn_mask = self._attn_mask_cache[key]
        else:
            attn_mask = None

        # partition windows
        x_windows = window_partition(x_grid, self.window_size)  # (B*nW, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse windows
        x_grid = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x_grid = torch.roll(x_grid, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x_grid.view(B, H * W, C)
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """
    2x2 patch merging: (H,W,C) -> (H/2,W/2,2C) via concat + linear.
    """
    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim * 2
        self.reduction = nn.Linear(dim * 4, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(dim * 4)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x: (B, H*W, C)
        """
        H, W = hw
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, (H // 2) * (W // 2), 4 * C)
        x = self.reduction(self.norm(x))
        return x, (H // 2, W // 2)


# ----------------------------
# CNN branch blocks
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


# ----------------------------
# Fusion block (paper-aligned logic)
# ----------------------------

class FusionBlock(nn.Module):
    def __init__(self, dim: int, is_final: bool = False):
        super().__init__()
        self.is_final = is_final
        self.fuse_op = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(inplace=True)
        )
        
        if is_final:
            self.final_reduce = nn.Conv2d(2 * dim, dim, kernel_size=1)
        else:
            self.to_cnn = nn.Conv2d(dim, dim, kernel_size=1)
            self.to_tr = nn.Linear(dim, dim)

    def forward(self, Ti, Ci, hw):
        H, W = hw
        Ti_2d = Ti.transpose(1, 2).view(Ti.shape[0], -1, H, W)
        
        fused = torch.cat([Ti_2d, Ci], dim=1)
        fused = self.fuse_op(fused)
        
        if self.is_final:
            fused = self.final_reduce(fused)
            return fused.flatten(2).transpose(1, 2), None
            
        T_part, C_part = torch.split(fused, fused.shape[1]//2, dim=1)
        Ci_out = Ci + self.to_cnn(C_part)
        Ti_out = Ti + self.to_tr(T_part.flatten(2).transpose(1, 2))
        
        return Ti_out, Ci_out


# ----------------------------
# CESTNet
# ----------------------------

@dataclass
class CESTNetConfig:
    num_classes: int
    embed_dim: int = 128
    window_size: int = 8          # stage1 默认8
    patch_len: int = 25
    ecg_pad_len: int = 1600
    stage_pairs: Tuple[int, int, int, int] = (1, 1, 3, 1)
    num_heads: Tuple[int, int, int, int] = (4, 8, 8, 8)
    drop: float = 0.0
    window_sizes: Tuple[int, int, int, int] = (8, 4, 2, 1)



class CESTNet(nn.Module):
    def __init__(self, cfg: CESTNetConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Linear(cfg.patch_len, cfg.embed_dim)
        d1 = cfg.embed_dim
        d2 = d1 * 2
        d3 = d2 * 2
        d4 = d2
        self.tr_dims = (d1, d2, d3, d4)
        ws1, ws2, ws3, ws4 = cfg.window_sizes

        self.stage1 = self._make_swin_stage(dim=d1, heads=cfg.num_heads[0], pairs=cfg.stage_pairs[0], window=ws1, drop=cfg.drop)
        self.stage2 = self._make_swin_stage(dim=d2, heads=cfg.num_heads[1], pairs=cfg.stage_pairs[1], window=ws2, drop=cfg.drop)
        self.stage3 = self._make_swin_stage(dim=d3, heads=cfg.num_heads[2], pairs=cfg.stage_pairs[2], window=ws3, drop=cfg.drop)
        self.stage4 = self._make_swin_stage(dim=d4, heads=cfg.num_heads[3], pairs=cfg.stage_pairs[3], window=ws4, drop=cfg.drop)

        self.merge1 = PatchMerging(d1, out_dim=d2)
        self.merge2 = PatchMerging(d2, out_dim=d3)
        self.merge3 = PatchMerging(d3, out_dim=d4)

        self.cnn_stem = self._make_cnn_stem(out_ch=d1)
        self.cnn_l2 = ConvBlock(d1, d2, stride=2)  # 8x8 -> 4x4
        self.cnn_l3 = ConvBlock(d2, d3, stride=2)  # 4x4 -> 2x2

        self.fuse1 = FusionBlock(dim=d1, is_final=False)  # at 8x8
        self.fuse2 = FusionBlock(dim=d2, is_final=False)  # at 4x4
        self.fuse3_final = FusionBlock(dim=d3, is_final=True)  # at 2x2, reduces 2*d3 -> d3 then sent to transformer

        self.norm = nn.LayerNorm(d4)
        self.d4 = d4
        self.head = nn.Linear(d4, cfg.num_classes)

    def modify_head(self, num_classes: int):
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None
        self.head = nn.Linear(self.d4, num_classes)
        for param in self.head.parameters():
            param.requires_grad = True
        for param in self.stage4.parameters():
            param.requires_grad = True
        for param in self.merge3.parameters():
            param.requires_grad = True
    @staticmethod
    def _make_swin_stage(dim: int, heads: int, pairs: int, window: int, drop: float) -> nn.ModuleList:
        blocks = []
        # each pair: [W-MSA, SW-MSA]
        for _ in range(pairs):
            blocks.append(SwinBlock(dim, heads, window_size=window, shift_size=0, drop=drop))
            shift = 0 if window == 1 else window // 2
            blocks.append(SwinBlock(dim, heads, window_size=window, shift_size=shift, drop=drop))
        return nn.ModuleList(blocks)

    @staticmethod
    def _make_cnn_stem(out_ch: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_ch = 1
        for i in range(5):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _ecg_to_tokens(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        ecg: (B,1000) or (B,1,1000) -> (B,64,C)
        """
        if ecg.dim() == 3:
            ecg = ecg.squeeze(1)
        B, L = ecg.shape
        assert L == 1000, f"Expected 1000 points heartbeat, got {L}"
        # zero-pad to 1600 :contentReference[oaicite:12]{index=12}
        if self.cfg.ecg_pad_len > L:
            ecg = F.pad(ecg, (0, self.cfg.ecg_pad_len - L))
        elif self.cfg.ecg_pad_len < L:
            ecg = ecg[:, : self.cfg.ecg_pad_len]

        # patch partition with patch_len=25 -> 64 patches :contentReference[oaicite:13]{index=13}
        P = self.cfg.patch_len
        assert ecg.shape[1] % P == 0
        tokens = ecg.view(B, ecg.shape[1] // P, P)  # (B,64,25)
        tokens = self.token_embed(tokens)            # (B,64,C)
        return tokens

    def _run_stage(self, blocks: nn.ModuleList, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        for blk in blocks:
            x = blk(x, hw)
        return x

    def forward(self, ecg: torch.Tensor, prem: torch.Tensor) -> torch.Tensor:
        """
        ecg:  (B,1000) or (B,1,1000)
        prem: (B,1,256,256)
        """
        B = prem.shape[0]
        assert prem.shape[1:] == (1, 256, 256), f"Expected prem (B,1,256,256), got {prem.shape}"

        T = self._ecg_to_tokens(ecg)        # (B,64,d1)
        hw1 = (8, 8)                        # 64 tokens -> 8x8 :contentReference[oaicite:14]{index=14}
        C = self.cnn_stem(prem)             # (B,d1,8,8)

        T = self._run_stage(self.stage1, T, hw1)
        T, C = self.fuse1(T, C, hw1)

        T, hw2 = self.merge1(T, hw1)        # (B,16,d2), (4,4)
        T = self._run_stage(self.stage2, T, hw2)
        C = self.cnn_l2(C)                  # (B,d2,4,4)
        T, C = self.fuse2(T, C, hw2)

        T, hw3 = self.merge2(T, hw2)        # (B,4,d3), (2,2)
        T = self._run_stage(self.stage3, T, hw3)
        C = self.cnn_l3(C)                  # (B,d3,2,2)

        T, _ = self.fuse3_final(T, C, hw3)  # (B,4,d3)

        T, hw4 = self.merge3(T, hw3)        # (B,1,d4), (1,1)
        T = self._run_stage(self.stage4, T, hw4)

        feat = self.norm(T[:, 0, :])        # (B,d4)
        logits = self.head(feat)            # (B,num_classes)
        return logits


# ----------------------------
# Quick sanity test
# ----------------------------
if __name__ == "__main__":
    cfg = CESTNetConfig(num_classes=90)  # e.g., ECG-ID has 90 subjects
    model = CESTNet(cfg)
    ecg = torch.randn(2, 1000)
    prem = torch.randn(2, 1, 256, 256)
    out = model(ecg, prem)
    print(out.shape)  # (2, 90)
