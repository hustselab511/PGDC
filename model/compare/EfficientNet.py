import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, reduced_dim)
        self.swish = Swish()
        self.fc2 = nn.Linear(reduced_dim, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.swish(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return module_input * x.view(x.size(0), x.size(1), 1)

class MBConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, dropout_rate=0.2):
        super(MBConvBlock1D, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expand_channels = in_channels * expand_ratio
        
        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, expand_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(expand_channels),
            Swish()
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(expand_channels, expand_channels, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size//2, groups=expand_channels, bias=False),
            nn.BatchNorm1d(expand_channels),
            Swish()
        )

        reduced_dim = max(1, int(in_channels * se_ratio))
        self.se = SEBlock1D(expand_channels, reduced_dim)

        self.project_conv = nn.Sequential(
            nn.Conv1d(expand_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.dropout = nn.Dropout(p=dropout_rate) if self.use_residual else nn.Identity() # [cite: 158, 180]

    def forward(self, x):
        res = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = self.dropout(x)
            x = x + res
        return x

class EfficientNet1D(nn.Module):
    def __init__(self, sigma_d=1.0, sigma_w=1.0, num_classes=100):
        super(EfficientNet1D, self).__init__()
        
        self.sigma_d = sigma_d
        self.sigma_w = sigma_w
        
        def scale_w(w):
            return int(w * self.sigma_w)

        def scale_d(l):
            return int(torch.ceil(torch.tensor(l * self.sigma_d)).item())

        self.stage1 = nn.Sequential(
            nn.Conv1d(1, scale_w(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(scale_w(32)),
            Swish()
        )

        configs = [
            (1, 3, 16, 1, 1), # Stage 2
            (6, 3, 24, 2, 2), # Stage 3
            (6, 5, 40, 2, 2), # Stage 4
            (6, 3, 80, 3, 2), # Stage 5
            (6, 5, 112, 3, 1), # Stage 6
            (6, 5, 192, 4, 2), # Stage 7
            (6, 3, 320, 1, 1), # Stage 8
        ]

        layers = []
        in_ch = scale_w(32)
        for n, k, c, l, s in configs:
            out_ch = scale_w(c)
            num_layers = scale_d(l)
            for i in range(num_layers):
                stride = s if i == 0 else 1
                layers.append(MBConvBlock1D(in_ch, out_ch, k, stride, n))
                in_ch = out_ch
        self.mbconv_blocks = nn.Sequential(*layers)

        self.final_conv = nn.Sequential(
            nn.Conv1d(in_ch, scale_w(1280), kernel_size=1, bias=False),
            nn.BatchNorm1d(scale_w(1280)),
            Swish()
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(scale_w(1280), num_classes)
    def modify_classifier(self, num_classes):
        # for param in self.parameters():
        #     param.requires_grad = False
        #     param.grad = None
        self.fc = nn.Linear(int(self.sigma_w * 1280), num_classes)
        # for param in self.fc.parameters():
        #     param.requires_grad = True
        # for param in self.final_conv.parameters():
        #     param.requires_grad = True
    def forward(self, x):
        # 输入形状为 (Batch, 1, N*L)
        x = self.stage1(x)
        x = self.mbconv_blocks(x)
        x = self.final_conv(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

class IntegratedEfficientNet1D(nn.Module):
    def __init__(self, model_list):
        super(IntegratedEfficientNet1D, self).__init__()
        # 初始化三个具有不同缩放因子的子模型 [cite: 215]
        self.model_list = model_list

    def forward(self, x):
        # 此方法仅用于训练
        return [model(x) for model in self.model_list]

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = [torch.argmax(model(x), dim=1) for model in self.model_list]
            combined = torch.stack(outputs, dim=1)
            final_pred, _ = torch.mode(combined, dim=1)
            return final_pred
    
    def get_logit(self, x):
        self.eval()
        with torch.no_grad():
            all_logits = [model(x) for model in self.model_list]
            avg_logit = torch.mean(torch.stack(all_logits), dim=0)
            return F.softmax(avg_logit, dim=1)