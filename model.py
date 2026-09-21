import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

# ==================================================================
# [创新点 1] 物理先验层 (Physics-Informed Color Deconvolution)
# 作用：在网络最前端，基于比尔-朗伯定律分离黑色素和血红蛋白
# ==================================================================
class ColorDeconvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 经典的 Ruifrok & Johnston 分离矩阵
        # 用于分离 Hematoxylin (近似黑色素) 和 Eosin (近似血红蛋白)
        full_matrix = np.array([
            [0.3453, 0.6408, 0.6857], # Component 1 (e.g., Melanin-dominant)
            [0.8086, 0.4906, 0.3248], # Component 2 (e.g., Hemoglobin-dominant)
            [0.3000, 0.1506, 0.9420]  # Component 3 (Residual/Background)
        ])
        # 计算逆矩阵以进行反卷积
        inv_matrix = np.linalg.pinv(full_matrix)
        
        # 我们只取前两个最有用的物理通道
        # 权重形状: [Out_C, In_C, 1, 1] => [2, 3, 1, 1]
        weight = torch.from_numpy(inv_matrix[:, :2].T).float().view(2, 3, 1, 1)
        
        self.conv = nn.Conv2d(3, 2, 1, bias=False)
        self.conv.weight.data = weight
        # [关键] 冻结参数，因为这是物理定律，不需要训练
        self.conv.weight.requires_grad = False 

    def forward(self, x):
        # x 必须是 0-1 之间的原始强度值
        # 1. 避免 log(0)
        x = torch.clamp(x, min=1e-6)
        # 2. 光密度变换 (Optical Density): OD = -log(I)
        x_od = -torch.log(x)
        # 3. 颜色反卷积 (1x1 Conv)
        stains = self.conv(x_od)
        return stains

# ==================================================================
# [创新点 2] 动态频域门控融合 (DFG-Fusion)
# 作用：在解码阶段，利用频域滤除浅层特征中的毛发噪声
# ==================================================================
class DFGFusion(nn.Module):
    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
            
        # 1. 空间融合分支 (保留 Scale-Aware 逻辑)
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, padding=0)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 3, padding=1),
            # Softmax 输出两个权重图
        )
        
        # 2. 频域门控参数 (可学习的全局滤波器)
        # 针对每个通道学习一个频域权重，抑制高频噪声
        self.freq_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

    def forward(self, x_low, x_high):
        # x_low: 浅层特征 (高分辨率, 含噪声)
        # x_high: 深层特征 (低分辨率, 语义强)
        
        # 1. 尺寸对齐
        if x_high.size()[2:] != x_low.size()[2:]:
            x_high = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
            
        # ================= [频域去噪核心] =================
        B, C, H, W = x_low.shape
        # 仅处理足够大的特征图
        if H >= 16 and W >= 16:
            # FFT 变换
            x_low_fft = torch.fft.rfft2(x_low, norm='ortho')
            
            # 门控滤波 (Sigmoid 归一化)
            gate = torch.sigmoid(self.freq_weight)
            x_low_fft = x_low_fft * gate
            
            # IFFT 还原
            x_low_filtered = torch.fft.irfft2(x_low_fft, s=(H, W), norm='ortho')
            
            # 残差连接：叠加去噪后的特征
            x_low = x_low + 0.1 * x_low_filtered
        # =================================================

        # 2. 空间融合
        feat = torch.cat([x_low, x_high], dim=1)
        feat = self.conv1x1(feat)
        
        att_map = self.spatial_gate(feat)
        att_map = F.softmax(att_map, dim=1)
        
        att_low = att_map[:, 0:1, :, :]
        att_high = att_map[:, 1:2, :, :]
        
        # 3. 加权输出
        out = x_low * att_low + x_high * att_high
        return out

# ==================================================================
# [组件] CBAM Attention (Skip Connection 标配)
# ==================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# ==================================================================
# 基础卷积模块
# ==================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(x)

class DecoderBlock_V11(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_attn = CBAM(skip_ch)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_attn(skip)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ==================================================================
# [CAS-UNet V11 Ultimate] 三位一体网络
# Physics (Input) + Frequency (Decoder) + Spatial (Skip)
# ==================================================================
class CAS_UNet_V11(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_ch=32):
        super().__init__()
        
        # 1. 物理先验层 (Physics Prior)
        self.physic_prior = ColorDeconvLayer()
        
        # 2. Encoder
        # [修改] 输入通道 = RGB(3) + 物理通道(2) = 5
        self.init_conv = ConvBlock(in_channels + 2, base_ch)
        
        self.enc1 = EncoderBlock(base_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = EncoderBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 3. Bridge (ASPP Style)
        self.bridge = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, 3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True)
        )
        
        # 4. Decoder (with CBAM)
        self.dec1 = DecoderBlock_V11(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec2 = DecoderBlock_V11(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec3 = DecoderBlock_V11(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec4 = DecoderBlock_V11(base_ch * 2, base_ch, base_ch)
        
        # 5. Projection Heads
        self.head1 = nn.Conv2d(base_ch * 8, 64, 1)
        self.head2 = nn.Conv2d(base_ch * 4, 64, 1)
        self.head3 = nn.Conv2d(base_ch * 2, 64, 1)
        self.head4 = nn.Conv2d(base_ch, 64, 1)
        
        # 6. DFG-Fusion (Frequency Awareness)
        self.saf1 = DFGFusion(64) 
        self.saf2 = DFGFusion(64)
        self.saf3 = DFGFusion(64)
        
        # 7. Final & Aux Heads
        self.final_head = nn.Conv2d(64, num_classes, 1)
        self.aux_head1 = nn.Conv2d(64, num_classes, 1)
        self.aux_head2 = nn.Conv2d(64, num_classes, 1)
        self.aux_head3 = nn.Conv2d(64, num_classes, 1)
        
        # 8. 反归一化参数 (用于恢复 RGB 进行物理计算)
        # 假设你 dataset 里用的是 ImageNet mean/std
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # [Step 1] 物理先验提取
        # 反归一化到 0~1
        x_raw = x * self.std + self.mean
        x_raw = torch.clamp(x_raw, 1e-6, 1.0) # 物理公式要求正数
        
        # 计算颜色反卷积特征 (2通道)
        physic_feat = self.physic_prior(x_raw)
        
        # 早期融合: RGB(3) + Physic(2) -> 5通道
        x_in = torch.cat([x, physic_feat], dim=1)
        
        # [Step 2] Encoder
        x0 = self.init_conv(x_in)
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # [Step 3] Bridge
        b = self.bridge(p4)
        
        # [Step 4] Decoder
        d1 = self.dec1(b, e4)
        d2 = self.dec2(d1, e3)
        d3 = self.dec3(d2, e2)
        d4 = self.dec4(d3, e1)
        
        # [Step 5] Projection
        h1 = self.head1(d1)
        h2 = self.head2(d2)
        h3 = self.head3(d3)
        h4 = self.head4(d4)
        
        # [Step 6] DFG-Fusion (频域去噪融合)
        f_1_2 = self.saf1(x_low=h2, x_high=h1)
        f_2_3 = self.saf2(x_low=h3, x_high=f_1_2)
        f_3_4 = self.saf3(x_low=h4, x_high=f_2_3)
        
        main_out = self.final_head(f_3_4)
        
        if self.training:
            return {
                'main_seg': main_out,
                'aux1': self.aux_head1(h1),
                'aux2': self.aux_head2(h2),
                'aux3': self.aux_head3(h3)
            }
        else:
            return main_out
