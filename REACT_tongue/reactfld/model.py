"""REACT-FLD 模型(论文版 §4-§13,v5 改进版)。

v5 = v4 全部改进(tokens 直连破概念瓶颈 + 年龄/性别逐通道调制 + log 舌体面积特征
+ 窄瓶颈融合头 + 色度增广 + EMA)+ 方案③ 多尺度视觉编码:
  ConvNeXt stage2/3/4 三尺度特征拼接(256@28)+ TIPNet 式 SelfAtten 注意力。
诊断依据:v2-v4 舌象通路稳定在 AUC 0.75-0.77,而 TIPNet 纯图像分支 0.845,
差距在 encoder 侧;SelfAtten 消融(TIPNet_9_Indicators 0.916 vs NoneAtten 0.858)。

视觉编码: ConvNeXt-Tiny → 1×1 卷积 → 256 维特征图(无背景抑制,无 GroupNorm/GELU 投影)
全局 token: q0 = z(M_T);区域 token: q_r = z(M_r) + MLP_box(B_r) + e_r
区域内交叉注意力 + 区域间 Transformer → 6 tokens → 18 概念头 → Ĉ ∈ R^50
生理通路: FT-Transformer;融合: Dirichlet 证据头 + 门控 + 跨模态交互头
单模型,无深度集成(论文版)

v3 改进(基于 v2 瓶颈诊断:舌象通路 AUC 0.664 vs TIPNet 纯图像 0.845,概念瓶颈截断信息):
  ① 破概念瓶颈:z_T = g_T([Ĉ, v, r_mean]),融合头直连调制后的全局/区域 token,
    概念头退化为可解释性分支;FLD 监督经 token 直达视觉编码器
  ② TIPNet 式人口学调制:2σ(MLP([Age,性别])) 逐通道作用于风险通路的视觉 token
    (概念头保持纯视觉)
  ④ log(舌体面积占比) 直连融合头(TIPNet 的 mask_rate 思想)
"""
from __future__ import annotations
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------- v5 视觉编码:多尺度 + SelfAtten(方案③)
class StdPool(nn.Module):
    def forward(self, x):
        b, c, _, _ = x.size()
        return x.view(b, c, -1).std(dim=2).reshape(b, c, 1, 1)


class ChannelGate(nn.Module):
    """通道门:avg+std 双池化,可学习融合权重,沿通道维 1D 卷积(移植自 TIPNet)。"""

    def __init__(self, gate_channels: int, k_size: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(2))
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1), StdPool()])
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1,
                              padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        w = torch.sigmoid(self.weight)
        out = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        return self.sigmoid(out)


class SpatialGate(nn.Module):
    """空间门:通道维 avg+std,可学习融合,3×3 卷积 + 平滑(移植自 TIPNet)。"""

    def __init__(self, k_size: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(2))
        self.conv = nn.Sequential(nn.Conv2d(1, 1, k_size, 1, padding=(k_size - 1) // 2, bias=False),
                                  nn.AvgPool2d(3, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg, std = x.mean(1, keepdim=True), x.std(1, keepdim=True)
        w = torch.sigmoid(self.weight)
        out = (0.5 + w[0]) * avg + (0.5 + w[1]) * std
        return self.sigmoid(self.conv(out))


class SelfAtten(nn.Module):
    """TIPNet SelfAtten:scale = ⅓·通道门 + ⅔·空间门,逐元素重标定。"""

    def __init__(self, gate_channels: int, k_size: int = 3):
        super().__init__()
        self.channel = ChannelGate(gate_channels, k_size)
        self.spatial = SpatialGate(k_size)

    def forward(self, x):
        return (self.channel(x) / 3.0 + 2.0 * self.spatial(x) / 3.0) * x


class MultiScaleVisualEncoder(nn.Module):
    """v5:ConvNeXt stage2(192@28)/stage3(384@14)/stage4(768@7) 三尺度特征
    各自 1×1 投影到 256 维、上采样到 28² 拼接(768 通道),过 SelfAtten 后
    输出 256@28 特征图(v4 为 256@7,区域掩码池化粒度提升 16 倍)。"""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        mc = config["model"]
        self.dim = int(mc["feature_dim"])
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if mc.get("pretrained", True) else None
        self.backbone = convnext_tiny(weights=weights).features
        self.proj2 = nn.Conv2d(192, self.dim, 1)
        self.proj3 = nn.Conv2d(384, self.dim, 1)
        self.proj4 = nn.Conv2d(768, self.dim, 1)
        self.self_atten = SelfAtten(self.dim * 3)
        self.out_proj = nn.Conv2d(self.dim * 3, self.dim, 1)

    def forward(self, image: torch.Tensor, body_mask: torch.Tensor | None = None) -> torch.Tensor:
        feats, t = [], image
        for j, block in enumerate(self.backbone):
            t = block(t)
            if j in (3, 5, 7):
                feats.append(t)
        p2 = self.proj2(feats[0])
        p3 = F.interpolate(self.proj3(feats[1]), size=p2.shape[-2:],
                           mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(feats[2]), size=p2.shape[-2:],
                           mode="bilinear", align_corners=False)
        return self.out_proj(self.self_atten(torch.cat([p2, p3, p4], 1)))


# --------------------------------------------- v10 视觉编码:TIPNet 三并行 ResNet(移植)
class BasicBlockK(nn.Module):
    """TIPNet BasicBlock:两个 k×k 卷积,块内 SelfAtten(BN 后、残差相加前)。"""
    expansion = 1

    def __init__(self, k: int, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        pad = {3: 1, 5: 2, 7: 3}[k]
        self.conv1 = nn.Conv2d(inplanes, planes, k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.attention = SelfAtten(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + identity)


class BottleneckK(nn.Module):
    """TIPNet Bottleneck:1×1 → k×k(stride) → 1×1×4exp,块内 SelfAtten。"""
    expansion = 4

    def __init__(self, k: int, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        pad = {3: 1, 5: 2, 7: 3}[k]
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, k, stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.attention = SelfAtten(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.attention(out)
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + identity)


class ResNetK(nn.Module):
    """TIPNet 单支 ResNet(kernel∈{3,5,7});返回 layer2 特征图 512@28(已乘背景掩码)。"""

    def __init__(self, kernel_size: int, layers: list[int]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlockK, kernel_size, 64, layers[0])
        self.layer2 = self._make_layer(BottleneckK, kernel_size, 128, layers[1], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, kernel_size: int, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        seq = [block(kernel_size, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            seq.append(block(kernel_size, self.inplanes, planes))
        return nn.Sequential(*seq)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        return self.layer2(x) * mask


class TIPNetVisualEncoder(nn.Module):
    """v10:移植 TIPNet 图像分支(三并行 ResNet 3/5/7 + 块内 SelfAtten)。
    忠实配方:输入 = 原图×舌体掩码(黑背景);内部按原式绿通道阈值(-1.7)+MaxPool(8)
    导出 28² 掩码做 layer2 背景抑制;三支 512@28 拼接 → 1×1 → 256@28。
    (TIPNet_9 同数据复现 AUC 0.845,阶段 A 用其 SGD 配方从零单任务训练。)"""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        mc = config["model"]
        self.dim = int(mc["feature_dim"])
        self.base_1 = ResNetK(3, [1, 2])
        self.base_2 = ResNetK(5, [1, 2])
        self.base_3 = ResNetK(7, [1, 2])
        self.out_proj = nn.Conv2d(512 * 3, self.dim, 1)

    def forward(self, image: torch.Tensor, body_mask: torch.Tensor) -> torch.Tensor:
        x = image * body_mask
        mask0 = torch.where(x[:, 1:2] > -1.7, 1.0, 0.0)
        mask1 = F.max_pool2d(mask0, 8, 8)
        f = torch.cat([self.base_1(x, mask1), self.base_2(x, mask1), self.base_3(x, mask1)], 1)
        return self.out_proj(f)


class TDFGapHead(nn.Module):
    """v11:TIPNet_9_Indicators 同款证据头(其单任务复现 AUC 0.916)。
    路线:GAP 视觉特征 × mask_rate × 2σ(embed(Age,性别one-hot)) 逐通道门控
    + 6 项 z-score 指标(Height/Waist/Hipline/Weight/SBP/DBP)MLP 6→512→512,
    拼接 → Dropout → 线性 → gap_logit。embed 末层零初始化 → 初始门控恒等(原实现细节)。"""

    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        mid = 512
        self.embed = nn.Sequential(
            nn.Linear(3, mid), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mid, dim), nn.Sigmoid())
        self.embed[3].weight.data.zero_()
        self.embed[3].bias.data.zero_()
        self.embed_concat = nn.Sequential(
            nn.Linear(6, mid), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mid, mid), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim + mid, 1))

    def forward(self, gap_feat: torch.Tensor, mask_rate: torch.Tensor,
                physiology: torch.Tensor) -> torch.Tensor:
        demo = ReactFLD.demo_input(physiology)                 # [Age_z, 性别one-hot(2)]
        mr = mask_rate.reshape(gap_feat.shape[0], 1)           # 兼容 (B,) / (B,1)
        xi = gap_feat * mr * (2.0 * self.embed(demo))          # 视觉×面积率×人口学门控
        xc = self.embed_concat(physiology[:, 2:8])             # 6 项 z-score 指标
        return self.head(torch.cat([xi, xc], -1)).squeeze(-1)


class EvidenceFusedHead(nn.Module):
    """v12:证据拼接 + 单一决策权。生理指标与概念概率只是证据 embedding,
    拼接后由唯一的 FLD 分类器决定采信;final = gap_logit + corr(证据),
    corr 末层零初始化 → 训练起点精确等于阶段A单任务性能,学习只能加分。"""

    def __init__(self, dim: int, concept_dim: int, physiology_dim: int, dropout: float = 0.2):
        super().__init__()
        self.phys_mlp = nn.Sequential(nn.Linear(physiology_dim, 32), nn.GELU())
        self.concept_mlp = nn.Sequential(nn.Linear(concept_dim, 32), nn.GELU())
        corr_in = dim + 1 + 32 + 32
        self.corr = nn.Sequential(
            nn.LayerNorm(corr_in), nn.Linear(corr_in, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1))
        nn.init.zeros_(self.corr[-1].weight)
        nn.init.zeros_(self.corr[-1].bias)

    def forward(self, gap_feat: torch.Tensor, gap_logit: torch.Tensor,
                physiology: torch.Tensor, concept_probs: torch.Tensor) -> torch.Tensor:
        evidence = torch.cat([gap_feat, gap_logit[:, None],
                              self.phys_mlp(physiology.to(gap_feat.dtype)),
                              self.concept_mlp(concept_probs.to(gap_feat.dtype))], -1)
        return gap_logit + self.corr(evidence).squeeze(-1)


def mask_pool(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """掩码池化: z = Σ M⊙F / ΣM(论文版 §5.1)。"""
    m = F.interpolate(mask, feats.shape[-2:], mode="nearest")
    return (feats * m).sum((-2, -1)) / m.sum((-2, -1)).clamp_min(1.0)


# ------------------------------------------------------- 概念头(论文版 §8.2)
class BinaryHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

    def probs_from_logits(self, logits):
        p1 = torch.sigmoid(logits)
        return torch.stack([1 - p1, p1], -1)


class OrdinalHead(nn.Module):
    """CORAL 有序回归: P(y>k)=sigmoid(w·x − b_k),b 单调。"""
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Linear(dim, 1, bias=False)
        self.raw_bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        biases = torch.cumsum(F.softplus(self.raw_bias), 0)
        return self.weight(x) - biases

    def probs_from_logits(self, cum):
        cum = torch.sigmoid(cum)
        ones = torch.ones_like(cum[:, :1])
        zeros = torch.zeros_like(cum[:, :1])
        bounds = torch.cat([ones, cum, zeros], -1)
        return (bounds[:, :-1] - bounds[:, 1:]).clamp_min(0)


class MulticlassHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    def probs_from_logits(self, logits):
        return F.softmax(logits, -1)


def make_head(dim: int, spec: dict) -> nn.Module:
    kind, k = spec["type"], len(spec["classes"])
    if kind == "binary":
        return BinaryHead(dim)
    if kind == "ordinal":
        return OrdinalHead(dim, k)
    return MulticlassHead(dim, k)


# ---------------------------------------------------- 生理编码器(论文版 §10)
class FTTransformerEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], num_features: int):
        super().__init__()
        mc = config["model"]
        dim = int(mc["phys_dim"])
        self.dim = dim
        self.gender_embedding = nn.Embedding(2, dim)
        self.feature_weights = nn.Parameter(torch.randn(num_features - 1, dim) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(num_features - 1, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, int(mc["transformer_heads"]), dim * 4,
                                           float(mc["dropout"]), batch_first=True, activation="gelu",
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, int(mc["phys_layers"]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        gender_id = (x[:, 0] > 0).long().clamp(0, 1)
        gender = self.gender_embedding(gender_id).unsqueeze(1)
        numerical = x[:, 1:]
        tokens = numerical.unsqueeze(-1) * self.feature_weights.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens * (1.0 - missing[:, 1:].unsqueeze(-1))
        tokens = torch.cat([gender, tokens], 1)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], 1)
        return self.norm(self.blocks(tokens)[:, 0])


# --------------------------------------------------------------- 不确定性(论文版 §11.4)
class EvidentialHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 2)

    def forward(self, x):
        # 退出 autocast,强制 fp32 计算,避免 fp16 溢出与除法 NaN
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_fp32 = x.float()
            w_fp32 = self.fc.weight.float()
            b_fp32 = self.fc.bias.float() if self.fc.bias is not None else None
            logits = F.linear(x_fp32, w_fp32, b_fp32)

            alpha = F.softplus(logits) + 1.0
            alpha = alpha.clamp(max=1e4)

            probs = alpha / alpha.sum(-1, keepdim=True)
            uncertainty = 2.0 / alpha.sum(-1)

        return probs[:, 1], uncertainty, alpha


# ---------------------------------------------------------------------- 主模型
class ReactFLD(nn.Module):
    def __init__(self, config: dict[str, Any], physiology_dim: int = 11,
                 global_specs: list[dict] | None = None, regional_specs: list[dict] | None = None):
        super().__init__()
        from .concepts import GLOBAL_CONCEPTS, REGIONAL_CONCEPTS, concept_vector_dim
        mc, dc = config["model"], config["data"]
        self.regions = list(dc["regions"])
        self.region_index = {r: i for i, r in enumerate(self.regions)}
        self.global_specs = global_specs or GLOBAL_CONCEPTS
        self.regional_specs = regional_specs or REGIONAL_CONCEPTS
        self.spec_by_key = {c["key"]: c for c in self.global_specs + self.regional_specs}
        d = int(mc["feature_dim"])
        self.dim = d

        # v10:encoder 可选 tipnet(三并行 ResNet 3/5/7,单任务可到 0.845)或 convnext(v5 多尺度)
        if str(mc.get("encoder", "convnext")) == "tipnet":
            self.encoder = TIPNetVisualEncoder(config)
        else:
            self.encoder = MultiScaleVisualEncoder(config)
        self.box_encoder = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.region_embedding = nn.Embedding(len(self.regions), d)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d, int(mc["transformer_heads"]), dropout=float(mc["dropout"]), batch_first=True)
            for _ in range(int(mc["cross_attn_layers"]))])
        layer = nn.TransformerEncoderLayer(d, int(mc["transformer_heads"]), d * 4,
                                           float(mc["dropout"]), batch_first=True, activation="gelu",
                                           norm_first=True)
        self.region_transformer = nn.TransformerEncoder(layer, int(mc["transformer_layers"]))
        self.global_heads = nn.ModuleDict({c["key"]: make_head(d, c) for c in self.global_specs})
        self.regional_heads = nn.ModuleDict({c["key"]: make_head(d, c) for c in self.regional_specs})
        self.concept_dim = concept_vector_dim()
        # v3 改进①:tongue 通路绕过概念瓶颈,z_T = g_T([Ĉ, v, r_mean]),FLD 监督直达视觉编码器;
        # 概念头保留全部监督,退化为可解释性分支而非风险信息必经之路
        self.tongue_encoder = nn.Sequential(
            nn.Linear(self.concept_dim + 2 * d, 128), nn.GELU(),
            nn.Dropout(float(mc["dropout"])), nn.Linear(128, 128))
        self.phys_encoder = FTTransformerEncoder(config, physiology_dim)
        self.tongue_evidence = EvidentialHead(128)
        self.phys_evidence = EvidentialHead(128)
        self.interaction_head = nn.Sequential(nn.Linear(128 * 4, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(128 * 2 + 2, 64), nn.GELU(), nn.Linear(64, 3))
        # v3 改进②:TIPNet 式人口学逐通道调制 2σ(MLP([Age,性别one-hot])),作用于风险通路的
        # 视觉 token(概念头仍用未调制 token,概念预测保持纯视觉)
        self.demo_mod = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, d))
        # v3 改进①+④:logit 级融合头。输入 [z_T,z_P,z_T⊙z_P,|z_T−z_P|](512) + physiology(11)
        #   + demo 调制后的全局/区域 token(2×256) + log(舌体面积占比)(1)。
        # 门控/单模态证据头全部保留用于可解释性
        # v6 直连分支(TIPNet 配方):demo 调制 token + log 面积 → MLP → direct_logit。
        # 独立 BCE 监督,给视觉编码器不被概念多任务稀释的 FLD 梯度;
        # 18 个概念头及其监督原样保留为可解释性分支
        self.direct_head = nn.Sequential(
            nn.Linear(2 * d + 1, 128), nn.GELU(),
            nn.Dropout(float(mc["dropout"])), nn.Linear(128, 1))
        # v11:gap 分支 = TIPNet_9_Indicators 同款 TDF 头(视觉×mask_rate×人口学门控
        # + 6 项指标 MLP),阶段 A 单任务复现(其协议 AUC 0.916)后低lr微调;
        # 不再依赖 log_area(mask_rate 即 TIPNet 的面积用法)
        self.gap_head = TDFGapHead(d, dropout=0.5)
        # v12-ens:多 seed 阶段A专家(冻结,纯证据供给):视觉证据 = 各专家
        # gap_logit 的均值 + gap_feat 的均值;主编码器(seed42)仍供 token/概念管线
        n_aux = int(mc.get("ensemble_experts", 0))
        self.ensemble_experts = n_aux
        if n_aux > 0:
            if str(mc.get("encoder", "convnext")) == "tipnet":
                self.aux_encoders = nn.ModuleList(
                    [TIPNetVisualEncoder(config) for _ in range(n_aux)])
            else:
                self.aux_encoders = nn.ModuleList(
                    [MultiScaleVisualEncoder(config) for _ in range(n_aux)])
            self.aux_gap_heads = nn.ModuleList(
                [TDFGapHead(d, dropout=0.5) for _ in range(n_aux)])
            for m in (*self.aux_encoders, *self.aux_gap_heads):
                for p in m.parameters():
                    p.requires_grad = False
        self.use_fusion_head = bool(mc.get("fusion_head", True))
        # v12:证据拼接融合(evidence_fusion),旧 1291 维融合头保留作回退
        self.evidence_fusion = bool(mc.get("evidence_fusion", False))
        if self.use_fusion_head and self.evidence_fusion:
            self.fusion_head = EvidenceFusedHead(d, self.concept_dim, physiology_dim)
        elif self.use_fusion_head:
            fusion_in = 128 * 4 + physiology_dim + 2 * d + 3   # v6:+direct_logit,v8:+gap_logit
            # v4:窄瓶颈融合头(1291→64→1)+ 高 dropout,抑制 v3 暴露的融合侧过拟合
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(fusion_in), nn.Linear(fusion_in, 64), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(64, 1))

    def encode(self, image, body_mask, region_masks, boxes):
        feats = self.encoder(image, body_mask)
        return self.tokens_from_feats(feats, body_mask, region_masks, boxes)

    def tokens_from_feats(self, feats, body_mask, region_masks, boxes):
        B, d, h, w = feats.shape
        global_token = mask_pool(feats, body_mask)
        flat = feats.flatten(2).transpose(1, 2)
        queries = []
        for i, _ in enumerate(self.regions):
            pooled = mask_pool(feats, region_masks[:, i:i + 1])
            q = pooled + self.box_encoder(boxes[:, i]) + self.region_embedding.weight[i]
            m = F.interpolate(region_masks[:, i:i + 1], (h, w), mode="nearest").flatten(1).bool()
            empty = ~m.any(-1)
            if empty.any():
                body = F.interpolate(body_mask, (h, w), mode="nearest").flatten(1).bool()
                m = torch.where(empty[:, None], body, m)
            for attn in self.cross_attn:
                attended, _ = attn(q.unsqueeze(1), flat, flat, key_padding_mask=~m)
                q = q + attended.squeeze(1)
            queries.append(q)
        tokens = torch.stack([global_token, *queries], 1)
        tokens = self.region_transformer(tokens)
        return tokens

    def concept_logits(self, tokens):
        gl = {c["key"]: self.global_heads[c["key"]](tokens[:, 0]) for c in self.global_specs}
        rl = {}
        for c in self.regional_specs:
            idx = self.region_index[c["region"]] + 1
            rl[c["key"]] = self.regional_heads[c["key"]](tokens[:, idx])
        return gl, rl

    def concept_probs(self, gl, rl):
        parts = []
        for c in self.global_specs:
            parts.append(self.global_heads[c["key"]].probs_from_logits(gl[c["key"]]))
        for c in self.regional_specs:
            parts.append(self.regional_heads[c["key"]].probs_from_logits(rl[c["key"]]))
        return torch.cat(parts, -1)

    @staticmethod
    def demo_input(physiology: torch.Tensor) -> torch.Tensor:
        """人口学调制输入 [Age(z-score), 性别one-hot(2)];性别编码与 FT-Transformer 一致。"""
        g = (physiology[:, 0] > 0).float()
        return torch.stack([physiology[:, 1], g, 1.0 - g], -1)

    def risk_from_concepts(self, concept_probs, physiology, phys_missing=None, visual_global=None,
                           region_mean=None, log_area=None, gap_feat=None, mask_rate=None,
                           gap_feats=None):
        """gap_feats:各阶段A专家 GAP 特征堆叠 [K,B,d] —— 视觉证据 = 逐专家
        gap_logit 的均值(K=1 时退化为单专家,与 v12 行为一致)。"""
        B = physiology.shape[0]
        if visual_global is None:
            visual_global = torch.zeros(B, self.dim, device=physiology.device)
        if region_mean is None:
            region_mean = torch.zeros(B, self.dim, device=physiology.device)
        if log_area is None:
            log_area = torch.zeros(B, device=physiology.device)
        if gap_feat is None:
            gap_feat = torch.zeros(B, self.dim, device=physiology.device)
        if mask_rate is None:
            mask_rate = torch.ones(B, device=physiology.device)
        visual_global = visual_global.to(concept_probs.dtype)
        region_mean = region_mean.to(concept_probs.dtype)
        log_area = log_area.to(concept_probs.dtype)
        gap_feat = gap_feat.to(concept_probs.dtype)
        mask_rate = mask_rate.to(concept_probs.dtype)
        # v3 ②:demo 逐通道调制,只作用于送入融合头的视觉表示
        demo_gate = 2.0 * torch.sigmoid(self.demo_mod(self.demo_input(physiology)))
        v_mod, r_mod = visual_global * demo_gate, region_mean * demo_gate
        # v3 ①:tongue 通路输入 = [Ĉ, 未调制 v, 未调制 r_mean](保持纯视觉,便于解释)
        zt = self.tongue_encoder(torch.cat([concept_probs, visual_global, region_mean], -1))
        zp = self.phys_encoder(physiology, phys_missing if phys_missing is not None
                               else torch.zeros_like(physiology))
        pt, ut, at = self.tongue_evidence(zt)
        pp, up, ap = self.phys_evidence(zp)
        gate_in = torch.cat([zt, zp, ut[:, None], up[:, None]], -1)
        alpha = F.softmax(self.gate(gate_in), -1)
        inter = torch.cat([zt, zp, zt * zp, torch.abs(zt - zp)], -1)
        ptp = torch.sigmoid(self.interaction_head(inter)).squeeze(-1)
        # v6:直连分支,纯视觉(demo 调制)+ 面积,不依赖概念(概念反事实不受影响)
        direct_logit = self.direct_head(
            torch.cat([v_mod, r_mod, log_area[:, None]], -1)).squeeze(-1)
        # v11:TDF 头分支(视觉 GAP×mask_rate×人口学门控 + 6 指标),阶段 A 权重直载
        # v12-ens:多专家时逐专家算 logit 再平均(生理反事实会带着新 physiology 重算)
        if gap_feats is not None:
            feats_list = ([gap_feats[i] for i in range(gap_feats.shape[0])]
                          if torch.is_tensor(gap_feats) else list(gap_feats))
            heads = [self.gap_head, *getattr(self, "aux_gap_heads", [])]
            logits = [h(f.to(concept_probs.dtype), mask_rate, physiology)
                      for h, f in zip(heads, feats_list)]
            gap_logit = torch.stack(logits).mean(0).to(concept_probs.dtype)
            gap_feat = torch.stack(
                [f.to(concept_probs.dtype) for f in feats_list]).mean(0)
        else:
            gap_logit = self.gap_head(gap_feat, mask_rate, physiology).to(concept_probs.dtype)
        if self.use_fusion_head and self.evidence_fusion:
            # v12:证据拼接 + gap_logit 恒等残差(零初始化 → 起步=阶段A单任务性能)
            fused_logit = self.fusion_head(gap_feat.to(concept_probs.dtype), gap_logit,
                                           physiology.to(concept_probs.dtype), concept_probs)
            pf = torch.sigmoid(fused_logit)
        elif self.use_fusion_head:
            # v3 ①+④ + v6 + v8:调制 token、log 面积、direct/gap logit 直连融合头
            fused_logit = self.fusion_head(torch.cat([inter, physiology, v_mod, r_mod,
                                                      log_area[:, None],
                                                      direct_logit[:, None],
                                                      gap_logit[:, None]], -1)).squeeze(-1)
            pf = torch.sigmoid(fused_logit)
        else:
            fused_logit = None
            pf = alpha[:, 0] * pt + alpha[:, 1] * pp + alpha[:, 2] * ptp
        return {"tongue_risk": pt, "phys_risk": pp, "tp_risk": ptp, "fused_risk": pf,
                "fused_logit": fused_logit, "visual_global": visual_global,
                "region_mean": region_mean, "log_area": log_area,
                "gap_feat": gap_feat, "mask_rate": mask_rate, "gap_feats": gap_feats,
                "direct_risk": torch.sigmoid(direct_logit), "direct_logit": direct_logit,
                "gap_risk": torch.sigmoid(gap_logit), "gap_logit": gap_logit,
                "gate": alpha, "tongue_uncertainty": ut, "phys_uncertainty": up,
                "modality_disagreement": torch.abs(pt - pp),
                "tongue_alpha": at, "phys_alpha": ap, "z_tongue": zt, "z_phys": zp}

    def forward(self, image, body_mask, region_masks, boxes, physiology, phys_missing=None):
        feats = self.encoder(image, body_mask)
        # v11:GAP 特征 + mask_rate(TIPNet 的面积用法)供 TDF 头使用
        gap_feat = feats.mean((-2, -1))
        mask1 = F.max_pool2d(body_mask, 8, 8)
        mask_rate = (mask1.shape[-2] * mask1.shape[-1]) / mask1.sum((-2, -1)).clamp_min(1.0)
        # v12-ens:辅助专家(冻结)各自前向,视觉证据 = 多专家平均
        gap_feats = [gap_feat]
        for enc in getattr(self, "aux_encoders", []):
            gap_feats.append(enc(image, body_mask).mean((-2, -1)))
        tokens = self.tokens_from_feats(feats, body_mask, region_masks, boxes)
        gl, rl = self.concept_logits(tokens)
        cv = self.concept_probs(gl, rl)
        # v3 ④:舌体面积占比 → log,作为体型相关特征直连融合头
        area_ratio = body_mask.flatten(1).sum(-1) / (body_mask.shape[-2] * body_mask.shape[-1])
        log_area = torch.log(area_ratio.clamp_min(1e-4))
        out = self.risk_from_concepts(cv, physiology, phys_missing, visual_global=tokens[:, 0],
                                      region_mean=tokens[:, 1:].mean(1), log_area=log_area,
                                      gap_feat=gap_feat, mask_rate=mask_rate,
                                      gap_feats=torch.stack(gap_feats, 0))
        out.update({"global_logits": gl, "regional_logits": rl, "concept_probs": cv})
        return out
