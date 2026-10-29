"""损失函数(论文版 §30,共 4 项)。

L = λ_c·L_concept + λ_r·L_risk + λ_a·L_aux + λ_e·L_evi

概念监督: binary→带 pos_weight 的 BCE;ordinal→CORAL;multiclass→CE
风险: 融合 BCE(带 pos_weight);辅助: 两单模态通路 BCE
证据正则: Sensoy 型 Dirichlet 正则(拟合 + 方差收缩 + 错误证据惩罚)
不再加入: 残差去相关、对称相关、一致性、不变性等额外正则项(论文版 §30.5)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from .concepts import MISSING


def evidential_regularizer(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 强制 fp32,防止 fp16 下 alpha 过大导致方差计算溢出
    alpha = alpha.float()
    target = target.float()
    onehot = F.one_hot(target.long(), 2).float()
    total = alpha.sum(-1, keepdim=True)
    probs = alpha / total
    fit = ((onehot - probs) ** 2).sum(-1)
    variance = (alpha * (total - alpha) / (total * total * (total + 1))).sum(-1)
    wrong_evidence = ((1 - onehot) * (alpha - 1).clamp_max(1e4)).sum(-1)
    return (fit + variance + 1e-3 * wrong_evidence).mean()


def binary_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if (target == MISSING).all():
        return logits.new_zeros(())
    keep = target != MISSING
    weight = torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[keep], target[keep].float(), pos_weight=weight)


def ordinal_loss(cum_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL:K-1 个累积阈值的二元交叉熵。"""
    if (target == MISSING).all():
        return cum_logits.new_zeros(())
    keep = target != MISSING
    t = target[keep]
    levels = torch.arange(cum_logits.shape[-1], device=cum_logits.device)
    y_level = (t.unsqueeze(-1) > levels.unsqueeze(0)).float()
    return F.binary_cross_entropy_with_logits(cum_logits[keep], y_level)


def multiclass_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, ignore_index=MISSING)


def _to_float32(obj):
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        return {k: _to_float32(v) for k, v in obj.items()}
    return obj


def compute_losses(output: dict, batch: dict, config: dict, layout: list[dict],
                   binary_pos_weights: dict[str, float], stage: str = "risk",
                   risk_pos_weight: float = 1.0):
    weights = config["training"]["weights"]
    output = _to_float32(output)

    concept = output["global_logits"][list(output["global_logits"])[0]].new_zeros(())
    for spec in layout:
        key = spec["key"]
        target = batch["labels"][key]
        logits = output["global_logits"].get(key, output["regional_logits"].get(key))
        if logits is None:
            continue
        if spec["type"] == "binary":
            concept = concept + binary_loss(logits, target, binary_pos_weights.get(key, 1.0))
        elif spec["type"] == "ordinal":
            concept = concept + ordinal_loss(logits, target)
        else:
            concept = concept + multiclass_loss(logits, target)
    concept = concept / max(len(layout), 1)

    y = batch["label"]

    def safe_bce(prob, target, pos_weight=None):
        prob = prob.float().clamp(1e-7, 1.0 - 1e-7)
        target = target.float()
        weight = torch.tensor(pos_weight, device=prob.device) if pos_weight else None
        return F.binary_cross_entropy(prob, target, weight=weight)

    fused_logit = output.get("fused_logit")
    if fused_logit is not None:
        # v2:logit 空间 BCE(带 pos_weight),避免概率空间双饱和
        pw = torch.tensor(risk_pos_weight, device=fused_logit.device) if risk_pos_weight else None
        fused = F.binary_cross_entropy_with_logits(fused_logit.float(), y.float(), pos_weight=pw)
    else:
        fused = safe_bce(output["fused_risk"], y, risk_pos_weight)
    aux = safe_bce(output["tongue_risk"], y, risk_pos_weight) + safe_bce(output["phys_risk"], y, risk_pos_weight)
    evi = evidential_regularizer(output["tongue_alpha"], y) + evidential_regularizer(output["phys_alpha"], y)

    parts = {"concept": concept, "fused": fused, "aux": aux, "evidential": evi}
    if stage == "concept":
        # 论文版 §31 Stage 1:仅概念监督
        total = weights["concept"] * concept
    else:
        # v12:风险阶段概念项权重单独可配(默认 0 —— 概念分支此时已冻结,
        # 该项只剩日志意义,不再参与"指挥"共享网络)
        total = (weights.get("concept_risk", 0.0) * concept + weights["risk"] * fused
                 + weights["aux_risk"] * aux + weights["evidential"] * evi)
        # v6:直连分支 BCE,给视觉编码器不被概念多任务稀释的 FLD 梯度
        if "direct_logit" in output:
            dlogit = output["direct_logit"]
            dpw = torch.tensor(risk_pos_weight, device=dlogit.device) if risk_pos_weight else None
            direct = F.binary_cross_entropy_with_logits(dlogit.float(), y.float(), pos_weight=dpw)
            total = total + weights.get("direct", 1.0) * direct
            parts["direct"] = direct
        # v8:GAP 纯 CNN 分支 BCE,与 direct 同权重,同模型对照两条视觉路线
        if "gap_logit" in output:
            glogit = output["gap_logit"]
            gpw = torch.tensor(risk_pos_weight, device=glogit.device) if risk_pos_weight else None
            gap = F.binary_cross_entropy_with_logits(glogit.float(), y.float(), pos_weight=gpw)
            total = total + weights.get("gap", 1.0) * gap
            parts["gap"] = gap
    parts["total"] = total
    return total, {k: float(v.detach()) for k, v in parts.items()}
