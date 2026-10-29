"""训练与评估引擎(论文版 §31)。

Stage 1 概念预训练(epochs_concept):仅概念监督,按验证概念损失选模
Stage 2 风险联合训练(epochs_risk):4 项损失,按验证 Brier 选模
Stage 3 校准:冻结模型,验证集并行拟合 Temperature/Isotonic 取 NLL 更低者
单模型训练,无深度集成(论文版)
评估:AUROC/AUPRC/F1/灵敏度/特异度(Youden 最优阈值)/Brier/ECE + bootstrap 95%CI
分概念指标(二分类 AUROC、多分类 Macro-F1、有序 Quadratic Kappa + Ordinal MAE)。
"""
from __future__ import annotations
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibration import bootstrap_metric_ci, expected_calibration_error, fit_best_calibrator
from .concepts import MISSING, concept_vector_layout
from .data import load_splits
from .losses import compute_losses, binary_loss, ordinal_loss, multiclass_loss
from .model import ReactFLD
from .utils import resolve_device, save_json, seed_everything


def move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {a: b.to(device, non_blocking=True) for a, b in v.items()}
        else:
            out[k] = v
    out["labels"] = {**out["global_labels"], **out["regional_labels"]}
    return out


class EMA:
    """v4:参数指数滑动平均;小数据集上稳定验证选模,best.pt 保存 EMA 权重。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module):
        """模型权重 ↔ EMA 权重互换:验证前换入 EMA,验证后换回继续训练。"""
        for k, v in model.state_dict().items():
            tmp = v.detach().clone()
            v.copy_(self.shadow[k].to(v.dtype))
            self.shadow[k] = tmp.float()


FLIP_PAIR = {"right": "left", "left": "right"}


def flip_batch(batch: dict, regions=("tip", "middle", "root", "right", "left")) -> dict:
    """水平翻转(image/body_mask/region_masks/boxes 同步;左右区域通道互换,
    boxes 为归一化 (x_center,y_center,w,h) → x_center'=1-x_center)。"""
    order = [regions.index(FLIP_PAIR.get(r, r)) for r in regions]
    out = dict(batch)
    out["image"] = batch["image"].flip(-1)
    out["body_mask"] = batch["body_mask"].flip(-1)
    out["region_masks"] = batch["region_masks"].flip(-1)[:, order]
    boxes = batch["boxes"].clone()
    boxes[..., 0] = 1.0 - boxes[..., 0]
    out["boxes"] = boxes[:, order]
    return out


def _logit_mean(*probs):
    p = torch.clamp(torch.stack(probs), 1e-6, 1 - 1e-6)
    return torch.sigmoid(torch.logit(p).mean(0))


@torch.no_grad()
def predict_split(model, loader, device, tta: bool = False) -> dict[str, np.ndarray]:
    """tta=True 时对同一批做原图+水平翻转两次前向,各概率输出按 logit 平均。"""
    model.eval()
    regions = tuple(getattr(model, "regions", ("tip", "middle", "root", "right", "left")))
    ys, fused, tongue, phys, unc, direct, gap = [], [], [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["body_mask"], batch["region_masks"],
                    batch["boxes"], batch["physiology"], batch["phys_missing"])
        if tta:
            fb = flip_batch(batch, regions)
            out2 = model(fb["image"], fb["body_mask"], fb["region_masks"],
                         fb["boxes"], fb["physiology"], fb["phys_missing"])
            for key in ("fused_risk", "tongue_risk", "phys_risk", "direct_risk", "gap_risk"):
                if key in out and key in out2:
                    out[key] = _logit_mean(out[key].float(), out2[key].float())
        ys.append(batch["label"].cpu())
        fused.append(out["fused_risk"].cpu())
        tongue.append(out["tongue_risk"].cpu())
        phys.append(out["phys_risk"].cpu())
        unc.append(((out["tongue_uncertainty"] + out["phys_uncertainty"]) / 2).cpu())
        if "direct_risk" in out:                       # v6:直连分支
            direct.append(out["direct_risk"].cpu())
        if "gap_risk" in out:                          # v8:GAP 纯 CNN 分支
            gap.append(out["gap_risk"].cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    result = {"y": cat(ys), "fused": cat(fused), "tongue": cat(tongue), "phys": cat(phys),
              "unc": cat(unc)}
    if direct:
        result["direct"] = cat(direct)
    if gap:
        result["gap"] = cat(gap)
    return result


def classification_metrics(y, p) -> dict[str, float]:
    """二分类风险预测全面指标(对齐医学顶刊标准)。"""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }
    if len(np.unique(y)) > 1:
        out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))
        fpr, tpr, thresholds = roc_curve(y, p)
        j = int(np.argmax(tpr - fpr))
        t = float(thresholds[j])
        pred_t = (p >= t).astype(int)
        pos, neg = y == 1, y == 0
        sens = float((pred_t[pos] == 1).mean()) if pos.sum() > 0 else 0.0
        spec = float((pred_t[neg] == 0).mean()) if neg.sum() > 0 else 0.0
        pred_pos_mask, pred_neg_mask = pred_t == 1, pred_t == 0
        prec_youden = float((y[pred_pos_mask] == 1).mean()) if pred_pos_mask.sum() > 0 else 0.0
        npv_youden = float((y[pred_neg_mask] == 0).mean()) if pred_neg_mask.sum() > 0 else 0.0
        out.update(sensitivity=round(sens, 4), specificity=round(spec, 4),
                   precision_youden=round(prec_youden, 4), npv_youden=round(npv_youden, 4),
                   operating_threshold=round(t, 4))
    else:
        out.update(sensitivity=0.0, specificity=0.0, precision_youden=0.0, npv_youden=0.0,
                   operating_threshold=0.5)
    return {k: round(float(v), 4) for k, v in out.items()}


@torch.no_grad()
def concept_metrics(model, loader, device, layout: list[dict]) -> dict[str, dict]:
    """分概念评估:binary→AUROC,ordinal→Quadratic Kappa,multiclass→Macro-F1。"""
    model.eval()
    preds: dict[str, list] = {c["key"]: [] for c in layout}
    targets: dict[str, list] = {c["key"]: [] for c in layout}
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["body_mask"], batch["region_masks"],
                    batch["boxes"], batch["physiology"], batch["phys_missing"])
        offset = 0
        for spec in layout:
            n = spec["classes"]
            probs = out["concept_probs"][:, offset:offset + n]
            preds[spec["key"]].extend(probs.detach().cpu().tolist())
            targets[spec["key"]].extend(batch["labels"][spec["key"]].cpu().tolist())
            offset += n
    metrics = {}
    for spec in layout:
        y = np.asarray(targets[spec["key"]])
        p = np.asarray(preds[spec["key"]])
        keep = y != MISSING
        if keep.sum() < 2 or len(np.unique(y[keep])) < 2:
            continue
        y, p = y[keep], p[keep]
        yhat = p.argmax(-1)
        entry = {"n_labeled": int(keep.sum()), "accuracy": round(float(accuracy_score(y, yhat)), 4)}
        if spec["type"] == "binary" and len(np.unique(y)) == 2:
            entry["auroc"] = round(float(roc_auc_score(y, p[:, 1])), 4)
            entry["precision"] = round(float(precision_score(y, yhat, zero_division=0)), 4)
            entry["recall"] = round(float(recall_score(y, yhat, zero_division=0)), 4)
        elif spec["type"] == "ordinal" and len(np.unique(y)) > 1:
            entry["quadratic_kappa"] = round(float(cohen_kappa_score(y, yhat, weights="quadratic")), 4)
            entry["ordinal_mae"] = round(float(np.mean(np.abs(yhat - y))), 4)
        else:
            entry["macro_f1"] = round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4)
            entry["balanced_accuracy"] = round(float(balanced_accuracy_score(y, yhat)), 4)
        metrics[spec["cn"]] = entry
    return metrics


def _binary_pos_weights(dataset, layout: list[dict]) -> dict[str, float]:
    weights = {}
    valid_nums = set(dataset.df["num"].tolist())
    for spec in layout:
        if spec["type"] != "binary":
            continue
        pos = 0
        neg = 0
        for num, labels in dataset.concept_lookup.items():
            if num not in valid_nums:
                continue
            val = labels[spec["key"]]
            if val == 1:
                pos += 1
            elif val == 0:
                neg += 1
        weights[spec["key"]] = float(max(neg, 1) / max(pos, 1))
    return weights


def _risk_pos_weight(dataset) -> float:
    y = dataset.df["FLD"].astype(float).values
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(max(neg, 1) / max(pos, 1))


def _save_training_history(history: list[dict], output: Path, title: str = "") -> None:
    save_json(history, output / "training_history.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["total"] for h in history], "o-", ms=3, label="train total")
    val = [(h["epoch"], h["score"]) for h in history if h["stage"] == "risk"]
    if val:
        ax = axes[0].twinx()
        ax.plot([e for e, _ in val], [s for _, s in val], "s--", ms=3, color="tab:red",
                label="-val AUROC")
        ax.set_ylabel("-val AUROC (risk stage)")
        ax.legend(loc="upper right")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].legend(loc="upper left")
    for key in ("concept", "fused", "aux", "evidential"):
        ys = [h.get(key) for h in history]
        if any(y is not None for y in ys):
            axes[1].plot(ep, [y if y is not None else float("nan") for y in ys], label=key)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=130)
    plt.close(fig)


def _lr_multiplier(epoch: int, total: int, warmup: int) -> float:
    """warmup 线性升温 + cosine 退火(对齐 TIPNet 协议),每阶段独立一轮。"""
    if warmup > 0 and epoch <= warmup:
        return epoch / warmup
    t = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * t))


def _fit_single(config: dict[str, Any], train_ds, val_ds, layout: list[dict],
                pos_weights: dict[str, float], risk_pos_weight: float,
                device, output: Path) -> None:
    """Stage1 概念预训练 → Stage2 风险联合训练(论文版 §31),最优权重存 output/best.pt。"""
    tc = config["training"]
    workers = int(tc.get("num_workers", 4))
    loaders = {name: DataLoader(ds, batch_size=int(tc["batch_size"]), shuffle=name == "train",
                                num_workers=workers, pin_memory=device.type == "cuda")
               for name, ds in (("train", train_ds), ("val", val_ds))}
    model = ReactFLD(config, physiology_dim=train_ds.physiology_dim).to(device)
    tr = config.get("transfer") or {}
    frozen_modules: list = []
    if tr.get("stage_a_checkpoint"):
        ck_a = torch.load(tr["stage_a_checkpoint"], map_location=device, weights_only=False)
        model.encoder.load_state_dict(ck_a["encoder"])
        model.gap_head.load_state_dict(ck_a["gap_head"])
        extra = tr.get("extra_checkpoints") or []
        if extra:
            assert len(extra) == model.ensemble_experts, \
                f"extra_checkpoints({len(extra)}) 与 ensemble_experts({model.ensemble_experts}) 不一致"
            for i, path in enumerate(extra):
                ck_e = torch.load(path, map_location=device, weights_only=False)
                model.aux_encoders[i].load_state_dict(ck_e["encoder"])
                model.aux_gap_heads[i].load_state_dict(ck_e["gap_head"])
            print(f"[transfer] 集成专家已加载:{[str(p) for p in extra]}")
        if tr.get("freeze_encoder", False):
            for p in model.encoder.parameters():
                p.requires_grad = False
        if tr.get("freeze_gap", False):
            for p in model.gap_head.parameters():
                p.requires_grad = False
        mode = ("冻结" if tr.get("freeze_encoder", False)
                else f"低lr微调(×{tr.get('encoder_lr_mult', 1.0)})")
        print(f"[transfer] 已加载单任务视觉权重,编码器{mode}:{tr['stage_a_checkpoint']}")
        if tr.get("freeze_encoder", False):
            # v12:冻结专家训练全程保持 eval —— 否则 train 模式下 BatchNorm 的
            # running stats 仍会被(带增强的)训练批更新,破坏阶段A的推理分布
            frozen_modules = [model.encoder]
            if tr.get("freeze_gap", False):
                frozen_modules.append(model.gap_head)
            if hasattr(model, "aux_encoders"):
                frozen_modules += [*model.aux_encoders, *model.aux_gap_heads]
    # v10:整个 encoder 一组(backbone_lr×encoder_lr_mult),其余参数一组(lr);
    # TIPNet 编码器无 encoder.backbone 前缀,旧分组方式会漏
    encoder_mult = float(tr.get("encoder_lr_mult", 1.0)) if tr.get("stage_a_checkpoint") else 1.0
    encoder_params = [p for n, p in model.named_parameters()
                      if n.startswith("encoder.") and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": float(tc["backbone_lr"]) * encoder_mult},
        {"params": other_params, "lr": float(tc["lr"])}],
        weight_decay=float(tc["weight_decay"]))
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    warmup = int(tc.get("warmup", 5))
    output.mkdir(parents=True, exist_ok=True)
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    stages = [("concept", int(tc["epochs_concept"])), ("risk", int(tc["epochs_risk"]))]
    global_epoch = 0
    history: list[dict] = []
    ema_cfg = tc.get("ema", {})
    ema = EMA(model, float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("enabled", False) else None

    for stage, epochs in stages:
        if stage == "risk" and tc.get("freeze_concept_in_risk", True):
            # v12:风险阶段冻结 encoder/gap_head(transfer 已冻)之外,连概念头与
            # token 管线一起冻结 —— 概念分支只在概念阶段由自身监督训练,
            # 风险阶段只有证据分类器与 agent 侧小头可学(梯度不可能回流共享网络)
            frozen = ("encoder.", "gap_head.", "global_heads.", "regional_heads.",
                      "box_encoder", "cross_attn", "region_transformer", "region_embedding",
                      "aux_encoders", "aux_gap_heads")
            n_frozen = sum(1 for n, p in model.named_parameters()
                           if n.startswith(frozen) and p.requires_grad)
            for n, p in model.named_parameters():
                if n.startswith(frozen):
                    p.requires_grad = False
            live = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([{"params": live, "lr": float(tc["lr"])}],
                                          weight_decay=float(tc["weight_decay"]))
            base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
            print(f"[v12] 风险阶段:再冻结 {n_frozen} 组概念/token 参数,重建 optimizer "
                  f"({len(live)} 个可训练张量,lr={tc['lr']})")
        best, stale = float("inf"), 0                       # 每阶段独立选模
        for epoch in range(1, epochs + 1):
            global_epoch += 1
            mult = _lr_multiplier(epoch, epochs, warmup)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * mult
            model.train()
            for m in frozen_modules:
                m.eval()
            sums = defaultdict(float)
            for batch in tqdm(loaders["train"], desc=f"{stage} {epoch}/{epochs}", leave=False):
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    out = model(batch["image"], batch["body_mask"], batch["region_masks"],
                                batch["boxes"], batch["physiology"], batch["phys_missing"])
                loss, parts = compute_losses(out, batch, config, layout, pos_weights,
                                             stage=stage, risk_pos_weight=risk_pos_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                for k, v in parts.items():
                    sums[k] += v

            if ema is not None:
                ema.swap(model)          # v4:验证与选模均基于 EMA 权重
            val = predict_split(model, loaders["val"], device)

            # 验证集概念损失,用于概念阶段选模
            val_concept_loss = 0.0
            n_val_batches = 0
            model.eval()
            with torch.no_grad():
                for batch in loaders["val"]:
                    batch = move_batch(batch, device)
                    with torch.amp.autocast("cuda", enabled=amp):
                        out = model(batch["image"], batch["body_mask"], batch["region_masks"],
                                    batch["boxes"], batch["physiology"], batch["phys_missing"])
                    c_loss = 0.0
                    for spec in layout:
                        key = spec["key"]
                        target = batch["labels"][key]
                        logits = out["global_logits"].get(key, out["regional_logits"].get(key))
                        if logits is None: continue
                        if spec["type"] == "binary":
                            c_loss += binary_loss(logits, target, pos_weights.get(key, 1.0))
                        elif spec["type"] == "ordinal":
                            c_loss += ordinal_loss(logits, target)
                        else:
                            c_loss += multiclass_loss(logits, target)
                    val_concept_loss += (c_loss / max(len(layout), 1)).item()
                    n_val_batches += 1
            val_concept_loss /= max(n_val_batches, 1)

            n_batches = max(len(loaders["train"]), 1)
            avg_concept = sums["concept"] / n_batches
            if stage == "risk":
                # v2:风险阶段按验证集 AUROC 选模(对齐基线协议),score 取负号保持"越小越好"
                if len(np.unique(val["y"])) > 1:
                    score = -float(roc_auc_score(val["y"], val["fused"]))
                else:
                    score = -0.5
                print(f"[{stage}] epoch={global_epoch} loss={sums['total']/n_batches:.4f} "
                      f"concept={avg_concept:.4f} val_auroc={-score:.4f} lr_mult={mult:.3f}")
            else:
                score = val_concept_loss
                print(f"[{stage}] epoch={global_epoch} loss={sums['total']/n_batches:.4f} "
                      f"train_concept={avg_concept:.4f} val_concept={score:.4f}")

            if score < best:
                best, stale = score, 0
                output.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "stats": train_ds.stats, "config": config},
                           output / "best.pt")
            else:
                stale += 1
                if stale >= int(tc["patience"]):
                    break
            if ema is not None:
                ema.swap(model)          # 换回训练权重;若上面触发 break,训练即结束,无需换回
            history.append({"epoch": global_epoch, "stage": stage,
                            **{k: v / n_batches for k, v in sums.items()}, "score": score})

    _save_training_history(history, output)


def best_accuracy_threshold(y, p) -> tuple[float, float]:
    """验证集上搜准确率最优阈值(并列时取最接近 0.5 者)。"""
    y, p = np.asarray(y), np.asarray(p)
    cands = np.unique(p)
    accs = np.array([np.mean((p >= t).astype(int) == y) for t in cands])
    i = int(np.argmax(accs - 1e-9 * np.abs(cands - 0.5)))
    return float(cands[i]), float(accs[i])


def train(config: dict[str, Any], config_path, output_dir: str, max_samples: int | None = None) -> Path:
    device = resolve_device(config["device"])
    seed_everything(int(config["seed"]))
    train_ds, val_ds, test_ds = load_splits(config, config_path)
    if max_samples:
        for ds in (train_ds, val_ds, test_ds):
            ds.df = ds.df.head(max_samples).reset_index(drop=True)
            ds.phys_raw_all = ds.phys_raw_all[:max_samples]
    tc = config["training"]
    layout = concept_vector_layout()
    pos_weights = _binary_pos_weights(train_ds, layout)
    risk_pos_weight = _risk_pos_weight(train_ds)
    workers = int(tc.get("num_workers", 4))
    output = Path(output_dir)

    _fit_single(config, train_ds, val_ds, layout, pos_weights, risk_pos_weight, device, output)

    model = ReactFLD(config, physiology_dim=train_ds.physiology_dim)
    ck = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    # Stage 3 校准(论文版 §31):验证集拟合,测试集出 raw + calibrated 双指标
    val_loader = DataLoader(val_ds, batch_size=int(tc["batch_size"]), shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=int(tc["batch_size"]), shuffle=False, num_workers=workers)
    val = predict_split(model, val_loader, device)
    calibrator = fit_best_calibrator(val["fused"], val["y"],
                                     mode=str(tc.get("calibrator", "auto")))
    test = predict_split(model, test_loader, device)
    calibrated = calibrator.transform(test["fused"])
    val_calibrated = calibrator.transform(val["fused"])
    thr, val_acc = best_accuracy_threshold(val["y"], val_calibrated)
    pred_thr = (calibrated >= thr).astype(int)
    y_bool = test["y"].astype(int)
    at_thr = {
        "threshold": round(thr, 4), "val_accuracy": round(val_acc, 4),
        "accuracy": round(float(accuracy_score(y_bool, pred_thr)), 4),
        "precision": round(float(precision_score(y_bool, pred_thr, zero_division=0)), 4),
        "recall": round(float(recall_score(y_bool, pred_thr, zero_division=0)), 4),
        "f1": round(float(f1_score(y_bool, pred_thr, zero_division=0)), 4),
        "specificity": round(float((pred_thr[y_bool == 0] == 0).mean()), 4),
    }
    result = {
        "test": classification_metrics(test["y"], test["fused"]),
        "test_calibrated": classification_metrics(test["y"], calibrated),
        "test_calibrated_at_val_threshold": at_thr,
        "tongue_only": classification_metrics(test["y"], test["tongue"]),
        "physiology_only": classification_metrics(test["y"], test["phys"]),
        "calibrator": calibrator.state_dict(),
        "auroc_ci95": bootstrap_metric_ci(test["y"], calibrated, roc_auc_score,
                                          n_boot=500, seed=int(config["seed"])),
        "concept_metrics": concept_metrics(model, test_loader, device, layout),
    }
    extra = {}
    if "direct" in test:                                # v6:直连分支单独评价
        result["direct_only"] = classification_metrics(test["y"], test["direct"])
        extra["direct"] = test["direct"]
    if "gap" in test:                                   # v8:GAP 纯 CNN 分支单独评价
        result["gap_only"] = classification_metrics(test["y"], test["gap"])
        extra["gap"] = test["gap"]
    np.savez(output / "test_predictions.npz", y=test["y"], fused_raw=test["fused"],
             calibrated=calibrated, tongue=test["tongue"], phys=test["phys"],
             val_y=val["y"], val_calibrated=val_calibrated, **extra)
    ckpt = {"model": ck["model"], "stats": train_ds.stats, "config": config,
            "calibrator": calibrator.state_dict()}
    torch.save(ckpt, output / "best.pt")
    save_json(result, output / "metrics.json")
    print(f"test calibrated: {result['test_calibrated']}")
    print(f"AUROC 95%CI: {result['auroc_ci95']}")
    return output / "best.pt"


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ReactFLD(ckpt["config"], physiology_dim=len(ckpt["stats"]["mean"]))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    state = ckpt.get("calibrator", {})
    from .calibration import IsotonicCalibrator, TemperatureScaler
    if state.get("kind") == "isotonic":
        cal = IsotonicCalibrator()
        cal.iso.fit(np.asarray(state["x"]), np.asarray(state["y"]))   # 阈值点上重拟合,恢复完整属性
        calibrator = cal
    else:
        calibrator = TemperatureScaler(state.get("temperature", 1.0))
    return model, ckpt, calibrator
