"""模型评价:Precision / Recall / F1-score / Accuracy / AUC(论文同款五指标)。

两种用法:
  1. 快速评价(推荐,分钟级):从 checkpoint 直接对 test split 批量预测
     python evaluate.py --checkpoint runs/v2_fusion/best.pt
  2. agentic 输出目录评价(需先跑 infer-batch):
     python evaluate.py runs/inference_full_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from reactfld.data import load_splits
from reactfld.engine import best_accuracy_threshold, load_checkpoint, predict_split
from reactfld.utils import load_config, resolve_device

BASELINES = {
    "TIPNet_9_Indicators": {"precision": 0.787, "recall": 0.734, "f1": 0.766,
                            "accuracy": 0.841, "auroc": 0.916},
    "TIPNet_9_NoneAtten": {"precision": 0.692, "recall": 0.674, "f1": 0.683,
                           "accuracy": 0.779, "auroc": 0.858},
    "REACT-FLD_v11(TIPNet复现+六损失多任务融合)": {"precision": 0.791, "recall": 0.697, "f1": 0.741,
                                             "accuracy": 0.828, "auroc": 0.892},
    "REACT-FLD_v10(TIPNet编码器+低lr微调两阶段)": {"precision": 0.844, "recall": 0.648, "f1": 0.733,
                                            "accuracy": 0.833, "auroc": 0.899},
    "REACT-FLD_v9(单任务视觉预训练+冻结融合)": {"precision": 0.709, "recall": 0.622, "f1": 0.663,
                                         "accuracy": 0.776, "auroc": 0.844},
    "REACT-FLD_v8(GAP纯CNN直连+token双路对照)": {"precision": 0.702, "recall": 0.723, "f1": 0.712,
                                           "accuracy": 0.794, "auroc": 0.861},
    "REACT-FLD_v7(跳过概念预训练联合直训)": {"precision": 0.745, "recall": 0.667, "f1": 0.704,
                                       "accuracy": 0.802, "auroc": 0.858},
    "REACT-FLD_v6(概念分支+FLD直连分支)": {"precision": 0.699, "recall": 0.704, "f1": 0.701,
                                       "accuracy": 0.788, "auroc": 0.857},
    "REACT-FLD_v5(多尺度encoder+SelfAtten)": {"precision": 0.683, "recall": 0.775, "f1": 0.726,
                                          "accuracy": 0.794, "auroc": 0.865},
    "REACT-FLD_v4(窄融合头+增广+EMA)": {"precision": 0.739, "recall": 0.730, "f1": 0.734,
                                     "accuracy": 0.813, "auroc": 0.858},
    "REACT-FLD_v3(tokens直连+demo调制+面积)": {"precision": 0.712, "recall": 0.730, "f1": 0.721,
                                          "accuracy": 0.800, "auroc": 0.863},
    "REACT-FLD_v2(logit级融合头)": {"precision": 0.771, "recall": 0.693, "f1": 0.730,
                                  "accuracy": 0.819, "auroc": 0.866},
    "REACT-FLD_v1(agentic_complete)": {"precision": 0.767, "recall": 0.652, "f1": 0.704,
                                       "accuracy": 0.807, "auroc": 0.878},
    "REACT-FLD_v1(final)": {"precision": 0.742, "recall": 0.723, "f1": 0.732,
                            "accuracy": 0.814, "auroc": 0.866},
    "MFF-TDF 论文自报(自有数据集,仅供参考)": {"precision": None, "recall": 0.847, "f1": 0.797,
                                           "accuracy": 0.753, "auroc": 0.924},
}


def five_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, p)),
    }


def auroc_ci95(y, p, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    boots = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y))
                                                 for _ in range(n_boot))
             if len(np.unique(y[i])) > 1]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def print_comparison(ours: dict, ours_ci, out_path: Path | None = None, name: str = None):
    lines = ["方法对比(同一 756 例测试集,阈值 0.5;论文行为其自有数据集)", "-" * 88,
             f"{'方法':38s} {'Precision':>9s} {'Recall':>7s} {'F1':>7s} {'Accuracy':>9s} {'AUC':>7s}"]
    name = name or "REACT-FLD_v12(冻结阶段A主干+证据拼接零初始化残差)"
    m = ours
    lines.append(f"{name:38s} {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
                 f"{m['accuracy']:9.3f} {m['auroc']:7.3f}  (AUC 95%CI {ours_ci[0]:.3f}-{ours_ci[1]:.3f})")
    for name, b in BASELINES.items():
        row = {k: (f"{v:.3f}" if v is not None else "-") for k, v in b.items()}
        lines.append(f"{name:38s} {row['precision']:>9s} {row['recall']:>7s} {row['f1']:>7s} "
                     f"{row['accuracy']:>9s} {row['auroc']:>7s}")
    lines.append("-" * 88)
    lines.append("目标:Accuracy > 0.841(TIPNet_9_Indicators)")
    text = "\n".join(lines)
    print(text)
    if out_path:
        out_path.write_text(text, encoding="utf-8")


def eval_checkpoint(ckpt_path: str, config_path: str, config: dict, name: str = None,
                    tta: bool = False):
    device = resolve_device(config["device"])
    model, ck, calibrator = load_checkpoint(ckpt_path, device)
    _, val_ds, test_ds = load_splits(config, config_path)
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=int(config["training"]["batch_size"]),
                        shuffle=False, num_workers=int(config["training"].get("num_workers", 4)))
    test = predict_split(model, loader, device, tta=tta)
    calibrated = calibrator.transform(test["fused"])
    y = test["y"].astype(int)

    m05 = five_metrics(y, calibrated, 0.5)
    ci = auroc_ci95(y, calibrated)
    # v3 修正:阈值在验证集上选(旧版在测试集上挑,结果偏乐观);主口径仍为 0.5
    val_loader = DataLoader(val_ds, batch_size=int(config["training"]["batch_size"]),
                            shuffle=False, num_workers=int(config["training"].get("num_workers", 4)))
    val = predict_split(model, val_loader, device, tta=tta)
    thr, _ = best_accuracy_threshold(val["y"], calibrator.transform(val["fused"]))
    m_thr = five_metrics(y, calibrated, thr)
    print(f"checkpoint={ckpt_path}  n={len(y)}  (calibrated_risk, 阈值 0.5)")
    print(f"Precision={m05['precision']:.4f}  Recall={m05['recall']:.4f}  F1={m05['f1']:.4f}  "
          f"Accuracy={m05['accuracy']:.4f}  AUC={m05['auroc']:.4f} (CI95 {ci[0]:.3f}-{ci[1]:.3f})")
    print(f"[参考] 验证集阈值 {thr:.3f} 时:Accuracy={m_thr['accuracy']:.4f}  "
          f"Precision={m_thr['precision']:.4f}  Recall={m_thr['recall']:.4f}  F1={m_thr['f1']:.4f}")
    print_comparison(m05, ci, Path(ckpt_path).parent / "comparison_table.md", name=name)
    result = {"n": int(len(y)), "at_0.5": m05, "auroc_ci95": ci,
              "val_threshold": thr, "at_threshold": m_thr}
    out = Path(ckpt_path).parent / "evaluate_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out}")


def eval_evidence_dir(out_dir: Path, config: dict, config_path: str, threshold: float):
    _, _, test_ds = load_splits(config, config_path)
    df = test_ds.df[["num", "FLD"]]
    y, p, missing = [], [], []
    for num, fld in zip(df["num"], df["FLD"]):
        stem = f"{int(num):05d}"
        path = out_dir / f"{stem}_evidence.json"
        if not path.exists():
            missing.append(stem)
            continue
        risk = json.loads(path.read_text(encoding="utf-8"))["evidence"]["risk"]
        y.append(float(fld))
        p.append(float(risk["calibrated_risk"]))
    if missing:
        print(f"[warn] {len(missing)} 例缺少推理结果: {missing[:5]}...")
    y, p = np.asarray(y), np.asarray(p)
    m = five_metrics(y, p, threshold)
    ci = auroc_ci95(y, p)
    print(f"目录={out_dir}  n={len(y)}  阈值={threshold}")
    print(f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}  "
          f"Accuracy={m['accuracy']:.4f}  AUC={m['auroc']:.4f} (CI95 {ci[0]:.3f}-{ci[1]:.3f})")
    print_comparison(m, ci)


def main():
    parser = argparse.ArgumentParser(description="REACT-FLD v2 五指标评价")
    parser.add_argument("output", nargs="?", default=None, help="infer-batch 输出目录")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="快速评价:直接加载 checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--name", default=None, help="对比表中的方法名")
    parser.add_argument("--tta", action="store_true",
                        help="推理协议含水平翻转 TTA(两次前向 logit 平均)")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.checkpoint:
        eval_checkpoint(args.checkpoint, args.config, config, name=args.name, tta=args.tta)
    elif args.output:
        eval_evidence_dir(Path(args.output), config, args.config, args.threshold)
    else:
        parser.error("需要 --checkpoint 或输出目录参数")


if __name__ == "__main__":
    main()
