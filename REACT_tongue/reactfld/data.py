"""数据管线（文档 §5/§8）：图像 + 全舌掩码 + 五区掩码 + 坐标 + 生理指标 + 概念标签。
生理向量 = 8 项原始指标 + 3 项派生指标（BMI/WHR/MAP），派生值一律由原始值重算
区域坐标框由区域掩码的包围盒推导（与 test_data_json.json 的标注框一致）
概念标签 5040 条全覆盖；无法判定的取值记 MISSING=-100，不参与损失
划分：按患者哈希 + FLD 分层（§20）。当前 CSV 无患者ID列，但已核验 5040 例的
8 项生理指标+FLD 无完全重复行（每图≈独立患者，按图划分等价于按患者划分）；
若检测到重复行会告警；未来在 data.patient_column 指定患者ID列后自动改为严格患者级分组
"""
from __future__ import annotations
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .concepts import ALL_CONCEPTS, GLOBAL_CONCEPTS, MISSING, REGIONAL_CONCEPTS, parse_label

RAW_PHYS = ["Gender", "Age", "Height", "Waist", "Hipline", "Weight", "SBP", "DBP"]
DERIVED_PHYS = ["BMI", "WHR", "MAP"]
PHYS_NAMES = RAW_PHYS + DERIVED_PHYS

# 【修复⑦】ImageNet 归一化参数，适配 ConvNeXt 预训练权重
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def compute_derived(raw: np.ndarray) -> np.ndarray:
    """由 8 项原始指标计算 3 项派生指标；反事实干预后必须调用本函数重算（§10.5）。"""
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    height_m = np.clip(raw[:, 2] / 100.0, 1.0, None)
    bmi = raw[:, 5] / (height_m ** 2)
    whr = raw[:, 3] / np.clip(raw[:, 4], 1.0, None)
    map_ = (raw[:, 6] + 2.0 * raw[:, 7]) / 3.0
    return np.column_stack([bmi, whr, map_])


def build_raw_physiology(df: pd.DataFrame) -> np.ndarray:
    raw = df[RAW_PHYS].astype(float).to_numpy(dtype=np.float64)
    derived = compute_derived(raw)
    return np.concatenate([raw, derived], axis=1).astype(np.float32)


def _load_rgb(path: Path, size: int, normalize: bool = True) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # 【修复⑦】转换为 (C, H, W) 并进行 ImageNet 归一化
    arr = arr.transpose(2, 0, 1)
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr)


def _load_body_mask(path: Path, size: int) -> torch.Tensor:
    m = np.asarray(Image.open(path)) if path.suffix != ".npy" else np.load(path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != (size, size):
        m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST))
    return torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)


def _load_region_masks(path: Path, size: int) -> torch.Tensor:
    m = np.load(path)
    if m.shape[1:] != (size, size):
        out = np.stack([
            np.asarray(Image.fromarray(m[i].astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST))
            for i in range(m.shape[0])])
        m = out
    return torch.from_numpy((m > 0).astype(np.float32))


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _load_coordinate_index(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    obj = json.load(open(path, encoding="utf-8-sig"))
    records = obj if isinstance(obj, list) else obj.get("data", obj.get("items", []))
    result = {}
    aliases = {"tip": "tongue_tip", "middle": "tongue_middle", "root": "tongue_root",
               "left": "tongue_left", "right": "tongue_right"}
    for item in records:
        value = item.get("image") or item.get("path") or item.get("quesid")
        if not value:
            continue
        number = int(Path(str(value).replace("\\", "/")).stem)
        regions = item.get("regions", {})
        result[number] = {name: regions.get(aliases[name], regions.get(name)) for name in aliases}
    return result


def mask_to_box(mask: torch.Tensor) -> list[float]:
    """掩码 → [cx, cy, w, h]（归一化）；空掩码返回中心默认框（§6.3 坐标先验）。"""
    ys, xs = torch.nonzero(mask.squeeze(0) > 0.5, as_tuple=True)
    if len(ys) == 0:
        return [0.5, 0.5, 0.0, 0.0]
    h, w = mask.shape[-2:]
    y1, y2, x1, x2 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
    return [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1 + 1) / w, (y2 - y1 + 1) / h]


class TongueDataset(Dataset):
    def __init__(self, root: str | Path, config: dict[str, Any], split: str, stats: dict | None = None):
        self.root = Path(root)
        dc = config["data"]
        self.size = int(dc["image_size"])
        self.regions = list(dc["regions"])
        self.cfg = config
        coord_name = dc.get("coordinates_json")
        self.coordinates = _load_coordinate_index(self.root / coord_name if coord_name else None)
        df = pd.read_csv(self.root / dc["csv"], dtype=str)
        df["num"] = df["quesid"].str.extract(r"(\d+)").astype(int)
        
        # 【关键修复】：CSV 中 Gender(_1/_2) 和 FLD(_0/_1) 带有下划线前缀，需先去除空格和下划线，再转 float
        for col in RAW_PHYS + ["FLD"]:
            df[col] = df[col].str.replace(" ", "", regex=False).str.replace("_", "", regex=False).astype(float)
            
        df["GenderRaw"] = df["Gender"]  # 1=男 2=女（匹配与参考用）

        labels = json.load(open(self.root / dc["concept_labels"], encoding="utf-8"))
        self.concept_lookup: dict[int, dict[str, int]] = {}
        for entry in labels:
            num = int(Path(entry["path"]).stem)
            parsed: dict[str, int] = {}
            infer = entry["infer"]
            for spec in GLOBAL_CONCEPTS:
                parsed[spec["key"]] = parse_label(spec, infer["请分析全局舌象特征"][spec["label_key"]])
            for spec in REGIONAL_CONCEPTS:
                parsed[spec["key"]] = parse_label(spec, infer["请分析五个区域局部舌象特征"][spec["label_key"]])
            self.concept_lookup[num] = parsed

        df = self._assign_split(df, config)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.phys_raw_all = build_raw_physiology(self.df)
        if stats is None:
            stats = {"mean": self.phys_raw_all.mean(0).tolist(), "std": (self.phys_raw_all.std(0) + 1e-6).tolist()}
        self.stats = stats
        # v4:仅训练集做色度扰动(增广不改变掩码/几何,验证测试路径与旧版逐位一致)
        self.augment = split == "train" and bool(config.get("training", {}).get("augment", False))
        if self.augment:
            from torchvision.transforms import ColorJitter
            self.train_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2)

    @staticmethod
    def _assign_split(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        seed = int(config["seed"])
        sp = config["data"]["split"]
        patient_col = config["data"].get("patient_column", "patient_id")
        if patient_col in df.columns:
            df["patient"] = df[patient_col].astype(str)
        else:
            n_dup = int(df.duplicated(subset=RAW_PHYS + ["FLD"]).sum())
            if n_dup:
                warnings.warn(
                    f"检测到 {n_dup} 行生理指标完全重复且无患者ID列（{patient_col}），"
                    "同一患者可能跨训练/测试集（违反文档 §20），请提供患者ID列", RuntimeWarning)
            df["patient"] = df["num"]  # 已核验无重复：每图≈独立患者

        def bucket(key) -> float:
            digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
            return int(digest[:8], 16) / 0xFFFFFFFF

        u = {p: bucket(p) for p in df["patient"].unique()}
        parts = []
        for fld, group in df.groupby("FLD"):  # FLD 分层下按患者整组划分
            patients = sorted(group["patient"].unique(), key=lambda p: u[p])
            n = len(patients)
            n_tr = int(round(sp["train"] * n))
            n_va = int(round(sp["val"] * n))
            assign = dict(zip(patients,
                              ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)))
            group = group.assign(split=group["patient"].map(assign))
            parts.append(group)
        return pd.concat(parts).sort_index()

    @property
    def physiology_dim(self) -> int:
        return len(PHYS_NAMES)

    def __len__(self) -> int:
        return len(self.df)

    def _path(self, name: str) -> Path:
        return self.root / name

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        num = int(row["num"])
        stem = f"{num:05d}"
        dc = self.cfg["data"]
        image_path = _first_existing([self._path(f"{dc['images_dir']}/{stem}{suffix}")
                                      for suffix in (".jpg", ".png", ".jpeg", ".JPG")])
        body_path = _first_existing([self._path(f"{dc['body_mask_dir']}/{stem}{suffix}")
                                     for suffix in (".npy", ".png", "_mask.png", "_body.png")])
        if image_path is None or body_path is None:
            raise FileNotFoundError(f"Missing image/body mask for {stem}")

        image = _load_rgb(image_path, self.size, normalize=not self.augment)
        if self.augment:
            if float(torch.rand(1)) < 0.8:
                image = self.train_jitter(image)
            image = (image - torch.from_numpy(IMAGENET_MEAN)) / torch.from_numpy(IMAGENET_STD)
        body = _load_body_mask(body_path, self.size)
        packed = self._path(f"{dc['region_mask_dir']}/{stem}.npy")
        if packed.exists():
            region_masks = _load_region_masks(packed, self.size)
        else:
            masks = []
            for region in self.regions:
                aliases = [region, {"middle": "center"}.get(region, region), f"tongue_{region}"]
                candidates = []
                for alias in aliases:
                    candidates.extend([
                        self._path(f"{dc['region_mask_dir']}/{stem}_{alias}.png"),
                        self._path(f"{dc['region_mask_dir']}/{alias}/{stem}.png"),
                        self._path(f"{dc['region_mask_dir']}/{stem}/{alias}.png"),
                    ])
                path = _first_existing(candidates)
                if path is None:
                    raise FileNotFoundError(f"Missing {region} mask for {stem}")
                masks.append(_load_body_mask(path, self.size)[0])
            region_masks = torch.stack(masks)

        if num in self.coordinates and all(self.coordinates[num].get(r) for r in self.regions):
            boxes = torch.tensor([[self.coordinates[num][r][key] for key in
                                   ("x_center", "y_center", "width", "height")]
                                  for r in self.regions], dtype=torch.float32)
        else:
            boxes = torch.tensor([mask_to_box(region_masks[i:i+1])
                                  for i in range(len(self.regions))], dtype=torch.float32)
        areas = region_masks.flatten(1).sum(-1)
        phys_raw = torch.from_numpy(self.phys_raw_all[index].astype(np.float32))
        phys_norm = (phys_raw - torch.tensor(self.stats["mean"], dtype=torch.float32)) / torch.tensor(self.stats["std"], dtype=torch.float32)
        missing = torch.zeros(len(PHYS_NAMES), dtype=torch.float32)
        concepts = self.concept_lookup[num]
        global_labels = {c["key"]: torch.tensor(concepts[c["key"]], dtype=torch.long) for c in GLOBAL_CONCEPTS}
        regional_labels = {c["key"]: torch.tensor(concepts[c["key"]], dtype=torch.long) for c in REGIONAL_CONCEPTS}

        return {
            "id": stem,
            "image": image,
            "body_mask": body,
            "region_masks": region_masks,
            "boxes": boxes,
            "areas": areas,
            "physiology": phys_norm,
            "physiology_raw": phys_raw,
            "phys_missing": missing,
            "label": torch.tensor(float(row["FLD"]), dtype=torch.float32),
            "global_labels": global_labels,
            "regional_labels": regional_labels,
        }


def load_splits(config: dict[str, Any], config_path: str | Path):
    """按 训练→验证→测试 顺序返回三个数据集，统计量只来自训练集。"""
    cfg_dir = Path(config_path).resolve().parent
    root = Path(config["data"]["root"])
    root = root if root.is_absolute() else cfg_dir / root
    train = TongueDataset(root, config, "train")
    val = TongueDataset(root, config, "val", train.stats)
    test = TongueDataset(root, config, "test", train.stats)
    return train, val, test