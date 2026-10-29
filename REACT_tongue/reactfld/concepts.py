"""概念定义表（文档 §7）：18 项舌象概念、中文取值归一化、正常类声明。

标签来源 demo_data/test_tongue_label.json，取值已按全量统计清洗：
- 合并同义值（"舌苔白"/"白"，"厚苔"/"舌苔厚"）
- 修正笔误（"该特征不壮"→"该特征不显著"）
- "该特征不显著"的语义按概念类型区分：
  * 存在类（裂纹/齿痕）→ 阴性（无该特征）
  * 舌形类 → 正常类（无形态异常）
  * 颜色/薄厚类 → 无法判定，记 MISSING 不参与损失
- 极稀有类别（舌苔黑 n=1）并入 MISSING
"""
from __future__ import annotations

MISSING = -100

VALUE_NORMALIZE = {
    "舌苔白": "白", "白": "白", "舌苔黄": "黄", "黄": "黄", "舌苔黑": "黑",
    "舌苔厚": "厚苔", "厚苔": "厚苔", "少苔": "少苔", "无苔": "无苔",
    "该特征不显著": "不显著", "该特征不壮": "不显著",
    "舌体瘦小": "瘦小", "正常": "正常", "舌体胖大": "胖大",
    "偏白": "偏白", "偏红": "偏红", "偏紫": "偏紫",
    "红": "红", "红点": "红点", "瘀点": "瘀点",
    "凹陷": "凹陷", "平直": "平直", "高凸": "高凸",
    "没有该特征": "无", "舌质裂纹": "有", "舌质齿痕": "有",
}

GLOBAL_CONCEPTS = [
    {"key": "fissure", "cn": "舌质裂纹", "label_key": "舌质裂纹特征", "type": "binary",
     "classes": ["无", "有"], "normal": 0, "missing_values": []},
    {"key": "tooth_mark", "cn": "舌质齿痕", "label_key": "舌质齿痕特征", "type": "binary",
     "classes": ["无", "有"], "normal": 0, "missing_values": []},
    {"key": "body_shape", "cn": "舌体胖瘦", "label_key": "舌体胖瘦特征", "type": "ordinal",
     "classes": ["瘦小", "正常", "胖大"], "normal": 1, "missing_values": []},
    {"key": "coating_thickness", "cn": "舌苔薄厚", "label_key": "舌苔薄厚特征", "type": "ordinal",
     "classes": ["无苔", "少苔", "厚苔"], "normal": 1, "missing_values": ["不显著"]},
    {"key": "coating_color", "cn": "舌苔颜色", "label_key": "舌苔颜色特征", "type": "multiclass",
     "classes": ["白", "黄"], "normal": 0, "missing_values": ["不显著", "黑"]},
    {"key": "body_color", "cn": "舌质颜色", "label_key": "舌质颜色特征", "type": "multiclass",
     "classes": ["偏白", "偏红", "偏紫"], "normal": 1, "missing_values": ["不显著"]},
]

REGIONAL_CONCEPTS = [
    {"key": "tip_color", "cn": "舌尖颜色", "label_key": "舌尖颜色特征", "region": "tip", "type": "multiclass",
     "classes": ["红", "红点", "瘀点"], "normal": 0, "missing_values": ["不显著"]},
    {"key": "tip_shape", "cn": "舌尖舌形", "label_key": "舌尖舌形特征", "region": "tip", "type": "multiclass",
     "classes": ["不显著", "凹陷", "平直", "高凸"], "normal": 0, "missing_values": []},
    {"key": "middle_coating_color", "cn": "舌中苔色", "label_key": "舌中舌苔颜色", "region": "middle", "type": "multiclass",
     "classes": ["白", "黄"], "normal": 0, "missing_values": ["不显著"]},
    {"key": "middle_coating_thickness", "cn": "舌中苔厚", "label_key": "舌中舌苔薄厚", "region": "middle", "type": "ordinal",
     "classes": ["无苔", "少苔", "厚苔"], "normal": 1, "missing_values": ["不显著"]},
    {"key": "middle_shape", "cn": "舌中舌形", "label_key": "舌中舌形特征", "region": "middle", "type": "multiclass",
     "classes": ["不显著", "凹陷", "高凸"], "normal": 0, "missing_values": []},
    {"key": "root_coating_color", "cn": "舌根苔色", "label_key": "舌根舌苔颜色", "region": "root", "type": "multiclass",
     "classes": ["白", "黄"], "normal": 0, "missing_values": ["不显著"]},
    {"key": "root_coating_thickness", "cn": "舌根苔厚", "label_key": "舌根舌苔厚薄", "region": "root", "type": "ordinal",
     "classes": ["无苔", "少苔", "厚苔"], "normal": 1, "missing_values": ["不显著"]},
    {"key": "root_shape", "cn": "舌根舌形", "label_key": "舌根舌形特征", "region": "root", "type": "multiclass",
     "classes": ["不显著", "凹陷", "高凸"], "normal": 0, "missing_values": []},
    {"key": "left_color", "cn": "舌左颜色", "label_key": "舌左侧颜色特征", "region": "left", "type": "multiclass",
     "classes": ["红", "红点", "瘀点"], "normal": 0, "missing_values": ["不显著"]},
    {"key": "left_shape", "cn": "舌左舌形", "label_key": "舌左侧舌形特征", "region": "left", "type": "multiclass",
     "classes": ["不显著", "凹陷", "高凸"], "normal": 0, "missing_values": []},
    {"key": "right_color", "cn": "舌右颜色", "label_key": "舌右侧颜色特征", "region": "right", "type": "multiclass",
     "classes": ["红", "红点", "瘀点"], "normal": 0, "missing_values": ["不显著"]},
    {"key": "right_shape", "cn": "舌右舌形", "label_key": "舌右侧舌形特征", "region": "right", "type": "multiclass",
     "classes": ["不显著", "凹陷", "高凸"], "normal": 0, "missing_values": []},
]

ALL_CONCEPTS = GLOBAL_CONCEPTS + REGIONAL_CONCEPTS
CONCEPT_BY_KEY = {c["key"]: c for c in ALL_CONCEPTS}


def parse_label(concept: dict, raw_value: str) -> int:
    """中文标签 → 类别索引；无法判定返回 MISSING。"""
    value = VALUE_NORMALIZE.get(str(raw_value).strip(), str(raw_value).strip())
    if value in concept["missing_values"]:
        return MISSING
    if concept["type"] == "binary" and value == "不显著":
        return concept["normal"]
    try:
        return concept["classes"].index(value)
    except ValueError:
        return MISSING


def concept_vector_layout() -> list[dict]:
    """概念瓶颈向量布局：softmax 概率拼接顺序（模型与归因共用）。"""
    layout = []
    for c in ALL_CONCEPTS:
        layout.append({"key": c["key"], "cn": c["cn"], "classes": len(c["classes"]),
                       "normal": c["normal"], "type": c["type"], "region": c.get("region", "global"),
                       "class_names": list(c["classes"])})
    return layout


def concept_vector_dim() -> int:
    return sum(len(c["classes"]) for c in ALL_CONCEPTS)
