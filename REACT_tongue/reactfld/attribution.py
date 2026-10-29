"""Patient-specific counterfactual tools used on demand by the agent.

All effects are calibrated model-level predictive contributions, not medical
causal effects. The reference bank must be the training split only.

论文版 §14-§18:Δp = p_factual − p_counterfactual。
概念反事实无需冻结残差特征:模型 z_T = g_T(Ĉ) 仅由概念向量生成,
替换概念块后直接重走 risk_from_concepts 即为纯概念效应(论文版 §16)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .concepts import CONCEPT_BY_KEY, concept_vector_layout
from .data import RAW_PHYS, compute_derived
from .schemas import EvidenceResult

FLIP_PAIR = {"right": "left", "left": "right"}
TTA_RISK_KEYS = ("fused_risk", "tongue_risk", "phys_risk", "tp_risk",
                 "direct_risk", "gap_risk")


def flip_sample(sample: dict, regions: list[str]) -> dict:
    """单样本水平翻转,与 engine.flip_batch 同语义:图像/掩码翻转,
    左右区域通道互换,boxes 归一化 x_center → 1-x_center。"""
    order = [regions.index(FLIP_PAIR.get(r, r)) for r in regions]
    out = dict(sample)
    out["image"] = sample["image"].flip(-1)
    out["body_mask"] = sample["body_mask"].flip(-1)
    out["region_masks"] = sample["region_masks"].flip(-1)[order]
    boxes = sample["boxes"].clone()
    boxes[..., 0] = 1.0 - boxes[..., 0]
    out["boxes"] = boxes[order]
    return out


def probs_logit_mean(probs: list[float]) -> float:
    t = torch.tensor(probs, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    return float(torch.sigmoid(torch.logit(t).mean()))


def interaction_counterfactual_risks(factual: float, concept_deltas: list[float],
                                     phys_deltas: list[float],
                                     joint_risks: list[float]) -> list[float]:
    """Δp^I = p̂ − p̂_a − p̂_b + p̂_ab(论文版 §18)。

    gamma = (p̂−p̂_ab) − Δa − Δb = −Δp^I;合成反事实风险 = p̂ + gamma = p̂_a+p̂_b−p̂_ab,
    使 summarize_evidence 的 delta = p̂ − cf 恰等于 Δp^I(协同增险为正)。
    """
    gammas = [(factual - joint) - cd - pd
              for joint, cd, pd in zip(joint_risks, concept_deltas, phys_deltas)]
    return [factual + g for g in gammas]


def summarize_evidence(evidence_id: str, target: str, level: str,
                       factual: float, counterfactual: list[float],
                       reference_ids: list[str], config: dict,
                       metadata: dict[str, Any] | None = None) -> EvidenceResult:
    values = np.asarray(counterfactual, dtype=float)
    deltas = factual - values
    mean = float(deltas.mean()) if len(deltas) else 0.0
    std = float(deltas.std()) if len(deltas) else 0.0
    stability = abs(mean) / (abs(mean) + std + 1e-8)
    consistency = (float(np.mean(np.sign(deltas) == np.sign(mean)))
                   if len(deltas) and abs(mean) > 1e-12 else 0.0)
    ac = config["attribution"]
    accepted = (abs(mean) >= float(ac["contribution_threshold"])
                and stability >= float(ac["stability_threshold"])
                and consistency >= float(ac.get("direction_threshold", 0.75)))
    return EvidenceResult(
        evidence_id=evidence_id, target=target, level=level,
        factual_risk=float(factual), counterfactual_risks=values.tolist(),
        deltas=deltas.tolist(), mean_delta=mean, std_delta=std,
        stability=float(stability), direction_consistency=consistency,
        direction="risk_supporting" if mean > 0 else "risk_protective" if mean < 0 else "neutral",
        accepted=bool(accepted), reference_ids=reference_ids,
        metadata=metadata or {},
    )


class MatchedReferenceBank:
    """Deterministic nearest non-FLD references from the training split."""

    def __init__(self, dataset, config: dict):
        self.dataset, self.cfg = dataset, config
        self.indices = [i for i, y in enumerate(dataset.df["FLD"].tolist()) if int(float(y)) == 0]
        if not self.indices:
            raise ValueError("The training reference bank contains no non-FLD samples.")

    def select(self, sample: dict, k: int | None = None) -> list[int]:
        """供体匹配只读生理值:直接用 phys_raw_all 数组,不再逐个解码供体
        图像/掩码(旧实现在 128 核机器上每例触发数千次微型 CPU 张量操作,
        曾与 GPU 前向交错后发生 OMP 线程活锁;数值与旧实现逐位一致)。"""
        k = int(k or self.cfg["attribution"]["donors"])
        target = sample["physiology_raw"].detach().cpu().numpy()
        target_gender, target_age, target_bmi = target[0], target[1], target[len(RAW_PHYS)]
        age_window = float(self.cfg["attribution"].get("match_age_window", 5))
        raw_all = self.dataset.phys_raw_all
        candidates: list[tuple[float, int]] = []
        for idx in self.indices:
            raw = raw_all[idx].astype(np.float32)
            if raw[0] != target_gender:
                continue
            age_penalty = abs(raw[1] - target_age)
            if age_penalty > age_window * 4:
                continue
            bmi_penalty = abs(raw[len(RAW_PHYS)] - target_bmi)
            distance = age_penalty / max(age_window, 1.0) + bmi_penalty / 4.0
            candidates.append((float(distance), idx))
        if not candidates:
            mean = np.asarray(self.dataset.stats["mean"], dtype=np.float32)
            std = np.asarray(self.dataset.stats["std"], dtype=np.float32)
            norm_all = (raw_all.astype(np.float32) - mean) / std
            target_norm = sample["physiology"].numpy()
            candidates = [(float(np.linalg.norm(norm_all[i] - target_norm)), i)
                          for i in self.indices]
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [idx for _, idx in candidates[:min(k, len(candidates))]]


@dataclass
class CounterfactualSession:
    sample: dict
    factual_output: dict
    factual_risk: float
    reference_indices: list[int]
    reference_samples: list[dict]
    cache: dict[str, Any]


class CounterfactualToolbox:
    """Four independent tools. No exhaustive attribution is performed implicitly."""

    def __init__(self, model, calibrator, training_dataset, config: dict[str, Any], device):
        self.model, self.calibrator = model, calibrator
        self.dataset, self.cfg, self.device = training_dataset, config, device
        self.bank = MatchedReferenceBank(training_dataset, config)
        self.tta = bool(self.cfg.get("agent", {}).get("tta", True))
        self.layout = concept_vector_layout()
        self.offsets, offset = {}, 0
        for spec in self.layout:
            self.offsets[spec["key"]] = offset
            offset += spec["classes"]

    def _batch(self, value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0).to(self.device)

    def _calibrate(self, risk: float) -> float:
        return float(self.calibrator.transform([risk])[0])

    @torch.no_grad()
    def _forward_pass(self, sample: dict) -> dict:
        return self.model(self._batch(sample["image"]), self._batch(sample["body_mask"]),
                          self._batch(sample["region_masks"]), self._batch(sample["boxes"]),
                          self._batch(sample["physiology"]), self._batch(sample["phys_missing"]))

    @torch.no_grad()
    def forward_tta(self, sample: dict) -> tuple[dict, list[dict]]:
        """官方推理协议:原图 + 水平翻转两次前向,风险输出按 logit 平均
        (与 engine.predict_split(tta=True) 同语义);concept_probs 等非风险
        输出取原图 pass。返回 (平均输出, 各 pass 原始输出)。"""
        out = self._forward_pass(sample)
        passes = [out]
        if self.tta:
            out2 = self._forward_pass(flip_sample(sample, list(self.model.regions)))
            passes.append(out2)
            avg = dict(out)
            for key in TTA_RISK_KEYS:
                if key in out and key in out2:
                    p = torch.clamp(torch.stack([out[key].float(), out2[key].float()]),
                                    1e-6, 1 - 1e-6)
                    avg[key] = torch.sigmoid(torch.logit(p).mean(0))
            out = avg
        return out, passes

    @torch.no_grad()
    def _forward(self, sample: dict) -> tuple[dict, float]:
        out, _ = self.forward_tta(sample)
        return out, self._calibrate(float(out["fused_risk"][0]))

    @torch.no_grad()
    def begin(self, sample: dict, factual_output: dict | None = None,
              factual_passes: list[dict] | None = None) -> CounterfactualSession:
        if factual_output is None:
            factual_output, factual_passes = self.forward_tta(sample)
        elif factual_passes is None:
            factual_passes = [factual_output]
        factual = self._calibrate(float(factual_output["fused_risk"][0]))
        ids = self.bank.select(sample)
        refs = [self.dataset[i] for i in ids]
        # 逐 pass 缓存图像侧证据(gap_feat/mask_rate/gap_feats/visual_global):
        # 概念/生理反事实重算时每个 pass 带着自己的视觉证据各走一遍,再 logit 平均,
        # 与官方 TTA 协议一致;概念反事实时图像证据保持事实分支值,
        # 否则零向量进 TDF 头会让 gap_logit 失真
        passes = []
        for o in factual_passes:
            passes.append({"concept_probs": o["concept_probs"].detach(),
                           "gap_feat": o["gap_feat"].detach(),
                           "mask_rate": o["mask_rate"].detach(),
                           "gap_feats": (o["gap_feats"].detach()
                                         if o.get("gap_feats") is not None else None),
                           "visual_global": (o["visual_global"].detach()
                                             if o.get("visual_global") is not None else None)})
        cache = {"passes": passes}
        return CounterfactualSession(sample, factual_output, factual, ids, refs, cache)

    @torch.no_grad()
    def _risk_from_latent(self, session: CounterfactualSession, concepts,
                          physiology: torch.Tensor) -> float:
        """concepts 可为张量(所有 pass 共用)或 pass 缓存 → 张量的 callable
        (逐 pass 基于各自事实概念向量做替换)。每个 pass 各自前向后按
        logit 平均,再统一校准 —— 与官方 TTA 协议一致。"""
        missing = torch.zeros_like(physiology).to(self.device)
        risks = []
        for pc in session.cache["passes"]:
            c = concepts(pc) if callable(concepts) else concepts
            out = self.model.risk_from_concepts(c, physiology.to(self.device), missing,
                                                visual_global=pc["visual_global"],
                                                gap_feat=pc["gap_feat"],
                                                mask_rate=pc["mask_rate"],
                                                gap_feats=pc["gap_feats"])
            risks.append(float(out["fused_risk"][0]))
        return self._calibrate(probs_logit_mean(risks))

    @staticmethod
    def _aligned_patch(donor_image: torch.Tensor, donor_mask: torch.Tensor,
                       target_mask: torch.Tensor) -> torch.Tensor:
        def bounds(mask):
            ys, xs = torch.where(mask > 0.5)
            return None if not len(ys) else (int(ys.min()), int(ys.max()) + 1,
                                              int(xs.min()), int(xs.max()) + 1)
        db, tb = bounds(donor_mask), bounds(target_mask)
        canvas = torch.zeros_like(donor_image)
        if db is None or tb is None:
            return canvas
        dy1, dy2, dx1, dx2 = db; ty1, ty2, tx1, tx2 = tb
        crop = donor_image[:, dy1:dy2, dx1:dx2].unsqueeze(0)
        crop = F.interpolate(crop, size=(ty2 - ty1, tx2 - tx1), mode="bilinear",
                             align_corners=False)[0]
        canvas[:, ty1:ty2, tx1:tx2] = crop
        return canvas

    @torch.no_grad()
    def verify_region(self, session: CounterfactualSession, region: str) -> EvidenceResult:
        ridx = self.cfg["data"]["regions"].index(region)
        target_mask = session.sample["region_masks"][ridx]
        risks, ids = [], []
        for donor in session.reference_samples:
            aligned = self._aligned_patch(donor["image"], donor["region_masks"][ridx], target_mask)
            cf = dict(session.sample)
            cf["image"] = session.sample["image"] * (1 - target_mask) + aligned * target_mask
            _, risk = self._forward(cf)
            risks.append(risk); ids.append(str(donor["id"]))
        return summarize_evidence(f"region:{region}", region, "region", session.factual_risk,
                                  risks, ids, self.cfg)

    @torch.no_grad()
    def verify_concept(self, session: CounterfactualSession, concept: str) -> EvidenceResult:
        if concept not in self.offsets:
            raise KeyError(f"Unknown concept: {concept}")
        spec = next(x for x in self.layout if x["key"] == concept)
        offset, n = self.offsets[concept], spec["classes"]
        risks, ids = [], []
        for donor in session.reference_samples:
            donor_out, _ = self._forward(donor)
            replacement = donor_out["concept_probs"][:, offset:offset + n].detach()
            if int(replacement.argmax(-1)[0]) != int(spec["normal"]):
                replacement = torch.zeros_like(replacement)
                replacement[:, int(spec["normal"])] = 1.0

            def replace(pc, _r=replacement):
                cf = pc["concept_probs"].clone()
                cf[:, offset:offset + n] = _r
                return cf
            risk = self._risk_from_latent(session, replace,
                                          self._batch(session.sample["physiology"]))
            risks.append(risk); ids.append(str(donor["id"]))
        concept_spec = CONCEPT_BY_KEY[concept]
        return summarize_evidence(
            f"concept:{concept}", concept, "concept", session.factual_risk, risks, ids, self.cfg,
            {"name": concept_spec["cn"],
             "normal_state": concept_spec["classes"][concept_spec["normal"]]},
        )

    def _normalized_physiology(self, raw_eight: np.ndarray) -> torch.Tensor:
        full = np.concatenate([raw_eight, compute_derived(raw_eight)[0]]).astype(np.float32)
        mean = np.asarray(self.dataset.stats["mean"], dtype=np.float32)
        std = np.asarray(self.dataset.stats["std"], dtype=np.float32)
        return torch.tensor((full - mean) / std, dtype=torch.float32).unsqueeze(0)

    @torch.no_grad()
    def verify_physiology(self, session: CounterfactualSession, group: str) -> EvidenceResult:
        groups = self.cfg["attribution"]["phys_groups"]
        if group not in groups:
            raise KeyError(f"Unknown physiology group: {group}")
        variables = list(groups[group]["raw"])
        indices = [RAW_PHYS.index(name) for name in variables]
        factual_raw = session.sample["physiology_raw"].numpy()[:len(RAW_PHYS)].copy()
        risks, ids = [], []
        for donor in session.reference_samples:
            donor_raw = donor["physiology_raw"].numpy()[:len(RAW_PHYS)]
            cf_raw = factual_raw.copy(); cf_raw[indices] = donor_raw[indices]
            risk = self._risk_from_latent(session, lambda pc: pc["concept_probs"],
                                          self._normalized_physiology(cf_raw))
            risks.append(risk); ids.append(str(donor["id"]))
        return summarize_evidence(
            f"physiology:{group}", group, "physiology", session.factual_risk,
            risks, ids, self.cfg, {"name": groups[group]["cn"], "variables": variables},
        )

    @torch.no_grad()
    def verify_interaction(self, session: CounterfactualSession, concept: str,
                           physiology_group: str) -> EvidenceResult:
        concept_result = self.verify_concept(session, concept)
        physiology_result = self.verify_physiology(session, physiology_group)
        spec = next(x for x in self.layout if x["key"] == concept)
        offset, n = self.offsets[concept], spec["classes"]
        variables = self.cfg["attribution"]["phys_groups"][physiology_group]["raw"]
        indices = [RAW_PHYS.index(name) for name in variables]
        factual_raw = session.sample["physiology_raw"].numpy()[:len(RAW_PHYS)].copy()
        joint_risks, ids = [], []
        for donor in session.reference_samples:
            donor_out, _ = self._forward(donor)
            replacement = donor_out["concept_probs"][:, offset:offset + n].detach()
            if int(replacement.argmax(-1)[0]) != int(spec["normal"]):
                replacement = torch.zeros_like(replacement); replacement[:, int(spec["normal"])] = 1

            def replace(pc, _r=replacement):
                cf = pc["concept_probs"].clone()
                cf[:, offset:offset + n] = _r
                return cf
            donor_raw = donor["physiology_raw"].numpy()[:len(RAW_PHYS)]
            cf_raw = factual_raw.copy(); cf_raw[indices] = donor_raw[indices]
            joint_risk = self._risk_from_latent(session, replace,
                                                self._normalized_physiology(cf_raw))
            joint_risks.append(joint_risk); ids.append(str(donor["id"]))
        interaction_risks = interaction_counterfactual_risks(
            session.factual_risk, concept_result.deltas, physiology_result.deltas, joint_risks)
        return summarize_evidence(
            f"interaction:{concept}:{physiology_group}", f"{concept}×{physiology_group}",
            "interaction", session.factual_risk, interaction_risks, ids, self.cfg,
            {"concept": concept, "physiology_group": physiology_group,
             "joint_counterfactual_risks": joint_risks},
        )

    def execute(self, session: CounterfactualSession, action) -> EvidenceResult:
        if action.action == "verify_region":
            return self.verify_region(session, str(action.target))
        if action.action == "verify_concept":
            return self.verify_concept(session, str(action.target))
        if action.action == "verify_physiology":
            return self.verify_physiology(session, str(action.target))
        if action.action == "verify_interaction":
            return self.verify_interaction(session, str(action.target), str(action.secondary_target))
        raise ValueError(f"Action {action.action!r} is not a counterfactual tool action")


def build_sample(dataset, index: int) -> dict:
    sample = dict(dataset[index])
    row = dataset.df.iloc[index]
    sample["age"] = float(row["Age"])
    sample["gender_raw"] = float(row["GenderRaw"])
    sample["bmi"] = float(dataset.phys_raw_all[index, len(RAW_PHYS)])
    return sample


CounterfactualAttributor = CounterfactualToolbox
