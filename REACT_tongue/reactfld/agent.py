"""Risk-state-driven evidence acquisition agent(论文版 §20-§27)。

动作空间严格限制为 5 个:
verify_region / verify_concept / verify_physiology / verify_interaction / stop
候选白名单由确定性规则生成;LLM 规划器(Qwen3.5-4B,enable_thinking=False,
贪心解码 ≤512 token,strip <think>)只做白名单内的动作选择,
非法输出自动回退确定性规划器。无报告验证闭环、无病机知识模块。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from typing import Any

import torch

from .attribution import CounterfactualToolbox, build_sample
from .concepts import CONCEPT_BY_KEY, concept_vector_layout
from .data import PHYS_NAMES
from .schemas import AgentAction, AgentState, Candidate, RiskState


REASON_CODES = {
    "HIGH_TONGUE_UNCERTAINTY", "HIGH_PHYS_UNCERTAINTY", "MODALITY_CONFLICT",
    "HIGH_FUSED_RISK", "BOUNDARY_RISK", "TOP_CANDIDATE",
    "EVIDENCE_SUFFICIENT", "BUDGET_EXHAUSTED", "NO_CANDIDATE",
}


def _entropy(probabilities: torch.Tensor) -> float:
    p = probabilities.detach().float().clamp_min(1e-9)
    return float(-(p * p.log()).sum())


def _physiology_flags(raw: dict[str, float], gender: float) -> list[dict[str, Any]]:
    flags = []
    bmi, whr, sbp, dbp = raw["BMI"], raw["WHR"], raw["SBP"], raw["DBP"]
    whr_limit = 0.85 if gender == 2.0 else 0.90
    if bmi >= 24:
        flags.append({"factor": "BMI", "value": bmi, "status": "肥胖" if bmi >= 28 else "超重"})
    if whr >= whr_limit:
        flags.append({"factor": "WHR", "value": whr, "status": "腹型肥胖"})
    if sbp >= 140 or dbp >= 90:
        flags.append({"factor": "blood_pressure", "value": [sbp, dbp], "status": "血压升高"})
    return flags


class CandidateGenerator:
    """Deterministic, reproducible candidate whitelist before LLM planning(论文版 §21)."""

    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self.layout = concept_vector_layout()
        self.offsets, offset = {}, 0
        for spec in self.layout:
            self.offsets[spec["key"]] = offset
            offset += spec["classes"]

    def concepts(self, output: dict) -> dict[str, dict[str, Any]]:
        result = {}
        vector = output["concept_probs"][0]
        for spec in self.layout:
            start, n = self.offsets[spec["key"]], spec["classes"]
            probs = vector[start:start+n]
            pred = int(probs.argmax())
            abnormal = 1.0 - float(probs[int(spec["normal"])])
            result[spec["key"]] = {
                "name": spec["cn"], "region": spec["region"],
                "predicted": spec["class_names"][pred], "predicted_index": pred,
                "probability": float(probs[pred]), "abnormal_probability": abnormal,
                "entropy": _entropy(probs), "normal_state": spec["class_names"][int(spec["normal"])],
            }
        return result

    def generate(self, concepts: dict[str, dict[str, Any]], risk: RiskState,
                 physiological_flags: list[dict]) -> list[Candidate]:
        ac = self.cfg["agent"]
        reasons = []
        if risk.tongue_uncertainty >= float(ac["tongue_uncertainty_threshold"]):
            reasons.append("HIGH_TONGUE_UNCERTAINTY")
        if risk.physiology_uncertainty >= float(ac["physiology_uncertainty_threshold"]):
            reasons.append("HIGH_PHYS_UNCERTAINTY")
        if risk.modality_disagreement >= float(ac["conflict_threshold"]):
            reasons.append("MODALITY_CONFLICT")
        if risk.calibrated_risk >= float(ac["high_risk_threshold"]):
            reasons.append("HIGH_FUSED_RISK")
        if abs(risk.calibrated_risk - float(ac["decision_threshold"])) <= float(ac["boundary_margin"]):
            reasons.append("BOUNDARY_RISK")
        if not reasons:
            reasons = ["TOP_CANDIDATE"]

        candidates: list[Candidate] = []
        for key, value in concepts.items():
            # Abnormality and normalized entropy proxy the value of verification.
            n_cls = len(CONCEPT_BY_KEY[key]["classes"])
            entropy_norm = value["entropy"] / max(math.log(n_cls), 1e-6)
            priority = 0.75 * value["abnormal_probability"] + 0.25 * entropy_norm
            if priority >= float(ac["candidate_threshold"]):
                candidates.append(Candidate(key, "concept", priority, list(reasons),
                                            {"region": value["region"], "name": value["name"]}))

        for region in self.cfg["data"]["regions"]:
            values = [v for v in concepts.values() if v["region"] == region]
            if values:
                priority = max(v["abnormal_probability"] for v in values)
                candidates.append(Candidate(region, "region", 0.9 * priority, list(reasons)))

        phys_groups = self.cfg["attribution"]["phys_groups"]
        abnormal_names = {f["factor"] for f in physiological_flags}
        for group, spec in phys_groups.items():
            abnormal = len(abnormal_names.intersection(set(spec["raw"] + spec.get("derived", []))))
            priority = 0.35 + 0.25 * abnormal
            if "HIGH_PHYS_UNCERTAINTY" in reasons or "MODALITY_CONFLICT" in reasons or abnormal:
                candidates.append(Candidate(group, "physiology", min(priority, 1.0), list(reasons),
                                            {"name": spec["cn"], "variables": spec["raw"]}))
        candidates.sort(key=lambda item: (-item.priority, item.level, item.target))
        return candidates[:int(ac["max_candidates"])]


class DeterministicPlanner:
    """Safe reproducible planner and automatic fallback when Qwen is unavailable(论文版 §24)."""

    @staticmethod
    def plan(state: AgentState) -> AgentAction:
        if state.remaining_budget <= 0:
            return AgentAction("stop", reason_code="BUDGET_EXHAUSTED")
        tested = {(e.level, e.target) for e in state.evidence}
        for candidate in state.candidates:
            if (candidate.level, candidate.target) in tested:
                continue
            if candidate.level == "interaction":
                concept, _, group = candidate.target.partition("×")
                return AgentAction("verify_interaction", concept, group, candidate.reason_codes[0],
                                   "Test non-additive cross-modal contribution")
            action = {"region": "verify_region", "concept": "verify_concept",
                      "physiology": "verify_physiology"}[candidate.level]
            return AgentAction(action, candidate.target, reason_code=candidate.reason_codes[0],
                               expected_information="Quantify calibrated risk change and stability")
        accepted_c = [e for e in state.evidence if e.accepted and e.level == "concept"]
        accepted_p = [e for e in state.evidence if e.accepted and e.level == "physiology"]
        tested_interactions = {e.target for e in state.evidence if e.level == "interaction"}
        if accepted_c and accepted_p:
            pair = f"{accepted_c[0].target}×{accepted_p[0].target}"
            if pair not in tested_interactions:
                return AgentAction("verify_interaction", accepted_c[0].target, accepted_p[0].target,
                                   "MODALITY_CONFLICT", "Test non-additive cross-modal contribution")
        return AgentAction("stop", reason_code="NO_CANDIDATE")


class QwenToolPlanner:
    """Constrained Qwen3.5-4B planner. Invalid output always falls back deterministically.

    解码配置(论文版要求): enable_thinking=False,贪心(do_sample=False)≤512 token,
    输出 strip <think> 块;动作必须 ∈ 白名单且 reason_code 合法。
    """

    def __init__(self, config: dict[str, Any], fallback: DeterministicPlanner):
        self.cfg, self.fallback, self.available = config, fallback, False
        self.model = self.tokenizer = None
        if not config["agent"]["llm"].get("enabled", False):
            return
        try:
            from transformers import AutoConfig, AutoTokenizer
            path = config["agent"]["llm"]["model_path"]
            archs = ",".join(getattr(AutoConfig.from_pretrained(path, trust_remote_code=True),
                                     "architectures", None) or [])
            if "CausalLM" in archs:
                from transformers import AutoModelForCausalLM as AutoModel
            else:
                # 多模态模型(如 Qwen3.5-4B,vision-language 架构)需要 ImageTextToText 类
                from transformers import AutoModelForImageTextToText as AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                path, dtype="auto", device_map="auto", trust_remote_code=True)
            self.available = True
        except Exception as exc:
            self.load_error = str(exc)

    def _allowed(self, state: AgentState) -> list[dict]:
        tested = {(e.level, e.target) for e in state.evidence}
        actions = []
        mapping = {"region": "verify_region", "concept": "verify_concept",
                   "physiology": "verify_physiology"}
        for c in state.candidates:
            if (c.level, c.target) in tested:
                continue
            if c.level == "interaction":
                concept, _, group = c.target.partition("×")
                actions.append({"action": "verify_interaction", "target": concept,
                                "secondary_target": group, "priority": round(c.priority, 4),
                                "reason_codes": c.reason_codes})
            else:
                actions.append({"action": mapping[c.level], "target": c.target,
                                "priority": round(c.priority, 4), "reason_codes": c.reason_codes})
        accepted_c = [e.target for e in state.evidence if e.accepted and e.level == "concept"]
        accepted_p = [e.target for e in state.evidence if e.accepted and e.level == "physiology"]
        for c in accepted_c[:2]:
            for p in accepted_p[:2]:
                actions.append({"action": "verify_interaction", "target": c,
                                "secondary_target": p, "reason_codes": ["MODALITY_CONFLICT"]})
        actions.append({"action": "stop",
                        "reason_codes": ["EVIDENCE_SUFFICIENT", "NO_CANDIDATE", "BUDGET_EXHAUSTED"]})
        return actions

    def plan(self, state: AgentState) -> AgentAction:
        if not self.available:
            return self.fallback.plan(state)
        allowed = self._allowed(state)
        prompt = {
            "task": "Select exactly one allowed action. Return JSON only.",
            "risk": asdict(state.risk),
            "accepted_evidence": [e.to_dict() for e in state.evidence if e.accepted],
            "rejected_targets": state.rejected_targets,
            "remaining_budget": state.remaining_budget,
            "allowed_actions": allowed,
            "schema": {"action": "string", "target": "string|null",
                       "secondary_target": "string|null", "reason_code": "string",
                       "expected_information": "string"},
        }
        messages = [{"role": "system", "content": "You are a constrained medical evidence acquisition controller."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                     pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        decoded = re.sub(r"<think>.*?</think>\s*", "", decoded, flags=re.S)
        try:
            obj = json.loads(re.search(r"\{.*\}", decoded, re.S).group())
            action = AgentAction(**obj)
            key = (action.action, action.target, action.secondary_target)
            allowed_keys = {(x["action"], x.get("target"), x.get("secondary_target")) for x in allowed}
            if key not in allowed_keys or action.reason_code not in REASON_CODES:
                raise ValueError("Planner selected a non-whitelisted action")
            return action
        except Exception:
            return self.fallback.plan(state)


class EvidenceAcquisitionAgent:
    def __init__(self, model, calibrator, training_dataset, config: dict[str, Any], device):
        self.model, self.calibrator, self.cfg, self.device = model, calibrator, config, device
        self.toolbox = CounterfactualToolbox(model, calibrator, training_dataset, config, device)
        self.generator = CandidateGenerator(config)
        self.fallback = DeterministicPlanner()
        self.planner = QwenToolPlanner(config, self.fallback)

    def _risk_state(self, output: dict) -> RiskState:
        pt, pp = float(output["tongue_risk"][0]), float(output["phys_risk"][0])
        raw = float(output["fused_risk"][0])
        calibrated = float(self.calibrator.transform([raw])[0])
        interaction = float(output["tp_risk"][0])
        gate = {name: float(value) for name, value in
                zip(("tongue", "physiology", "interaction"), output["gate"][0])}
        return RiskState(pt, pp, interaction, raw, calibrated,
                         float(output["tongue_uncertainty"][0]),
                         float(output["phys_uncertainty"][0]), abs(pt - pp), gate)

    def _sufficient(self, state: AgentState) -> bool:
        cfg = self.cfg["agent"]
        accepted = [e for e in state.evidence if e.accepted and e.direction == "risk_supporting"]
        if len(accepted) < int(cfg["min_accepted_evidence"]):
            return False
        mean_stability = sum(e.stability for e in accepted) / len(accepted)
        if mean_stability < float(cfg["evidence_sufficiency_threshold"]):
            return False
        if state.risk.modality_disagreement >= float(cfg["conflict_threshold"]):
            tongue = any(e.level in ("region", "concept") for e in accepted)
            physiology = any(e.level == "physiology" for e in accepted)
            return tongue and physiology
        return True

    @staticmethod
    def _has_open_candidates(state: AgentState) -> bool:
        tested = {(e.level, e.target) for e in state.evidence}
        return any((c.level, c.target) not in tested for c in state.candidates)

    @staticmethod
    def _snapshot(state: AgentState) -> dict[str, Any]:
        return {"remaining_budget": state.remaining_budget,
                "n_accepted": sum(e.accepted for e in state.evidence),
                "modality_disagreement": state.risk.modality_disagreement}

    @torch.no_grad()
    def initialize(self, dataset, index: int) -> tuple[AgentState, Any, dict]:
        sample = build_sample(dataset, index)
        # 事实前向走 toolbox(官方推理协议:原图+翻转 TTA logit 平均),
        # begin 的逐 pass 视觉证据缓存同时供反事实重算使用
        session = self.toolbox.begin(sample)
        output = session.factual_output
        risk = self._risk_state(output)
        concepts = self.generator.concepts(output)
        class_probs: dict[str, dict[str, float]] = {}
        vector = output["concept_probs"][0]
        for spec in self.generator.layout:
            start, n = self.generator.offsets[spec["key"]], spec["classes"]
            probs = vector[start:start + n]
            class_probs[spec["key"]] = {name: float(probs[i])
                                        for i, name in enumerate(spec["class_names"])}
        raw = {name: float(value) for name, value in zip(PHYS_NAMES, sample["physiology_raw"])}
        flags = _physiology_flags(raw, sample["gender_raw"])
        candidates = self.generator.generate(concepts, risk, flags)
        state = AgentState(sample["id"], risk, concepts, candidates,
                           remaining_budget=int(self.cfg["agent"]["max_tool_calls"]))
        context = {"sample": sample, "output": output, "physiology": raw,
                   "physiology_flags": flags, "concept_class_probs": class_probs}
        return state, session, context

    def run(self, dataset, index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        state, session, context = self.initialize(dataset, index)

        def stop_record(reason: str) -> dict[str, Any]:
            return {"step": state.step, "state_before": self._snapshot(state),
                    "action": asdict(AgentAction("stop", reason_code=reason)),
                    "observation": {"stopped": True}}

        while True:
            # 论文版 §26 停止条件:EVIDENCE_SUFFICIENT ∨ BUDGET_EXHAUSTED ∨ NO_CANDIDATE
            if self._sufficient(state) or not self._has_open_candidates(state):
                state.stop_reason = ("EVIDENCE_SUFFICIENT" if self._sufficient(state)
                                     else "BUDGET_EXHAUSTED" if state.remaining_budget <= 0
                                     else "NO_CANDIDATE")
                state.history.append(stop_record(state.stop_reason))
                break
            if state.remaining_budget <= 0:
                state.stop_reason = "BUDGET_EXHAUSTED"
                state.history.append(stop_record(state.stop_reason))
                break

            action = self.planner.plan(state)
            record = {"step": state.step, "state_before": self._snapshot(state),
                      "action": asdict(action), "cost": 1}
            if action.action == "stop":
                state.stop_reason = action.reason_code
                record["observation"] = {"stopped": True}
                state.history.append(record)
                break
            try:
                evidence = self.toolbox.execute(session, action)
                state.evidence.append(evidence)
                if not evidence.accepted:
                    state.rejected_targets.append(evidence.target)
                record["observation"] = evidence.to_dict()
            except Exception as exc:
                state.rejected_targets.append(str(action.target))
                record["observation"] = {"error": type(exc).__name__, "message": str(exc)}
            state.remaining_budget -= 1
            state.step += 1
            state.history.append(record)

        payload = {
            "patient_id": state.patient_id,
            "risk": asdict(state.risk),
            "concepts": state.concepts,
            "physiology": context["physiology"],
            "physiology_flags": context["physiology_flags"],
            "concept_class_probs": context["concept_class_probs"],
            "evidence": [e.to_dict() for e in state.evidence],
            "accepted_evidence": [e.to_dict() for e in state.evidence if e.accepted],
            "stop_reason": state.stop_reason,
            "remaining_budget": state.remaining_budget,
            "planner": "qwen_llm" if self.planner.available else "deterministic_fallback",
            "model_level_effect_disclaimer": True,
        }
        return payload, state.history


EvidenceAgent = EvidenceAcquisitionAgent
