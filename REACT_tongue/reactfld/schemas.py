"""Typed contracts shared by the specialist tools and the evidence agent.

动作空间严格限制为 5 个(论文版 §24):
verify_region / verify_concept / verify_physiology / verify_interaction / stop
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ActionType = Literal[
    "verify_region", "verify_concept", "verify_physiology",
    "verify_interaction", "stop",
]


@dataclass
class RiskState:
    tongue_risk: float
    physiology_risk: float
    interaction_risk: float
    fused_risk_raw: float
    calibrated_risk: float
    tongue_uncertainty: float
    physiology_uncertainty: float
    modality_disagreement: float
    gate: dict[str, float]


@dataclass
class Candidate:
    target: str
    level: Literal["region", "concept", "physiology", "interaction"]
    priority: float
    reason_codes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    action: ActionType
    target: str | None = None
    secondary_target: str | None = None
    reason_code: str = "NONE"
    expected_information: str = ""


@dataclass
class EvidenceResult:
    evidence_id: str
    target: str
    level: str
    factual_risk: float
    counterfactual_risks: list[float]
    deltas: list[float]
    mean_delta: float
    std_delta: float
    stability: float
    direction_consistency: float
    direction: str
    accepted: bool
    reference_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    patient_id: str
    risk: RiskState
    concepts: dict[str, Any]
    candidates: list[Candidate]
    evidence: list[EvidenceResult] = field(default_factory=list)
    rejected_targets: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    remaining_budget: int = 8
    step: int = 0
    stop_reason: str | None = None

    def evidence_for(self, target: str, level: str | None = None) -> EvidenceResult | None:
        for item in self.evidence:
            if item.target == target and (level is None or item.level == level):
                return item
        return None

    def serializable(self) -> dict[str, Any]:
        return asdict(self)
