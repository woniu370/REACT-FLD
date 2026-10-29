"""结构化最终报告(论文版 §28)。

报告为一次性模板:校准风险 + 概念预测 + 生理异常 + 已接受反事实证据。
无声明验证闭环:未经反事实验证的因素不出现在"模型主要影响因素"中,
所有 ΔRisk 均为模型级影响,不构成医学因果结论。
"""
from __future__ import annotations

from typing import Any

from .concepts import CONCEPT_BY_KEY

PHYS_GROUPS_CN = {
    "general_obesity": "一般肥胖", "abdominal_obesity": "腹型肥胖",
    "blood_pressure": "血压状态", "basics": "基础因素",
}


class StructuredReportBuilder:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config

    def _risk_level(self, value: float) -> str:
        levels = self.cfg["report"]["risk_levels"]
        return "较高" if value >= float(levels["high"]) else "中等" if value >= float(levels["low"]) else "较低"

    def render(self, payload: dict[str, Any]) -> str:
        risk = payload["risk"]
        concepts: dict[str, dict] = payload.get("concepts", {})
        flags = payload.get("physiology_flags", [])
        accepted = sorted((e for e in payload.get("accepted_evidence", [])
                           if e.get("direction") == "risk_supporting"),
                          key=lambda e: abs(e.get("mean_delta", 0.0)), reverse=True)

        lines = [f"# FLD风险评估报告(患者 {payload['patient_id']})", "",
                 f"FLD风险:{risk['calibrated_risk']:.1%}({self._risk_level(risk['calibrated_risk'])}风险)"]

        # 主要舌象表现:预测为非正常类的概念
        lines += ["", "主要舌象表现:"]
        abnormal = [v for v in concepts.values() if v["predicted"] != v["normal_state"]]
        if abnormal:
            for v in abnormal:
                lines.append(f"- {v['name']}:{v['predicted']}(置信度 {v['probability']:.0%})")
        else:
            lines.append("- 未发现明显异常舌象表现")

        # 主要生理异常
        lines += ["", "主要生理异常:"]
        if flags:
            for flag in flags:
                if flag["factor"] == "blood_pressure":
                    sbp, dbp = flag["value"]
                    lines.append(f"- 血压:{sbp:.0f}/{dbp:.0f} mmHg,{flag['status']}")
                else:
                    lines.append(f"- {flag['factor']}:{flag['value']:.2f},{flag['status']}")
        else:
            lines.append("- 未发现明显生理指标异常")

        # 模型主要影响因素:仅列已接受的反事实证据(论文版 §25 Evidence Bank)
        lines += ["", "模型主要影响因素:"]
        if accepted:
            for e in accepted:
                name = e.get("metadata", {}).get("name", e.get("target", ""))
                delta = e["mean_delta"]
                verb = "下降" if delta > 0 else "上升"
                lines.append(f"- {name}:替换为正常参考后,风险{verb}约{abs(delta) * 100:.0f}个百分点"
                             f"(稳定性 {e['stability']:.2f},方向一致率 {e['direction_consistency']:.0%})")
        else:
            lines.append("- 本次未获得满足接受标准的反事实证据")

        # 综合提示
        top = "、".join(e.get("metadata", {}).get("name", e.get("target", "")) for e in accepted[:3])
        summary = (f"模型综合舌象与生理信息判断当前FLD风险{self._risk_level(risk['calibrated_risk'])}。"
                   + (f"{top}是当前模型高风险判断中影响较明显的因素。" if top else ""))
        lines += ["", "综合提示:", summary, "",
                  "说明:上述风险变化反映各因素对模型预测结果的影响,不代表临床因果关系。",
                  "本报告仅用于辅助风险评估,不能替代临床诊断。"]
        return "\n".join(lines)


def generate_report(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    builder = StructuredReportBuilder(config)
    return {"report": builder.render(payload)}


def evidence_name(evidence: dict[str, Any]) -> str:
    if evidence.get("level") == "interaction":
        meta = evidence.get("metadata", {})
        ccn = CONCEPT_BY_KEY.get(meta.get("concept", ""), {}).get("cn", meta.get("concept", ""))
        pcn = PHYS_GROUPS_CN.get(meta.get("physiology_group", ""), meta.get("physiology_group", ""))
        return f"{ccn}×{pcn}交互"
    return evidence.get("metadata", {}).get("name", evidence.get("target", ""))
