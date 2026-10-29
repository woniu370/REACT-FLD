"""Command-line interface for training and agentic inference.

LLM 仅用于规划器(--qwen-planner,Qwen3.5-4B);报告为确定性最终模板(论文版 §28)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import EvidenceAcquisitionAgent
from .data import load_splits
from .engine import load_checkpoint, train
from .report import generate_report
from .utils import load_config, resolve_device, save_json, seed_everything


def _load_existing_result(output: Path, patient_id: str) -> dict | None:
    """Return the saved result for a case, or None if absent/corrupt (中断续跑)。"""
    evidence_path = output / f"{patient_id}_evidence.json"
    if not evidence_path.exists() or not (output / f"{patient_id}_report.md").exists():
        return None
    try:
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_agent(checkpoint: str, config: dict, config_path: str, split: str):
    device = resolve_device(config["device"])
    model, _, calibrator = load_checkpoint(checkpoint, device)
    train_ds, val_ds, test_ds = load_splits(config, config_path)
    targets = {"train": train_ds, "val": val_ds, "test": test_ds}
    # The reference bank is always train_ds, irrespective of the target split.
    agent = EvidenceAcquisitionAgent(model, calibrator, train_ds, config, device)
    return agent, targets[split]


def main():
    parser = argparse.ArgumentParser(description="REACT-FLD agentic evidence acquisition")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--output", default="runs/default")

    run = sub.add_parser("infer")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--split", choices=["train", "val", "test"], default="test")
    run.add_argument("--index", type=int, default=0)
    run.add_argument("--output", default="runs/inference")
    run.add_argument("--qwen-planner", action="store_true")

    batch = sub.add_parser("infer-batch")
    batch.add_argument("--checkpoint", required=True)
    batch.add_argument("--split", choices=["val", "test"], default="test")
    batch.add_argument("--limit", type=int, default=0)
    batch.add_argument("--output", default="runs/inference_batch")
    batch.add_argument("--qwen-planner", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in ("infer", "infer-batch"):
        import torch
        # 大核数机器上默认 CPU 线程数(=物理核)与 GPU 前向交错会触发 OMP 活锁,显式限流
        torch.set_num_threads(8)
    seed_everything(int(config["seed"]))
    if args.command == "train":
        train(config, args.config, args.output)
        return

    config["agent"]["llm"]["enabled"] = bool(args.qwen_planner)
    agent, dataset = _load_agent(args.checkpoint, config, args.config, args.split)
    if args.qwen_planner and not agent.planner.available:
        raise SystemExit(f"[abort] planner LLM 加载失败,拒绝静默回退到确定性 planner: "
                         f"{getattr(agent.planner, 'load_error', 'unknown error')}")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    indices = [args.index] if args.command == "infer" else list(range(len(dataset)))
    if args.command == "infer-batch" and args.limit > 0:
        indices = indices[:args.limit]

    summaries = []
    for position, index in enumerate(indices, start=1):
        stem = f"{int(dataset.df.iloc[index]['num']):05d}"
        saved = _load_existing_result(output, stem) if args.command == "infer-batch" else None
        if saved is not None:
            evidence, trajectory = saved["evidence"], saved["trajectory"]
            print(f"[{position}/{len(indices)}] {stem} skip (already done)", flush=True)
        else:
            evidence, trajectory = agent.run(dataset, index)
            report = generate_report(evidence, config)["report"]
            patient_id = evidence["patient_id"]
            save_json({"evidence": evidence, "trajectory": trajectory},
                      output / f"{patient_id}_evidence.json")
            (output / f"{patient_id}_report.md").write_text(report, encoding="utf-8")
            print(f"[{position}/{len(indices)}] {stem} done "
                  f"(stop={evidence['stop_reason']}, "
                  f"accepted={len(evidence['accepted_evidence'])})", flush=True)
        summaries.append({"patient_id": evidence["patient_id"], "planner": evidence["planner"],
                          "tool_calls": sum(1 for x in trajectory
                                            if str(x.get("action", {}).get("action", "")).startswith("verify_")),
                          "accepted_evidence": len(evidence["accepted_evidence"]),
                          "stop_reason": evidence["stop_reason"]})
        if args.command == "infer" and saved is None:
            print(report)
    if args.command == "infer-batch":
        save_json({"cases": summaries}, output / "agentic_metrics.json")


if __name__ == "__main__":
    main()
