# REACT-FLD

**Risk-state-driven Evidence ACquisition for fatty liver disease (FLD) risk prediction
from tongue images and physiological indicators.**

This is the official repository of the paper. The code in `reactfld/`, `stage_a.py` and
`evaluate.py` is the exact research codebase used to produce the paper's results,
included unmodified.

## Method overview

REACT-FLD fuses two modality branches — an anatomy-aware tongue-image branch
(5 regions: tip / middle / root / left / right) and a physiological-indicator branch
(8 raw indicators, 3 derived: BMI / WHR / MAP) — and explains every risk decision with
verified evidence:

1. **Stage A – single-task visual pretraining.** A TIPNet encoder (triple-parallel
   ResNet with kernel sizes 3/5/7 + in-block self-attention) is pretrained for FLD
   risk; five seeds form a frozen expert ensemble.
2. **Stage B – anatomy-aware concept learning.** 12 regional + 6 global tongue
   concepts are learned on top of the frozen encoder via region tokens and boxes.
3. **Stage C – evidence-fusion risk training.** A single zero-initialized decision
   weight on the Stage-A logit (identity residual) is trained with concept/region
   evidence concatenated, so evidence can only modulate, never dominate.
4. **Risk-state-driven evidence acquisition agent.** For each case the agent observes
   the risk state (per-modality risks, uncertainty, modality disagreement), plans
   evidence acquisition under a tool budget, and verifies every candidate factor with
   **counterfactual probing** (replace the factor with a healthy reference value,
   re-run the frozen model) plus three reliability filters: contribution size,
   stability across donors, and direction consistency.
5. **Structured report generation.** A deterministic template renders the verified
   evidence into a clinically structured report (in Chinese), covering main tongue
   findings, main physiological abnormalities, model decision factors and a
   comprehensive suggestion.

## Installation

```bash
git clone <this-repo-url> REACT_public
cd REACT_public
pip install -r requirements.txt
```

Python >= 3.10, PyTorch >= 2.0 (CUDA recommended; the demo also runs CPU-only).

## Data

`data/` contains a **10-case public demo subset** (5 FLD-positive, 5 FLD-negative)
sampled from the *training* split of the paper dataset:

```
data/
├── images/                     # tongue photos, 00001.jpg ...
├── masks_body/                 # tongue-body masks (npy)
├── masks_region/               # 5-channel region masks (npy), order: tip/middle/root/right/left
├── Physiological_Indicators.csv  # quesid, Gender(_1/_2), Age, Height, Waist, Hipline,
|                                  # Weight, SBP, DBP, FLD(_0/_1)
├── test_tongue_label.json      # per-case concept ground truth (global + regional)
└── test_data_json.json         # per-case region bounding-box coordinates
```

The **full dataset (5,040 cases) is not redistributed** with this repository for
ethical/privacy reasons; it may be requested from the authors under a data use
agreement.

## Quick start (demo, no pretrained weights needed)

```bash
bash scripts/run_demo.sh
```

This runs the complete pipeline end-to-end on the 10-case subset — Stage A x5 seeds,
Stage B+C training, agentic inference on the test split, evaluation, and prints one
generated structured report. Outputs land in `runs/`:

```
runs/v12_stage_a*/best_visual.pt   # Stage A checkpoints (5 seeds)
runs/v12_ens5_demo/best.pt         # final model + temperature calibrator
runs/v12_ens5_demo/metrics.json    # test metrics of the demo run
runs/demo_inference/00004_evidence.json   # per-case evidence chain + trajectory
runs/demo_inference/00004_report.md       # per-case structured report
```

> **Why `config_demo.yaml`?** The paper split (70/15/15, `config.yaml`) is kept
> verbatim for full-data reproduction. With only 10 cases it mathematically yields an
> empty test subset (per FLD stratum of 5 cases: 4 train / 1 val / 0 test), so the
> demo uses `config_demo.yaml`, which differs **only** in the split ratio (60/20/20 →
> 6/2/2 with both classes in val and test). No source code is modified.

Demo metrics are meaningless (6 training cases); the demo demonstrates that the full
pipeline executes, not the method's performance.

## Reproducing the paper (full dataset)

1. Obtain the full dataset (see *Data*) and point `data.root` in `config.yaml` to it.
2. Train Stage A for seeds 42–46 (protocol: SGD, cosine schedule, 150 epochs):

   ```bash
   for s in 42 43 44 45 46; do
     out=runs/v12_stage_a; [ $s -ne 42 ] && out=runs/v12_stage_a_s$s
     python3 stage_a.py --config config.yaml --seed $s --output $out
   done
   ```

3. Train Stage B + C and run agentic inference over the test split:

   ```bash
   python3 -m reactfld --config config.yaml train --output runs/v12_ens5
   python3 -m reactfld --config config.yaml infer-batch \
     --checkpoint runs/v12_ens5/best.pt --split test --output runs/inference_test
   ```

4. Evaluate:

   ```bash
   python3 evaluate.py --config config.yaml runs/inference_test --threshold 0.5
   # or directly from a checkpoint (with the paper's flip-TTA protocol):
   python3 evaluate.py --config config.yaml --checkpoint runs/v12_ens5/best.pt --tta
   ```

Single-case inference with the LLM planner (optional; requires a local Qwen model and
`transformers`):

```bash
python3 -m reactfld --config config.yaml infer --checkpoint runs/v12_ens5/best.pt \
  --index 0 --qwen-planner --output runs/inference_llm
```

Without `--qwen-planner` the agent uses the deterministic planner, which produced all
paper results.

## Repository structure

```
reactfld/            # core package (models, data, losses, engine, agent, report, CLI)
├── model.py         #   two-branch model: TIPNet encoder + region tokens + fusion
├── concepts.py      #   6 global + 12 regional tongue concepts
├── data.py          #   dataset, seeded patient-level stratified split
├── losses.py        #   risk / concept / evidential losses
├── engine.py        #   staged training, calibration, prediction, metrics
├── agent.py         #   risk-state-driven evidence acquisition agent (+LLM planner)
├── attribution.py   #   counterfactual toolbox (donor matching, healthy references)
├── calibration.py   #   temperature calibration, bootstrap CIs
├── report.py        #   deterministic structured report (Chinese)
└── cli.py           #   train / infer / infer-batch entry points
stage_a.py           # Stage A single-task visual pretraining
evaluate.py          # five-metric evaluation (checkpoint or inference directory)
config.yaml          # paper configuration (verbatim; only data paths made relative)
config_demo.yaml     # demo configuration (only the split ratio differs)
scripts/run_demo.sh  # end-to-end demo on the 10-case subset
tools/               # maintainer tools (demo subset extraction)
data/                # 10-case public demo subset
```

## Citation

If you use this code, please cite:

```bibtex
@article{reactfld2026,
  title   = {REACT-FLD: Risk-state-driven Evidence Acquisition for Fatty Liver Disease
             Risk Prediction from Tongue Images and Physiological Indicators},
  author  = {…},
  journal = {…},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The demo data is
provided for reproducibility research only and may not be redistributed.
