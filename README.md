

<img src="Pictures/title_01.png" alt="title" border="0">

<p float="left"><img src="https://img.shields.io/badge/python-v3.12+-red"> <img src="https://img.shields.io/badge/pytorch-v2.6+-blue">



# Tongue Diagnosis

This repository contains the official implementation of **REACT-FLD**, a novel framework for explainable Fatty Liver Disease (FLD) risk assessment.

<img src="Pictures/gap_01.png" alt="fig2" width="90%" border="0">

**Fig.1**.  Examples illustrating two limitations of existing FLD assessment methods and the motivation of REACT-FLD. (a) Existing multimodal methods typically adopt fixed global fusion strategies, applying shared modality weights across patients despite individual differences, which may lead to over- or under-estimated risk. (b) Existing explanation methods often rely on one-shot attribution or generic explanations without patient-specific counterfactual verification, resulting in unverified evidence and limited clinical trust. (c) REACT-FLD integrates anatomy-aware multimodal risk modeling, patient-specific evidence acquisition and verification, and evidence-grounded explanation to provide reliable FLD risk assessment with verifiable supporting evidence.

## ✨ Overview

<img src="Pictures/framework_01.png" alt="fig2" width="95%" border="0">

**Fig. 2.**  Overall framework of REACT-FLD. Stage I: Anatomy-aware multimodal risk modeling jointly encodes tongue images and physiological variables for FLD risk prediction, while region-wise cross-attention and anatomical modeling learn global and regional tongue concepts. Stage II: Counterfactual evidence reasoning constructs candidate factors from tongue concepts, anatomical regions, physiological variables, and cross-modal interactions, and verifies them against matched non-FLD references according to predictive contribution, stability, and directional consistency. Stage III: Evidence-grounded report generation organizes the predicted risk and validated patient-specific evidence into structured, verified, and interpretable clinical reports.

## 📊 Experiments

<div align="center">
<img src="Pictures/hyperparam_threshold_01.png" alt="fig3" width="95%" border="0">
</div>


**Fig. 3.**   Hyperparameter analysis of the counterfactual evidence acceptance thresholds in REACT-FLD. (a) Contribution threshold $\tau_{\Delta}$; (b) stability threshold $\tau_{S}$; (c) directional-consistency threshold $\tau_{D}$.

<div align="center">
<img src="Pictures/efficiency_analysis_final_01.png" alt="fig4" width="85%" border="0">
</div>


**Fig.4.**   Efficiency analysis of REACT-FLD. (a) Training time of the Stage-I risk prediction model compared with representative visual and tongue-specific baselines. (b) Module-wise inference latency under deterministic and LLM-based planning. Both settings show comparable costs for risk prediction, concept inference, and counterfactual verification, while the LLM-based planner introduces substantially higher planning overhead.



## 🏆 Main Results

REACT-FLD consistently outperforms state-of-the-art LLMs, visual models, and specialized baselines across all evaluation metrics, demonstrating superior risk assessment capability and evidence quality.

### Table 1. Sensitivity comparison at clinically relevant fixed specificity levels

| Model                | Sens@Spec=0.80 $\uparrow$ | Sens@Spec=0.90 $\uparrow$ | Sens@Spec=0.95 $\uparrow$ |
| :------------------- | :-----------------------: | :-----------------------: | :-----------------------: |
| DeepSeek-V4.1-Flash  |      0.6517 ± 0.0471      |      0.4757 ± 0.0672      |      0.2434 ± 0.0594      |
| GLM-5.3-Flash        |      0.6667 ± 0.0570      |      0.4682 ± 0.0484      |      0.3558 ± 0.0590      |
| Qwen3.5              |      0.6704 ± 0.0420      |      0.4007 ± 0.0580      |      0.2921 ± 0.0918      |
| Doubao-Seed-2.0      |      0.6217 ± 0.0345      |      0.3858 ± 0.0618      |      0.3221 ± 0.0544      |
| Kimi-K2.6            |      0.6030 ± 0.0502      |      0.5243 ± 0.0601      |      0.2846 ± 0.0622      |
| *---*                |           *---*           |           *---*           |           *---*           |
| ResNet50             |      0.6105 ± 0.0351      |      0.4794 ± 0.0325      |      0.4082 ± 0.0359      |
| ConvNeXt-Tiny        |      0.6742 ± 0.0404      |      0.5468 ± 0.0402      |      0.4382 ± 0.0397      |
| ViT-B                |      0.5880 ± 0.0353      |      0.4682 ± 0.0382      |      0.3858 ± 0.0407      |
| TIPNet               |      0.7154 ± 0.0333      |      0.5243 ± 0.0342      |      0.4569 ± 0.0409      |
| TIPNet_Indicators    |  <u>0.8464 ± 0.0246</u>   |  <u>0.7228 ± 0.0385</u>   |  <u>0.6142 ± 0.0432</u>   |
| **REACT-FLD (Ours)** |    **0.8689 ± 0.0267**    |    **0.7528 ± 0.0387**    |    **0.6479 ± 0.0449**    |

### Table 2. Comparison of counterfactual evidence quality with baselines

| Method               | Evidence F1 $\uparrow$ |  Coverage $\uparrow$   | CF Stability $\uparrow$ |
| :------------------- | :--------------------: | :--------------------: | :---------------------: |
| DeepSeek-V4.1-Flash  |    0.1194 ± 0.0056     |    0.0878 ± 0.0043     |     0.3758 ± 0.0211     |
| GLM-5.3-Flash        | <u>0.1436 ± 0.0075</u> | <u>0.1084 ± 0.0059</u> |     0.4387 ± 0.0222     |
| Qwen3.5              |    0.1085 ± 0.0056     |    0.0774 ± 0.0041     |     0.3829 ± 0.0211     |
| Doubao-Seed-2.0      |    0.1146 ± 0.0071     |    0.0774 ± 0.0050     | <u>0.5169 ± 0.0211</u>  |
| Kimi-K2.6            |    0.0856 ± 0.0049     |    0.0600 ± 0.0035     |     0.4035 ± 0.0205     |
| *---*                |         *---*          |         *---*          |          *---*          |
| **REACT-FLD (Ours)** |  **0.3791 ± 0.0087**   |  **0.3235 ± 0.0083**   |   **0.8311 ± 0.0150**   |

---

## 💡 Ablation Study

We conduct ablation studies to validate the contribution of each core component in the counterfactual evidence reasoning pipeline.

### Table 3. Ablation of counterfactual evidence reasoning

| Method                    | Evidence  | Stability $\uparrow$ | Direction Consistency $\uparrow$ |
| :------------------------ | :-------: | :------------------: | :------------------------------: |
| w/o CF Verification       |   6.000   |          --          |                --                |
| w/o Reliability Filtering |   3.239   |        0.4639        |              0.9540              |
| **REACT-FLD (Full)**      | **0.426** |      **0.7503**      |            **0.9876**            |

*(Note: Lower "Evidence" count with higher Stability and Consistency indicates more precise and reliable evidence acquisition.)*

---

## 🔍 Qualitative Analysis: Case Study

The following cases demonstrate how REACT-FLD acquires and verifies patient-specific counterfactual evidence. 

- <span style="color: #d9534f; font-weight: bold;">Red text</span> indicates factors that were **Rejected** (failed to meet stability or directional consistency thresholds).
- <span style="color: #5cb85c; font-weight: bold;">Green text</span> indicates factors that were successfully **Validated** as reliable supporting evidence.

### Table 4. Representative cases of patient-specific counterfactual evidence verification

*Middle, Right, and Root denote the corresponding anatomical tongue regions.*

|  No.  | Risk & Candidates                                            | Counterfactual Verification                                  |
| :---: | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **1** | **Risk:** 96.5%<br>**Candidates:** Middle, Right, Root       | **Middle:** Δp=0.218, S=0.67, D=1.00<br>**Right:** Δp=0.088, S=0.78, D=1.00<br><span style="color: #d9534f;">Root rejected: weak and unstable effect.</span><br><span style="color: #5cb85c; font-weight: bold;">Validated: Middle, Right</span> |
| **2** | **Risk:** 75.8%<br>**Candidates:** 5 concepts+Root           | <span style="color: #d9534f;">Concept candidates: Δp ≈ 0.</span><br>**Root:** Δp=0.097, S=0.84, D=1.00<br><span style="color: #5cb85c; font-weight: bold;">Validated: Root</span> |
| **3** | **Risk:** 93.2%<br>**Candidates:** Root, Middle, Tip, Right, Left, Coating color | **Middle:** Δp=0.177, S=0.62, D=1.00<br><span style="color: #d9534f;">Root/coating color: negligible effect.</span><br><span style="color: #d9534f;">Tip/Right/Left: insufficient stability.</span><br><span style="color: #5cb85c; font-weight: bold;">Validated: Middle</span> |

### 📝 Case Study: Generated Explainable Clinical Report

Beyond numerical metrics, REACT-FLD generates structured, evidence-grounded clinical reports. Below is a representative output for a high-risk patient, demonstrating how counterfactual reasoning is translated into interpretable clinical insights.

> **🏥 FLD Risk Assessment Report (Patient #05544)**
>
> **📊 Predicted Risk:** 🔴 **89.3% (High Risk)**
>
> **👅 Key Tongue Manifestations:**
>
> - **Body Shape:** Thin/Small (Confidence: 99%)
> - **Tip Shape:** Protruding/High (Confidence: 100%)
> - **Fissure:** Present (Confidence: 62%)
> - **Coating (Middle/Root):** Scanty/Absent (Confidence: 39% / 50%)
> - **Right Side Shape:** Depressed (Confidence: 76%)
>
> **🩺 Physiological Abnormalities:**
>
> - **BMI:** 28.50 (Obesity)
> - **Blood Pressure:** 136/98 mmHg (Elevated)
>
> **🔍 Key Counterfactual Evidence (Model Attribution):**
>
> - **Tip Region:** Replacing the 'tip' features with a normal reference decreases the predicted risk by **~8%** *(Stability: 0.75, Directional Consistency: 100%)*.
>
> **💡 Comprehensive Insight:** 
> The model integrates tongue and physiological data to indicate high FLD risk, identifying the 'tip' region as a significant, verified contributing factor to this assessment.
>
> ---
>
> *⚠️ Disclaimer: The risk variations reflect the model's attribution and do not imply clinical causality. This report is for auxiliary risk assessment only and does not replace professional clinical diagnosis.*

## 💾 Dataset: Enhanced MFF-TDF for REACT-FLD

We construct an enhanced, richly annotated version of the publicly available **MFF-TDF** dataset to support anatomy-aware representation learning and explainable Fatty Liver Disease (FLD) risk assessment.

### 📊 Dataset Statistics & Modalities

- **Total Samples**: 5,040 high-quality cases (filtered from an initial 5,717 via strict quality control, removing severe blur, abnormal exposure, occlusion, or incomplete tongue capture).
- **Visual Modality**: High-resolution tongue images.
- **Physiological Modality**: 11 variables (8 original clinical variables + 3 derived: BMI, Waist-to-Hip Ratio (WHR), Mean Arterial Pressure (MAP)).
- **Target**: Binary FLD label.

### ✨ Key Annotation Features

To enable fine-grained, clinically interpretable modeling, each sample is meticulously annotated with:

1. **Anatomical Priors**: 
   - A whole-tongue segmentation mask.
   - 5 region-specific masks and normalized bounding boxes (`tip`, `middle`, `root`, `left`, `right`), localized via a YOLOv5-based detector.
2. **18 Clinically Interpretable Tongue Concepts**:
   - **6 Global Concepts**: Fissure, tooth mark, body shape, coating thickness, coating color, and body color.
   - **12 Regional Concepts**: Characterizing localized variations in color, coating, and morphology across the 5 anatomical regions.


## 🌟 Contributions and suggestions are welcome!
