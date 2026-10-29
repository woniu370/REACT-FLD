"""风险校准与置信区间（文档 §9.5）。

Temperature Scaling 与 Isotonic Regression 二选一（验证集 NLL 更低者胜）；
指标 95% 置信区间用 bootstrap；个体风险区间由证据不确定性近似。
"""
from __future__ import annotations

import numpy as np


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, probabilities, labels):
        from scipy.optimize import minimize_scalar
        logits = _logit(probabilities)
        y = np.asarray(labels)

        def nll(t):
            q = 1 / (1 + np.exp(-logits / t))
            return -np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9))

        self.temperature = float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)
        return self

    def transform(self, probabilities):
        return 1 / (1 + np.exp(-_logit(probabilities) / self.temperature))

    def nll(self, probabilities, labels):
        q = self.transform(probabilities)
        y = np.asarray(labels)
        return float(-np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9)))

    def state_dict(self):
        return {"kind": "temperature", "temperature": self.temperature}


class IsotonicCalibrator:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)

    def fit(self, probabilities, labels):
        self.iso.fit(np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=float))
        return self

    def transform(self, probabilities):
        return np.asarray(self.iso.predict(np.asarray(probabilities, dtype=float)), dtype=float)

    def nll(self, probabilities, labels):
        q = np.clip(self.transform(probabilities), 1e-6, 1 - 1e-6)
        y = np.asarray(labels)
        return float(-np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9)))

    def state_dict(self):
        return {"kind": "isotonic", "x": self.iso.X_thresholds_.tolist(), "y": self.iso.y_thresholds_.tolist()}


def fit_best_calibrator(probabilities, labels, mode: str = "auto"):
    """§9.5：比较 Temperature 与 Isotonic，返回验证集 NLL 更低者。
    v12:mode="temperature" 时只拟合温度缩放(严格单调,AUROC 排序无损)。"""
    temp = TemperatureScaler().fit(probabilities, labels)
    if mode == "temperature":
        return temp
    iso = IsotonicCalibrator().fit(probabilities, labels)
    return min([temp, iso], key=lambda c: c.nll(probabilities, labels))


def expected_calibration_error(y, p, bins=10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def bootstrap_metric_ci(y, p, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """指标（如 AUROC）的 bootstrap 95% 置信区间（§18.2）。"""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        sample_y, sample_p = y[idx], p[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        stats.append(metric_fn(sample_y, sample_p))
    stats = np.asarray(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(np.mean(stats)), "ci_low": float(lo), "ci_high": float(hi)}


def risk_interval(p: float, uncertainty: float) -> tuple[float, float]:
    """个体风险区间近似：p ± 1.96·u/2（u 为证据不确定性，∈[0,1]）。"""
    delta = 1.96 * min(uncertainty, 1.0) / 2.0
    return max(0.0, p - delta), min(1.0, p + delta)
