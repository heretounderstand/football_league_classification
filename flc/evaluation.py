"""Métriques d'évaluation (multi-classes, hiérarchiques et probabilistes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)

from .config import CLASSES, LEAGUES, RANDOM_STATE

COUNTRY = np.array([LEAGUES[c][0] for c in CLASSES])
TIER = np.array([LEAGUES[c][1] for c in CLASSES])


def _labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in "iu":
        return y
    index = {c: i for i, c in enumerate(CLASSES)}
    return np.array([index[v] for v in y])


def metrics(y_true, proba: np.ndarray) -> dict[str, float]:
    """`y_true` : libellés ou indices de classes ; `proba` : (n, 10) dans l'ordre de CLASSES."""
    y = _labels(y_true)
    pred = proba.argmax(axis=1)
    k = np.arange(len(CLASSES))
    # Pays / niveau déduits en sommant les probabilités des ligues concernées.
    countries = list(dict.fromkeys(COUNTRY))
    p_country = np.stack([proba[:, COUNTRY == c].sum(axis=1) for c in countries], axis=1)
    p_tier = np.stack([proba[:, TIER == t].sum(axis=1) for t in (1, 2)], axis=1)
    y_country = np.array([countries.index(c) for c in COUNTRY[y]])
    y_tier = TIER[y] - 1
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "top2_accuracy": top_k_accuracy_score(y, proba, k=2, labels=k),
        "log_loss": log_loss(y, np.clip(proba, 1e-12, 1), labels=k),
        "country_accuracy": accuracy_score(y_country, p_country.argmax(axis=1)),
        "tier_accuracy": accuracy_score(y_tier, p_tier.argmax(axis=1)),
    }


def bootstrap_ci(y_true, proba, n_boot: int = 500, alpha: float = 0.05,
                 seed: int = RANDOM_STATE) -> dict[str, tuple[float, float]]:
    """Intervalles de confiance par bootstrap (rééchantillonnage des matchs du test)."""
    y = _labels(y_true)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(metrics(y[idx], proba[idx]))
    d = pd.DataFrame(draws)
    return {c: (d[c].quantile(alpha / 2), d[c].quantile(1 - alpha / 2)) for c in d.columns}


def reliability_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Calibration de la classe prédite : confiance moyenne vs taux de bonnes réponses."""
    y = _labels(y_true)
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    t = pd.DataFrame({"bin": b, "conf": conf, "correct": correct})
    out = t.groupby("bin").agg(confidence=("conf", "mean"), accuracy=("correct", "mean"),
                               count=("conf", "size")).reset_index()
    out["bin_low"] = bins[out["bin"]]
    out["bin_high"] = bins[out["bin"] + 1]
    return out
