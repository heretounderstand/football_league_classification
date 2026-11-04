"""Modèles et réglage des hyperparamètres sur la saison de validation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLASSES, RANDOM_STATE

N_CLASSES = len(CLASSES)


@dataclass
class Tuned:
    name: str
    params: dict
    valid_log_loss: float
    history: list[dict] = field(default_factory=list)


def make_dummy():
    return DummyClassifier(strategy="prior")


CATEGORICAL = ("kickoff_hour", "weekday", "month")


def make_logreg(C: float = 1.0, columns: list[str] | None = None):
    """Régression logistique multinomiale ; heure/jour/mois en one-hot (effets non linéaires)."""
    columns = list(columns or [])
    cat = [c for c in CATEGORICAL if c in columns]
    num = [c for c in columns if c not in cat]
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre = ColumnTransformer([
        ("num", numeric, num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=3000))


def make_lgbm(**params):
    base = dict(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )
    base.update(params)
    return lgb.LGBMClassifier(**base)


def tune_logreg(X_tr, y_tr, X_va, y_va, grid=(0.01, 0.1, 1.0, 10.0)) -> Tuned:
    history = []
    for C in grid:
        m = make_logreg(C, list(X_tr.columns)).fit(X_tr, y_tr)
        history.append({"C": C, "valid_log_loss": log_loss(y_va, m.predict_proba(X_va),
                                                           labels=range(N_CLASSES))})
    best = min(history, key=lambda h: h["valid_log_loss"])
    return Tuned("logreg", {"C": best["C"]}, best["valid_log_loss"], history)


def _sample_lgbm_params(rng: np.random.Generator) -> dict:
    return {
        "num_leaves": int(rng.choice([7, 15, 31, 63])),
        "min_child_samples": int(rng.choice([20, 50, 100, 200])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.4, 1.0)),
        "reg_lambda": float(10 ** rng.uniform(-2, 1.5)),
    }


def tune_lgbm(X_tr, y_tr, X_va, y_va, n_iter: int = 20, seed: int = RANDOM_STATE) -> Tuned:
    """Recherche aléatoire, arrêt précoce sur la log-loss de la saison de validation."""
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(n_iter):
        params = _sample_lgbm_params(rng)
        m = make_lgbm(n_estimators=3000, **params)
        with warnings.catch_warnings():  # 'eval_set' déprécié dans LightGBM >= 4.6, gardé pour compatibilité
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        params["n_estimators"] = int(m.best_iteration_)
        history.append({**params, "valid_log_loss": float(m.best_score_["valid_0"]["multi_logloss"])})
    best = min(history, key=lambda h: h["valid_log_loss"])
    params = {k: v for k, v in best.items() if k != "valid_log_loss"}
    return Tuned("lightgbm", params, best["valid_log_loss"], history)
