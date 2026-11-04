"""Audit de l'ancienne méthodologie (dossier legacy/) : d'où venait le 96 % d'accuracy ?

On reproduit le protocole d'origine à partir des données brutes, puis on retire
une à une les sources de fuite :

A. protocole d'origine : stats manquantes remplacées par 0, "force" des équipes
   calculée sur tout le jeu de données, découpage aléatoire 70/30 ;
B. A sans la force des équipes ;
C. B en ne gardant que les matchs qui ont réellement des statistiques ;
D. C avec un découpage temporel (test = 2021-22 et 2022-23).

Usage : ``python scripts/audit_legacy.py``  ->  artifacts/legacy_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flc.config import ARTIFACTS, CLASSES, RANDOM_STATE, STAT_PAIRS, TEST_SEASONS  # noqa: E402
from flc.data import load_raw  # noqa: E402
from flc.modeling import make_lgbm  # noqa: E402


def legacy_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    stat_cols = [c for p in STAT_PAIRS.values() for c in p]
    df[stat_cols] = df[stat_cols].fillna(0)  # comme rename_feature.py
    f = pd.DataFrame(index=df.index)
    for name, (h, a) in STAT_PAIRS.items():
        f[f"home_{name}"], f[f"away_{name}"] = df[h], df[a]
        f[f"total_{name}"] = df[h] + df[a]
    f["home_dominance_yellow"] = df["HY"] / f["total_yellow"].replace(0, 1)
    f["weekday"] = df["Date"].dt.weekday
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)

    # "Force" des équipes : points par match sur TOUT le jeu (passé, futur et test compris)
    pts_home = df["FTR"].map({"H": 3, "D": 1, "A": 0})
    pts_away = df["FTR"].map({"A": 3, "D": 1, "H": 0})
    long = pd.concat([
        pd.DataFrame({"team": df["HomeTeam"], "pts": pts_home}),
        pd.DataFrame({"team": df["AwayTeam"], "pts": pts_away}),
    ])
    strength = long.groupby("team")["pts"].mean()
    f["home_team_strength"] = df["HomeTeam"].map(strength)
    f["away_team_strength"] = df["AwayTeam"].map(strength)
    return f


def score(X, y, train_mask, test_mask) -> dict:
    m = make_lgbm(n_estimators=300, num_leaves=63).fit(X[train_mask], y[train_mask])
    pred = m.predict(X[test_mask])
    return {"accuracy": round(float(accuracy_score(y[test_mask], pred)), 4),
            "macro_f1": round(float(f1_score(y[test_mask], pred, average="macro")), 4),
            "n_test": int(test_mask.sum())}


def main() -> None:
    raw = load_raw()
    X = legacy_features(raw)
    y = raw["league"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    no_strength = [c for c in X.columns if "strength" not in c]

    idx = np.arange(len(raw))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    rand_tr = np.isin(idx, tr_idx)
    rand_te = ~rand_tr
    has = raw["has_stats"].to_numpy()
    temporal_te = raw["season"].isin(TEST_SEASONS).to_numpy()

    # Ce que révèle la seule absence de statistiques : un modèle qui ne voit que (saison, stats manquantes ?)
    leak_X = pd.DataFrame({"season": raw["season"], "missing_stats": (~has).astype(int)})

    rows = [
        ("A", "Protocole d'origine (stats manquantes = 0, force des équipes, split aléatoire)",
         score(X, y, rand_tr, rand_te)),
        ("A0", "Seulement la force des équipes (aucune statistique de match)",
         score(X[["home_team_strength", "away_team_strength"]], y, rand_tr, rand_te)),
        ("A1", "Seulement la saison et l'indicateur « stats manquantes »",
         score(leak_X, y, rand_tr, rand_te)),
        ("B", "A sans la force des équipes", score(X[no_strength], y, rand_tr, rand_te)),
        ("C", "B, uniquement les matchs avec de vraies statistiques",
         score(X[no_strength], y, rand_tr & has, rand_te & has)),
        ("D", "C avec découpage temporel (test 2021-23)",
         score(X[no_strength], y, has & ~temporal_te, has & temporal_te)),
    ]
    out = [{"id": i, "scenario": s, **m} for i, s, m in rows]
    for r in out:
        print(f"{r['id']:>3}  acc {r['accuracy']:.3f}  macroF1 {r['macro_f1']:.3f}  {r['scenario']}")
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "legacy_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
