"""Pipeline complet : données -> variables -> modèles -> artefacts pour le dashboard.

Usage : ``python -m flc.pipeline [--n-iter 20] [--n-boot 500]``
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .data import coverage_table, load_raw, modelling_frame
from .evaluation import bootstrap_ci, metrics, reliability_table
from .features import build_feature_sets, feature_group
from .modeling import make_dummy, make_lgbm, make_logreg, tune_lgbm, tune_logreg

PROFILE_STATS = {
    "Tirs": "total_shots",
    "Tirs cadrés": "total_shots_on_target",
    "Fautes": "total_fouls",
    "Corners": "total_corners",
    "Cartons jaunes": "total_yellow",
    "Cartons rouges": "total_red",
    "Buts": "total_goals",
}


def grouped_permutation_importance(model, X: pd.DataFrame, y: np.ndarray, n_repeats=5,
                                   seed=C.RANDOM_STATE) -> pd.DataFrame:
    """Hausse de log-loss quand on permute (ensemble) les colonnes d'un même groupe ou une seule colonne."""
    rng = np.random.default_rng(seed)
    labels = range(len(C.CLASSES))
    base = log_loss(y, model.predict_proba(X), labels=labels)
    rows = []
    units = [("group", g, [c for c in X.columns if feature_group(c) == g])
             for g in dict.fromkeys(feature_group(c) for c in X.columns)]
    units += [("feature", c, [c]) for c in X.columns]
    for kind, name, cols in units:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            deltas.append(log_loss(y, model.predict_proba(Xp), labels=labels) - base)
        rows.append({"kind": kind, "name": name,
                     "group": name if kind == "group" else feature_group(name),
                     "importance": float(np.mean(deltas)), "std": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values(["kind", "importance"], ascending=[True, False])


def export_simulator_presets(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque ligue, un match réel de la dernière saison, le plus proche du profil médian de la ligue.

    (Des médianes colonne par colonne donneraient un match artificiel et incohérent.)
    """
    raw_cols = C.STAT_COLUMNS + ["FTHG", "FTAG", "HTHG", "HTAG"]
    last = df["season"] == df["season"].max()
    d = df.loc[last].copy()
    z = (d[raw_cols] - d[raw_cols].mean()) / d[raw_cols].std()
    rows = []
    for lg in C.CLASSES:
        m = d["league"] == lg
        dist = ((z[m] - z[m].median()) ** 2).sum(axis=1)
        i = dist.idxmin()
        rows.append({"league": lg, **d.loc[i, raw_cols].to_dict(),
                     "kickoff_hour": X.loc[i, "kickoff_hour"], "weekday": X.loc[i, "weekday"],
                     "month": X.loc[i, "month"], "home": d.loc[i, "HomeTeam"], "away": d.loc[i, "AwayTeam"],
                     "date": d.loc[i, "Date"].date().isoformat()})
    out = pd.DataFrame(rows).set_index("league")
    out.to_csv(C.ARTIFACTS / "simulator_presets.csv")
    return out


def main(n_iter: int = 20, n_boot: int = 500) -> None:
    t0 = time.time()
    out = C.ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    coverage_table(raw).to_csv(out / "coverage.csv", index=False)
    df = modelling_frame(raw)
    X, sets = build_feature_sets(df)
    y = df["league"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()

    split = np.select([df["season"].isin(C.TRAIN_SEASONS), df["season"].isin(C.VALID_SEASONS),
                       df["season"].isin(C.TEST_SEASONS)], ["train", "valid", "test"], "unused")
    tr, va, te = (split == s for s in ("train", "valid", "test"))
    trva = tr | va
    print(f"Matchs exploitables : {len(df)}  (train {tr.sum()}, valid {va.sum()}, test {te.sum()})")

    def fit_predict(factory, params, n_ref, fit_mask, pred_mask, cols):
        """Entraîne sur `fit_mask` (itérations LightGBM mises à l'échelle) et prédit `pred_mask`."""
        params = dict(params)
        if "n_estimators" in params:
            params["n_estimators"] = int(round(params["n_estimators"] * fit_mask.sum() / n_ref))
        m = factory(**params).fit(X.loc[fit_mask, cols], y[fit_mask])
        return m, m.predict_proba(X.loc[pred_mask, cols]), params

    results, tuning, test_proba, last_models = [], {}, {}, {}
    seasons = df["season"].to_numpy()
    for set_name, cols in sets.items():
        Xs = X[cols]
        t1 = time.time()
        candidates = {
            "dummy": (None, make_dummy),
            "logreg": (tune_logreg(Xs[tr], y[tr], Xs[va], y[va]), partial(make_logreg, columns=cols)),
            "lightgbm": (tune_lgbm(Xs[tr], y[tr], Xs[va], y[va], n_iter=n_iter), make_lgbm),
        }
        print(f"[{set_name}] réglage : {time.time() - t1:.0f}s")
        for model_name, (tuned, factory) in candidates.items():
            if model_name == "dummy" and set_name != "match":
                continue  # la baseline ne dépend pas des variables
            params = tuned.params if tuned else {}
            # 1) validation : entraîné sur les saisons d'entraînement seulement
            _, p_va, _ = fit_predict(factory, params, tr.sum(), tr, va, cols)
            valid = metrics(y[va], p_va)
            # 2) test "walk-forward" : chaque saison de test est prédite par un modèle
            #    ré-entraîné sur toutes les saisons antérieures (comme en production).
            p = np.zeros((te.sum(), len(C.CLASSES)))
            per_season = {}
            for s_test in C.TEST_SEASONS:
                fit_mask = (seasons < s_test) & (split != "unused")
                pred_mask = seasons == s_test
                m, p_s, used = fit_predict(factory, params, tr.sum(), fit_mask, pred_mask, cols)
                p[pred_mask[te]] = p_s
                per_season[f"{s_test}-{str(s_test + 1)[-2:]}"] = metrics(y[pred_mask], p_s)
            key = "baseline" if model_name == "dummy" else f"{model_name}__{set_name}"
            test_proba[key] = p
            last_models[key] = (m, cols)
            test = metrics(y[te], p)
            ci = bootstrap_ci(y[te], p, n_boot=n_boot)
            results.append({
                "key": key, "model": model_name,
                "feature_set": "—" if model_name == "dummy" else set_name,
                "n_features": 0 if model_name == "dummy" else len(cols),
                "params": params,
                "valid": valid, "test": test, "test_by_season": per_season,
                "test_ci": {k: list(v) for k, v in ci.items()},
            })
            if tuned:
                tuning[key] = tuned.history
            print(f"  {key:28s} valid acc {valid['accuracy']:.3f} | test acc {test['accuracy']:.3f}"
                  f" macroF1 {test['macro_f1']:.3f} logloss {test['log_loss']:.3f}")

    best_key = min((r for r in results if r["model"] != "dummy"),  # choix sur la validation
                   key=lambda r: r["valid"]["log_loss"])["key"]

    # --- Prédictions du test (une ligne par match et par modèle) ---
    meta = df.loc[te, ["match_id", "Date", "season_label", "league", "country", "tier",
                       "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].reset_index(drop=True)
    frames = []
    for key, p in test_proba.items():
        f = meta.copy()
        f["model_key"] = key
        f["pred"] = np.array(C.CLASSES)[p.argmax(axis=1)]
        f["confidence"] = p.max(axis=1)
        for i, c in enumerate(C.CLASSES):
            f[f"p_{c}"] = p[:, i].astype("float32")
        frames.append(f)
    pd.concat(frames, ignore_index=True).to_parquet(out / "test_predictions.parquet", index=False)

    # --- Calibration et importance des variables ---
    rel = []
    for key, p in test_proba.items():
        r = reliability_table(y[te], p)
        r["model_key"] = key
        rel.append(r)
    pd.concat(rel).to_csv(out / "calibration.csv", index=False)

    # Importance : modèle de la dernière saison de test, évalué sur cette saison.
    imps = []
    last = seasons == C.TEST_SEASONS[-1]
    for key in [k for k in last_models if k.startswith("lightgbm")]:
        model, cols = last_models[key]
        t1 = time.time()
        imp = grouped_permutation_importance(model, X.loc[last, cols], y[last])
        imp["model_key"] = key
        imps.append(imp)
        print(f"importance {key}: {time.time() - t1:.0f}s")
    pd.concat(imps).to_csv(out / "permutation_importance.csv", index=False)

    # --- Profils de jeu par ligue et par saison (descriptif) ---
    stats = X.loc[:, list(PROFILE_STATS.values())].rename(columns={v: k for k, v in PROFILE_STATS.items()})
    stats[["league", "season", "season_label", "kickoff_hour", "split"]] = (
        df[["league", "season", "season_label", "kickoff_hour"]].assign(split=split))
    stats.groupby(["league", "season", "season_label"]).mean(numeric_only=True).reset_index() \
        .to_csv(out / "league_season_profiles.csv", index=False)
    # échantillon de matchs (hors test) pour les distributions du dashboard
    stats.loc[trva].groupby("league", group_keys=False).sample(n=600, random_state=C.RANDOM_STATE) \
        .to_parquet(out / "match_sample.parquet", index=False)

    # --- Modèle du simulateur : LightGBM "match + contexte", ré-entraîné sur toutes les saisons ---
    sim_key = "lightgbm__match_context"
    sim_cols = sets["match_context"]
    sim_params = next(r["params"] for r in results if r["key"] == sim_key)
    everything = split != "unused"
    sim_params = {**sim_params,
                  "n_estimators": int(round(sim_params["n_estimators"] * everything.sum() / tr.sum()))}
    sim_model = make_lgbm(**sim_params).fit(X.loc[everything, sim_cols], y[everything])
    joblib.dump({"model": sim_model, "features": sim_cols, "classes": C.CLASSES},
                out / "simulator_model.joblib", compress=3)
    ref = X.loc[everything, sim_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    ref.to_csv(out / "simulator_feature_ranges.csv")
    export_simulator_presets(df, X)

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "classes": C.CLASSES,
        "leagues": {k: {"country": v[0], "tier": v[1]} for k, v in C.LEAGUES.items()},
        "splits": {
            "train": {"seasons": [f"{s}-{str(s + 1)[-2:]}" for s in C.TRAIN_SEASONS], "n": int(tr.sum())},
            "valid": {"seasons": [f"{s}-{str(s + 1)[-2:]}" for s in C.VALID_SEASONS], "n": int(va.sum())},
            "test": {"seasons": [f"{s}-{str(s + 1)[-2:]}" for s in C.TEST_SEASONS], "n": int(te.sum())},
        },
        "n_raw_matches": int(len(raw)),
        "n_raw_with_stats": int(raw["has_stats"].sum()),
        "n_modelling_matches": int(len(df)),
        "feature_sets": sets,
        "best_model": best_key,
        "simulator_model": sim_key,
        "results": results,
        "tuning": tuning,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"Meilleur modèle (log-loss de validation) : {best_key} — {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    main(a.n_iter, a.n_boot)
