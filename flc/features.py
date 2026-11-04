"""Construction des variables explicatives.

Trois familles, empilées en trois jeux de variables comparés dans l'étude :

- ``match``   : uniquement ce qui se passe sur le terrain pendant le match ;
- ``context`` : contexte calendaire (heure du coup d'envoi, jour, mois) ;
- ``form``    : profil statistique des matchs *précédents* des deux équipes
                (moyennes glissantes, décalées d'un match : aucune information
                du match courant ni du futur).

Aucune variable n'utilise le nom des équipes, l'arbitre ou la cible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FORM_WINDOW, STAT_PAIRS


def _safe_ratio(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def match_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for name, (h, a) in STAT_PAIRS.items():
        f[f"home_{name}"] = df[h]
        f[f"away_{name}"] = df[a]
        f[f"total_{name}"] = df[h] + df[a]
        f[f"diff_{name}"] = df[h] - df[a]
    f["home_goals"] = df["FTHG"]
    f["away_goals"] = df["FTAG"]
    f["total_goals"] = df["FTHG"] + df["FTAG"]
    f["diff_goals"] = df["FTHG"] - df["FTAG"]
    f["ht_total_goals"] = df["HTHG"] + df["HTAG"]
    f["second_half_goals"] = f["total_goals"] - f["ht_total_goals"]
    f["home_shot_accuracy"] = _safe_ratio(df["HST"], df["HS"])
    f["away_shot_accuracy"] = _safe_ratio(df["AST"], df["AS"])
    f["conversion_rate"] = _safe_ratio(f["total_goals"], f["total_shots_on_target"])
    f["fouls_per_card"] = f["total_fouls"] / (f["total_yellow"] + f["total_red"] + 1)
    f["corners_per_shot"] = _safe_ratio(f["total_corners"], f["total_shots"])
    return f


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["kickoff_hour"] = df["kickoff_hour"]  # manquant avant 2019-20, pour toutes les ligues
    f["weekday"] = df["Date"].dt.weekday
    f["month"] = df["Date"].dt.month
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    return f


FORM_STATS = ["shots", "shots_on_target", "fouls", "corners", "yellow", "red", "goals"]


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (match, équipe) avec les totaux du match et les valeurs de l'équipe."""
    totals = {s: df[h] + df[a] for s, (h, a) in STAT_PAIRS.items()}
    totals["goals"] = df["FTHG"] + df["FTAG"]
    parts = []
    for side, team_col, own in (("home", "HomeTeam", "H"), ("away", "AwayTeam", "A")):
        p = pd.DataFrame({
            "match_id": df["match_id"].values,
            "Date": df["Date"].values,
            "team": df[team_col].values,
            "side": side,
        })
        for s in FORM_STATS:
            p[f"tot_{s}"] = totals[s].values
        p["own_fouls"] = df[f"{own}F"].values
        p["own_yellow"] = df[f"{own}Y"].values
        parts.append(p)
    return pd.concat(parts, ignore_index=True).sort_values(["team", "Date", "match_id"])


def form_features(df: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Moyennes des `window` matchs précédents de chaque équipe, moyennées entre les deux équipes.

    Le décalage ``shift(1)`` garantit que le match courant n'entre jamais dans sa propre forme.
    """
    long = _team_long(df)
    value_cols = [c for c in long.columns if c.startswith(("tot_", "own_"))]
    rolled = (
        long.groupby("team", sort=False)[value_cols]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled["n_prev"] = long.groupby("team", sort=False).cumcount().clip(upper=window)
    rolled[["match_id", "side"]] = long[["match_id", "side"]]
    wide = rolled.pivot(index="match_id", columns="side")
    wide.columns = [f"{side}_{col}" for col, side in wide.columns]

    out = pd.DataFrame(index=wide.index)
    for c in value_cols:
        out[f"form_{c}"] = wide[[f"home_{c}", f"away_{c}"]].mean(axis=1)
    out["form_n_prev"] = wide[["home_n_prev", "away_n_prev"]].min(axis=1)
    out = out.reindex(df["match_id"].values)
    out.index = df.index
    return out


def build_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    m, c, fo = match_features(df), context_features(df), form_features(df)
    X = pd.concat([m, c, fo], axis=1).astype(float)
    sets = {
        "match": list(m.columns),
        "match_context": list(m.columns) + list(c.columns),
        "full": list(m.columns) + list(c.columns) + list(fo.columns),
    }
    return X, sets


FEATURE_GROUPS = {
    "Tirs": ("shots", "shot_accuracy", "corners_per_shot"),
    "Fautes & cartons": ("fouls", "yellow", "red", "per_card"),
    "Corners": ("corners",),
    "Buts": ("goals", "conversion"),
    "Contexte calendaire": ("kickoff_hour", "weekday", "month", "is_weekend"),
}


def feature_group(name: str) -> str:
    if name.startswith("form_"):
        return "Forme récente"
    for group, keys in FEATURE_GROUPS.items():
        if any(k in name for k in keys):
            return group
    return "Autre"
