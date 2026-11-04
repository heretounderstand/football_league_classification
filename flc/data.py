"""Chargement et nettoyage des données brutes.

Principes :
- aucune imputation des statistiques de match : une ligne sans statistiques est
  exclue de la modélisation (l'absence de données dépend de la ligue et de la
  saison, l'imputer reviendrait à encoder la cible) ;
- les identifiants (équipes, arbitre) sont conservés pour l'affichage et le
  calcul de la forme récente, jamais utilisés comme variables explicatives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FIRST_SEASON, LEAGUES, RAW_DATA, STAT_COLUMNS


def season_start_year(season: str) -> int:
    """'93-94' -> 1993, '05-06' -> 2005."""
    yy = int(str(season).split("-")[0])
    return 1900 + yy if yy >= 50 else 2000 + yy


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Time": "string", "Referee": "string"})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["season"] = df["Season"].map(season_start_year)
    df["season_label"] = df["season"].map(lambda y: f"{y}-{str(y + 1)[-2:]}")
    df["kickoff_hour"] = pd.to_numeric(df["Time"].str.slice(0, 2), errors="coerce").astype(float)
    df["league"] = df["Div"]
    df["country"] = df["league"].map(lambda l: LEAGUES[l][0])
    df["tier"] = df["league"].map(lambda l: LEAGUES[l][1])
    df["has_stats"] = df[STAT_COLUMNS].notna().all(axis=1)
    df = df.sort_values(["Date", "league", "HomeTeam"], kind="stable").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    return df


def quality_issues(df: pd.DataFrame) -> pd.Series:
    """Masque des lignes incohérentes (tirs cadrés > tirs, valeurs négatives...)."""
    s = df
    bad = (s["HST"] > s["HS"]) | (s["AST"] > s["AS"])
    bad |= (s[STAT_COLUMNS] < 0).any(axis=1)
    bad |= s["HTHG"] > s["FTHG"]
    bad |= s["HTAG"] > s["FTAG"]
    return bad.fillna(False)


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Matchs exploitables : saisons couvertes par les 10 ligues, stats complètes, cohérentes."""
    keep = (df["season"] >= FIRST_SEASON) & df["has_stats"] & ~quality_issues(df)
    keep &= df[["HTHG", "HTAG"]].notna().all(axis=1)
    return df.loc[keep].reset_index(drop=True)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Part des matchs avec statistiques complètes par ligue et saison (format long)."""
    cov = (
        df.groupby(["league", "season"])
        .agg(matches=("match_id", "size"), with_stats=("has_stats", "sum"))
        .reset_index()
    )
    cov["coverage"] = cov["with_stats"] / cov["matches"]
    return cov
