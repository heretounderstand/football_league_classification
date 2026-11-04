"""Configuration centrale du projet : chemins, ligues, découpage temporel."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "matches_raw.csv"
ARTIFACTS = ROOT / "artifacts"

RANDOM_STATE = 42

# Ligue -> (pays, niveau). L'ordre fixe l'ordre des classes partout (figures, matrices).
LEAGUES: dict[str, tuple[str, int]] = {
    "Premier League": ("Angleterre", 1),
    "EFL Championship": ("Angleterre", 2),
    "LaLiga": ("Espagne", 1),
    "LaLiga 2": ("Espagne", 2),
    "Serie A": ("Italie", 1),
    "Serie B": ("Italie", 2),
    "Bundesliga": ("Allemagne", 1),
    "Bundesliga 2": ("Allemagne", 2),
    "Ligue 1": ("France", 1),
    "Ligue 2": ("France", 2),
}
CLASSES = list(LEAGUES)

# Statistiques de match (domicile/extérieur) issues de football-data.co.uk.
STAT_PAIRS = {
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "fouls": ("HF", "AF"),
    "corners": ("HC", "AC"),
    "yellow": ("HY", "AY"),
    "red": ("HR", "AR"),
}
STAT_COLUMNS = [c for pair in STAT_PAIRS.values() for c in pair]

# Découpage temporel (année de début de saison).
# Avant 2017-18, les divisions 2 (sauf Championship) n'ont pas de statistiques :
# on ne garde que les saisons où les 10 ligues sont renseignées, pour que la
# présence/absence de données ne trahisse pas la ligue.
FIRST_SEASON = 2017
TRAIN_SEASONS = (2017, 2018, 2019)
VALID_SEASONS = (2020,)
TEST_SEASONS = (2021, 2022)

# Fenêtre des moyennes glissantes "forme récente" (matchs précédents de l'équipe).
FORM_WINDOW = 5
