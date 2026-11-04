"""Tests anti-fuite et de cohérence. Lancer avec ``pytest -q``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flc import config as C
from flc.data import load_raw, modelling_frame, season_start_year
from flc.evaluation import metrics
from flc.features import build_feature_sets, form_features


@pytest.fixture(scope="module")
def frame():
    return modelling_frame(load_raw())


def test_season_parsing():
    assert season_start_year("93-94") == 1993
    assert season_start_year("05-06") == 2005
    assert season_start_year("22-23") == 2022


def test_modelling_frame_has_real_stats_only(frame):
    assert frame[C.STAT_COLUMNS].notna().all().all()
    assert frame["season"].min() >= C.FIRST_SEASON
    assert (frame["HST"] <= frame["HS"]).all()


def test_splits_are_disjoint_and_ordered():
    assert max(C.TRAIN_SEASONS) < min(C.VALID_SEASONS) < min(C.TEST_SEASONS)
    assert not set(C.TRAIN_SEASONS) & set(C.TEST_SEASONS)


def test_no_identifier_or_target_in_features(frame):
    X, sets = build_feature_sets(frame.head(500))
    forbidden = ("team", "referee", "league", "div", "country", "tier", "season", "ftr", "result")
    for cols in sets.values():
        for c in cols:
            assert not any(f in c.lower() for f in forbidden), c


def test_form_uses_only_past_matches(frame):
    """Modifier un match ne doit changer ni sa propre forme ni celle des matchs antérieurs."""
    df = frame[frame["season"] == 2018].head(2000).reset_index(drop=True)
    base = form_features(df)
    target = 1500
    altered = df.copy()
    altered.loc[target, C.STAT_COLUMNS] = altered.loc[target, C.STAT_COLUMNS] * 10 + 50
    new = form_features(altered)
    earlier_or_same = df["Date"] <= df.loc[target, "Date"]
    pd.testing.assert_frame_equal(base[earlier_or_same], new[earlier_or_same])
    # ...mais influence bien les matchs suivants des deux équipes
    teams = {df.loc[target, "HomeTeam"], df.loc[target, "AwayTeam"]}
    later = (~earlier_or_same) & (df["HomeTeam"].isin(teams) | df["AwayTeam"].isin(teams))
    assert not base[later].equals(new[later])


def test_metrics_perfect_and_hierarchy():
    y = np.arange(len(C.CLASSES))
    proba = np.eye(len(C.CLASSES))
    m = metrics(y, proba)
    assert m["accuracy"] == 1 and m["country_accuracy"] == 1 and m["tier_accuracy"] == 1
    # Confondre Premier League et Championship : pays juste, niveau faux, ligue fausse
    wrong = proba.copy()
    wrong[0] = np.eye(len(C.CLASSES))[1]
    m = metrics(y, wrong)
    assert m["accuracy"] < 1 and m["country_accuracy"] == 1 and m["tier_accuracy"] < 1
