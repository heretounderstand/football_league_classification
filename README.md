# Quelle ligue ? Classification de matchs de football

Peut-on reconnaître **la ligue d'un match** parmi les 10 premières et deuxièmes divisions d'Angleterre,
d'Espagne, d'Italie, d'Allemagne et de France à partir de ses **statistiques** (tirs, fautes,
cartons, corners, buts) ?

Ce dépôt contient le pipeline d'analyse. Les résultats sont présentés dans le dashboard Streamlit
[`football_league_classification_dashboard`](https://github.com/heretounderstand/football_league_classification_dashboard).

## Résultats (test : saisons 2021-22 et 2022-23, 7 809 matchs)

| Modèle | Variables | Accuracy [IC 95 %] | F1 macro | Top-2 | Pays correct | Log-loss |
|---|---|---|---|---|---|---|
| Baseline (classe la plus fréquente) | — | 14.1 % [13.4–14.9] | 2.5 % | 26.0 % | 23.9 % | 2.293 |
| Régression logistique | stats du match | 17.6 % [16.8–18.3] | 14.4 % | 34.0 % | 32.1 % | 2.225 |
| LightGBM | stats du match | 18.5 % [17.7–19.4] | 15.4 % | 34.1 % | 33.1 % | 2.217 |
| Régression logistique | + contexte calendaire | 44.1 % [43.0–45.3] | 42.6 % | 64.8 % | 56.7 % | 1.594 |
| LightGBM | + contexte calendaire | 48.1 % [47.1–49.3] | 46.8 % | 67.8 % | 60.2 % | 1.471 |
| Régression logistique | + forme récente | 48.8 % [47.7–49.8] | 48.3 % | 73.4 % | 67.1 % | 1.418 |
| **LightGBM** | **+ forme récente** | **54.5 % [53.4–55.7]** | **54.4 %** | **76.5 %** | **71.3 %** | **1.364** |

Ce qu'il faut retenir :

- **Un match isolé dit peu de chose** : avec ses seules statistiques, on fait à peine mieux que la baseline
  (18,5 % contre 14,1 %). Les distributions des 10 ligues se chevauchent largement.
- **Le calendrier est très informatif** : l'heure du coup d'envoi et le jour de la semaine font passer
  l'accuracy à 48 %. C'est un signal réel, mais il vient de la programmation TV, pas du jeu.
- **La forme récente** (moyennes des 5 matchs précédents des deux équipes) agrège plusieurs matchs du
  même championnat et monte à 54,5 %. Dans 71 % des cas, le pays est correct.
- Parmi les statistiques de jeu, ce sont **les fautes et les cartons** (culture arbitrale) qui distinguent
  le mieux les ligues, loin devant les tirs et les corners.

## Pourquoi l'ancienne version annonçait 96 % (et pourquoi c'était faux)

La première version (scripts conservés dans [`legacy/`](legacy/)) affichait 96,4 % d'accuracy. L'audit
[`scripts/audit_legacy.py`](scripts/audit_legacy.py) reproduit ce protocole à partir des données brutes,
puis retire une à une les sources de fuite :

| Scénario | Accuracy |
|---|---|
| A · Protocole d'origine (stats manquantes = 0, « force » des équipes, split aléatoire) | 92.8 % |
| A0 · **Seulement** la « force » des équipes, sans aucune statistique de match | **87.0 %** |
| A1 · Seulement la saison et l'indicateur « stats manquantes ? » | 21.7 % |
| B · A sans la force des équipes | 32.1 % |
| C · B, uniquement les matchs avec de vraies statistiques | 33.7 % |
| D · C avec découpage temporel | 20.3 % |

1. **Fuite par l'identité des équipes.** `Home/Away_Team_Strength` (points par match calculés sur *tout*
   le jeu, test compris) prend une valeur unique par club. Le modèle reconnaissait le club, et le club donne
   la ligue. C'étaient les deux variables les plus importantes du LightGBM.
2. **Fuite par les valeurs manquantes.** 48 % des matchs n'ont pas de statistiques, et l'absence dépend de
   la ligue et de la saison (pas de stats en divisions 2 avant 2017). Remplacer les trous par 0 encodait
   la cible.
3. **Découpage aléatoire** de données temporelles : des matchs de la même saison et des mêmes équipes se
   retrouvaient à la fois en entraînement et en test.
4. Autres défauts : l'heure manquante remplacée par `0` était classée « Morning », la PCA et la LDA étaient
   calculées sans servir aux modèles, et aucune baseline n'était comparée.

## Méthode

1. **Données** (`flc/data.py`) : 113 514 matchs bruts de football-data.co.uk (1993-2023). On ne garde que
   les saisons où **les 10 ligues ont des statistiques** (2017-18 → 2022-23) et les matchs dont les
   statistiques sont complètes et cohérentes, soit 22 870 matchs. Aucune imputation des statistiques.
2. **Variables** (`flc/features.py`), en trois niveaux cumulés :
   - `match` : statistiques domicile / extérieur, totaux, écarts, ratios (précision des tirs, fautes par carton…) ;
   - `context` : heure du coup d'envoi (heure britannique, disponible depuis 2019-20 pour toutes les ligues), jour, mois ;
   - `form` : moyennes glissantes des 5 matchs **précédents** de chaque équipe (`shift(1)`, jamais le match
     courant ni le futur).

   Aucun nom d'équipe, aucun arbitre et aucune variable dérivée de la cible.
3. **Découpage temporel** (`flc/config.py`) : entraînement sur 2017-18 → 2019-20, réglage des
   hyperparamètres sur 2020-21, test sur 2021-22 et 2022-23 en **walk-forward**. Chaque saison de test est
   prédite par un modèle ré-entraîné sur toutes les saisons précédentes. Ce protocole est nécessaire car
   les données dérivent : par exemple, les tirs en Serie A passent de 25,6 à 20,4 par match entre 2017-18
   et 2018-19, probablement à cause d'un changement de fournisseur.
4. **Modèles** (`flc/modeling.py`) : baseline (prior), régression logistique multinomiale (heure, jour et
   mois en one-hot) et LightGBM (recherche aléatoire avec arrêt précoce sur la log-loss de validation).
   Le modèle retenu est celui qui a la meilleure log-loss **de validation**.
5. **Évaluation** (`flc/evaluation.py`) : accuracy, accuracy équilibrée, F1 macro, top-2, log-loss, pays et
   division corrects (probabilités agrégées), intervalles de confiance par bootstrap, courbe de calibration
   et importance par permutation (par variable et par famille de variables).

## Reproduire

```bash
pip install -r requirements.txt
python scripts/audit_legacy.py      # ~2 min  -> artifacts/legacy_audit.json
python -m flc.pipeline              # ~12 min -> artifacts/
pytest -q                           # tests anti-fuite
```

Le dossier `artifacts/` (4,5 Mo) contient tout ce que lit le dashboard : métriques, prédictions du test,
calibration, importances, profils des ligues et modèle du simulateur. Pour mettre à jour le dashboard :

```bash
cp artifacts/* ../football_league_classification_dashboard/data/
```

## Structure

```
data/raw/matches_raw.csv   données sources (football-data.co.uk)
flc/                       package : config, data, features, modeling, evaluation, pipeline
scripts/audit_legacy.py    audit chiffré de l'ancienne méthode
tests/                     tests (pas de fuite temporelle, pas d'identifiant parmi les variables…)
artifacts/                 sorties consommées par le dashboard
legacy/                    scripts de la première version, pour mémoire
```

## Limites et pistes

- Le modèle est légèrement **surconfiant** sur le test (dérive entre saisons). Un recalibrage sur la saison
  la plus récente améliorerait les probabilités.
- Le contexte calendaire domine. Pour isoler le « style de jeu », on peut regarder les modèles `match`
  et `full` sans contexte (à ajouter comme jeu de variables).
- Les ruptures de collecte (fournisseur de statistiques) mériteraient une normalisation par saison.
