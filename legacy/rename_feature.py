import pandas as pd

# Charger le dataset
df = pd.read_csv("matches_raw.csv")

# Renommage des colonnes
column_mapping = {
    'Season': 'League_Season',
    'Div': 'League_Division',
    'Date': 'Match_Date',
    'Time': 'Match_Time',
    'HomeTeam': 'Home_Team',
    'AwayTeam': 'Away_Team',
    'FTHG': 'FullTime_Home_Goals',
    'FTAG': 'FullTime_Away_Goals',
    'FTR': 'FullTime_Result',
    'HTHG': 'HalfTime_Home_Goals',
    'HTAG': 'HalfTime_Away_Goals',
    'HTR': 'HalfTime_Result',
    'Referee': 'Match_Referee',
    'HS': 'Home_Shots',
    'AS': 'Away_Shots',
    'HST': 'Home_Shots_On_Target',
    'AST': 'Away_Shots_On_Target',
    'HC': 'Home_Corners',
    'AC': 'Away_Corners',
    'HF': 'Home_Fouls',
    'AF': 'Away_Fouls',
    'HY': 'Home_Yellow_Cards',
    'AY': 'Away_Yellow_Cards',
    'HR': 'Home_Red_Cards',
    'AR': 'Away_Red_Cards'
}
df.rename(columns=column_mapping, inplace=True)

# Convertir la date au format standard
if 'Match_Date' in df.columns:
    df['Match_Date'] = pd.to_datetime(df['Match_Date'], dayfirst=True, errors='coerce')

# Supprimer les lignes avec des dates nulles (si des erreurs de conversion existent)
df.dropna(subset=['Match_Date'], inplace=True)

# Remplacer les valeurs manquantes par des zéros pour les stats numériques
df.fillna(0, inplace=True)

# Sauvegarde du fichier nettoyé
df.to_csv('matches_original.csv', index=False)

print("Dataset nettoyé et sauvegardé sous 'matches_original.csv'")
