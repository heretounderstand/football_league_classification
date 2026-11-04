import pandas as pd

# Charger le dataset
df = pd.read_csv("matches_original.csv")

# 1. Supprimer les colonnes avec trop de valeurs manquantes (plus de 50%)
# Match_Referee: 90652 valeurs manquantes (>70%)
df = df.drop(columns=['Match_Referee'])

# 2. Pour Match_Time, créer une variable catégorielle indiquant le moment de la journée
df['Match_Time_Category'] = df['Match_Time'].apply(
    lambda x: 'Unknown' if pd.isna(x) else 
             'Morning' if x < '12:00' else
             'Afternoon' if x < '18:00' else 
             'Evening'
)

# 3. Pour les statistiques de match avec valeurs manquantes (~55000)

## Option B: Remplacer par des valeurs médianes par ligue
for col in ['Home_Shots', 'Away_Shots', 'Home_Shots_On_Target', 'Away_Shots_On_Target',
           'Home_Fouls', 'Away_Fouls', 'Home_Corners', 'Away_Corners',
           'Home_Yellow_Cards', 'Away_Yellow_Cards', 'Home_Red_Cards', 'Away_Red_Cards']:
    df[col] = df.groupby('League_Division')[col].transform(
        lambda x: x.fillna(x.median())
    )

# 4. Pour les statistiques de mi-temps (moins de 10% manquantes)
for col in ['HalfTime_Home_Goals', 'HalfTime_Away_Goals']:
    # Remplacer par 0 si manquant
    df[col] = df[col].fillna(0)
    
df['HalfTime_Result'] = df['HalfTime_Result'].fillna('D')  # Supposer nul si manquant

# 1. Extraire des informations temporelles
# Détecter et convertir les formats mixtes de dates
def convert_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
        
    # Format avec année à 2 chiffres (ex: '04/08/17')
    try:
        date = pd.to_datetime(date_str, format='%d/%m/%y')
        # Ajustement pour les dates avant 2000
        if date.year > 2025:
            date = date.replace(year=date.year-100)
        return date
    except:
        # Format avec année à 4 chiffres (ex: '15/02/1998')
        try:
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            return pd.NaT

# Appliquer la fonction à votre colonne
df['Match_Date'] = df['Match_Date'].apply(convert_date)

df['Match_Year'] = pd.to_datetime(df['Match_Date']).dt.year
df['Match_Month'] = pd.to_datetime(df['Match_Date']).dt.month
df['Match_Weekday'] = pd.to_datetime(df['Match_Date']).dt.weekday
df['Is_Weekend'] = df['Match_Weekday'].isin([5, 6]).astype(int)

# 2. Caractéristiques du match
df['Total_Goals'] = df['FullTime_Home_Goals'] + df['FullTime_Away_Goals']
df['Goal_Difference'] = abs(df['FullTime_Home_Goals'] - df['FullTime_Away_Goals'])
df['Home_Win'] = (df['FullTime_Result'] == 'H').astype(int)
df['Away_Win'] = (df['FullTime_Result'] == 'A').astype(int)
df['Draw'] = (df['FullTime_Result'] == 'D').astype(int)

# 3. Dynamique de match
df['Score_Changed_After_HT'] = (df['FullTime_Result'] != df['HalfTime_Result']).astype(int)

# 4. Statistiques relatives (si disponibles)
stats_columns = ['Shots', 'Shots_On_Target', 'Fouls', 'Corners', 'Yellow_Cards', 'Red_Cards']
for stat in stats_columns:
    home_col = f'Home_{stat}'
    away_col = f'Away_{stat}'
    if home_col in df.columns and away_col in df.columns:
        df[f'Total_{stat}'] = df[home_col] + df[away_col]
        df[f'Home_Dominance_{stat}'] = df[home_col] / df[f'Total_{stat}'].replace(0, 1)
        
# 1. Extraire la saison comme entier
df['Season_Start_Year'] = df['League_Season'].str.split('-').str[0].astype(int)
df['Season_Start_Year'] = df['Season_Start_Year'].apply(lambda x: x + 1900 if x > 25 else x + 2000)

# 2. Encodage One-Hot pour la variable cible (pour analyse exploratoire)
league_dummies = pd.get_dummies(df['League_Division'], prefix='League')
df = pd.concat([df, league_dummies], axis=1)

# 3. Encodage des équipes
# Chaque équipe reçoit un score de force basé sur ses performances historiques
team_strength = {}
for team in set(df['Home_Team'].unique()).union(set(df['Away_Team'].unique())):
    home_games = df[df['Home_Team'] == team]
    away_games = df[df['Away_Team'] == team]
    
    home_points = (home_games['FullTime_Result'] == 'H').sum() * 3 + (home_games['FullTime_Result'] == 'D').sum()
    away_points = (away_games['FullTime_Result'] == 'A').sum() * 3 + (away_games['FullTime_Result'] == 'D').sum()
    
    total_games = len(home_games) + len(away_games)
    if total_games > 0:
        team_strength[team] = (home_points + away_points) / total_games
    else:
        team_strength[team] = 0
        
df['Home_Team_Strength'] = df['Home_Team'].map(team_strength)
df['Away_Team_Strength'] = df['Away_Team'].map(team_strength)
df['Team_Strength_Difference'] = df['Home_Team_Strength'] - df['Away_Team_Strength']

# 1. Supprimer les colonnes redondantes ou non nécessaires
columns_to_drop = ['Match_Date', 'Match_Time', 'FullTime_Result', 'HalfTime_Result', 
                   'League_Season', 'Home_Team', 'Away_Team']
df_clean = df.drop(columns=columns_to_drop)

# Vérifier les valeurs manquantes restantes
missing_after_cleaning = df_clean.isnull().sum()
print("Colonnes avec valeurs manquantes restantes:")
print(missing_after_cleaning[missing_after_cleaning > 0])

# 2. Vérifier qu'il ne reste plus de valeurs manquantes
assert df_clean.isnull().sum().sum() == 0, "Il reste des valeurs manquantes!"

# Sauvegarde du fichier nettoyé
df_clean.to_csv('matches.csv', index=False)