import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# Création des dossiers nécessaires s'ils n'existent pas
os.makedirs('resultats_texte_preparation', exist_ok=True)
os.makedirs('visualisations_preparation', exist_ok=True)
os.makedirs('datasets_preprocesses', exist_ok=True)

# 1. CHARGEMENT DES DONNÉES
print("Chargement du dataset...")
df = pd.read_csv('matches.csv')
df.drop(columns=[col for col in df.columns if col.startswith('League_') and col != 'League_Division'], inplace=True)
print(f"Dimensions du dataset: {df.shape}")

# Vérification que League_Division existe dans le dataset
if 'League_Division' not in df.columns:
    raise ValueError("La colonne 'League_Division' n'existe pas dans le dataset")

# 2. EXPLORATION DE LA VARIABLE CIBLE (LEAGUE_DIVISION)
print("\nAnalyse de la distribution de League_Division...")
target_distribution = df['League_Division'].value_counts()
target_percentage = df['League_Division'].value_counts(normalize=True) * 100

# Sauvegarder les résultats textuels
with open('resultats_texte_preparation/distribution_target.txt', 'w') as f:
    f.write("DISTRIBUTION DE LA VARIABLE CIBLE (LEAGUE_DIVISION)\n")
    f.write("="*50 + "\n\n")
    f.write("Nombre d'instances par classe:\n")
    f.write(str(target_distribution) + "\n\n")
    f.write("Pourcentage par classe:\n")
    f.write(str(target_percentage) + "\n\n")
    
    # Vérification du déséquilibre des classes
    min_class = target_distribution.min()
    max_class = target_distribution.max()
    ratio = max_class / min_class if min_class > 0 else float('inf')
    f.write(f"Ratio de déséquilibre (classe max / classe min): {ratio:.2f}\n")
    if ratio > 10:
        f.write("ALERTE: Fort déséquilibre entre les classes. Considérer des techniques de rééquilibrage.\n")
    elif ratio > 3:
        f.write("NOTE: Déséquilibre modéré entre les classes. Envisager des techniques de rééquilibrage.\n")
    else:
        f.write("Les classes semblent relativement équilibrées.\n")

# Visualisation de la distribution des classes
plt.figure(figsize=(12, 6))
sns.countplot(x='League_Division', data=df)
plt.title('Distribution des classes League_Division')
plt.xlabel('League_Division')
plt.ylabel('Nombre d\'instances')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualisations_preparation/distribution_classes.png')

# Création d'un pie chart pour la distribution en pourcentage
plt.figure(figsize=(10, 10))
plt.pie(target_distribution, labels=target_distribution.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution en pourcentage des classes League_Division')
plt.axis('equal')
plt.tight_layout()
plt.savefig('visualisations_preparation/distribution_classes_pie.png')

# 3. DIVISION DU DATASET
print("\nDivision du dataset en ensembles d'entraînement et de test...")
X = df.drop('League_Division', axis=1)
y = df['League_Division']

# Stratification pour maintenir la distribution des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vérification de la stratification
train_distribution = y_train.value_counts(normalize=True) * 100
test_distribution = y_test.value_counts(normalize=True) * 100

with open('resultats_texte_preparation/verification_stratification.txt', 'w') as f:
    f.write("VÉRIFICATION DE LA STRATIFICATION\n")
    f.write("="*50 + "\n\n")
    f.write(f"Taille de l'ensemble d'entraînement: {len(X_train)} instances ({len(X_train)/len(df):.1%})\n")
    f.write(f"Taille de l'ensemble de test: {len(X_test)} instances ({len(X_test)/len(df):.1%})\n\n")
    f.write("Distribution des classes dans l'ensemble d'entraînement:\n")
    f.write(str(train_distribution) + "\n\n")
    f.write("Distribution des classes dans l'ensemble de test:\n")
    f.write(str(test_distribution) + "\n\n")
    
    # Calcul de la différence de distribution entre train et test
    diff = abs(train_distribution - test_distribution).max()
    f.write(f"Différence maximale de distribution entre train et test: {diff:.2f}%\n")
    if diff < 1:
        f.write("La stratification est excellente.\n")
    elif diff < 3:
        f.write("La stratification est acceptable.\n")
    else:
        f.write("ALERTE: La stratification pourrait être améliorée.\n")

# 4. PRÉTRAITEMENT DES DONNÉES
print("\nPrétraitement des données...")

# Gestion des valeurs manquantes
missing_values = X_train.isnull().sum()
with open('resultats_texte_preparation/valeurs_manquantes.txt', 'w') as f:
    f.write("ANALYSE DES VALEURS MANQUANTES\n")
    f.write("="*50 + "\n\n")
    f.write("Nombre de valeurs manquantes par colonne:\n")
    f.write(str(missing_values[missing_values > 0]) + "\n\n")
    
    if missing_values.sum() == 0:
        f.write("Aucune valeur manquante détectée dans le dataset.\n")
    else:
        f.write(f"Total des valeurs manquantes: {missing_values.sum()}\n")
        f.write(f"Pourcentage de valeurs manquantes: {missing_values.sum() / (X_train.shape[0] * X_train.shape[1]):.2%}\n")

# Identification des colonnes catégorielles
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

# Encodage des variables catégorielles
if 'Match_Time_Category' in categorical_cols:
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cat = encoder.fit_transform(X_train[categorical_cols])
    encoded_cat_test = encoder.transform(X_test[categorical_cols])
    
    # Création de noms pour les colonnes encodées
    encoded_feature_names = []
    for i, category in enumerate(encoder.categories_):
        for j, value in enumerate(category[1:], 1):  # Skip the first category (dropped)
            encoded_feature_names.append(f"{categorical_cols[i]}_{value}")
    
    # Conversion en DataFrame
    encoded_train_df = pd.DataFrame(encoded_cat, columns=encoded_feature_names, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_cat_test, columns=encoded_feature_names, index=X_test.index)
    
    with open('resultats_texte_preparation/encodage_variables.txt', 'w') as f:
        f.write("ENCODAGE DES VARIABLES CATÉGORIELLES\n")
        f.write("="*50 + "\n\n")
        f.write(f"Variables catégorielles encodées: {categorical_cols}\n\n")
        f.write(f"Nouvelles variables après encodage one-hot: {encoded_feature_names}\n\n")
        f.write(f"Nombre de variables après encodage: {len(encoded_feature_names)}\n")

# Standardisation des variables numériques
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train[numerical_cols])
scaled_test = scaler.transform(X_test[numerical_cols])

# Conversion en DataFrame
scaled_train_df = pd.DataFrame(scaled_train, columns=numerical_cols, index=X_train.index)
scaled_test_df = pd.DataFrame(scaled_test, columns=numerical_cols, index=X_test.index)

# Combinaison des variables numériques standardisées et des variables catégorielles encodées
if len(categorical_cols) > 0:
    X_train_processed = pd.concat([scaled_train_df, encoded_train_df], axis=1)
    X_test_processed = pd.concat([scaled_test_df, encoded_test_df], axis=1)
else:
    X_train_processed = scaled_train_df
    X_test_processed = scaled_test_df

# 5. SÉLECTION DES FEATURES PERTINENTES
print("\nSélection des features pertinentes...")

# Utilisation de ANOVA F-value pour la sélection de caractéristiques
selector = SelectKBest(f_classif, k='all')
selector.fit(X_train_processed, y_train)
feature_scores = pd.DataFrame({
    'Feature': X_train_processed.columns,
    'F_Score': selector.scores_,
    'P_Value': selector.pvalues_
})
feature_scores = feature_scores.sort_values('F_Score', ascending=False)

# Sauvegarde des scores des features
feature_scores.to_csv('resultats_texte_preparation/feature_scores.csv', index=False)

with open('resultats_texte_preparation/selection_features.txt', 'w') as f:
    f.write("SÉLECTION DES FEATURES PERTINENTES (ANOVA F-TEST)\n")
    f.write("="*50 + "\n\n")
    f.write("Top 10 features les plus importantes:\n")
    f.write(str(feature_scores.head(10)) + "\n\n")
    
    # Identification des features non significatives (p > 0.05)
    non_significant = feature_scores[feature_scores['P_Value'] > 0.05]
    if len(non_significant) > 0:
        f.write("Features statistiquement non significatives (p > 0.05):\n")
        f.write(str(non_significant) + "\n\n")
        f.write(f"Nombre de features non significatives: {len(non_significant)}\n\n")
    else:
        f.write("Toutes les features sont statistiquement significatives (p < 0.05).\n\n")

# Visualisation des scores des features
plt.figure(figsize=(14, 10))
sns.barplot(x='F_Score', y='Feature', data=feature_scores.head(20))
plt.title('Top 20 features selon le score F (ANOVA)')
plt.xlabel('Score F')
plt.tight_layout()
plt.savefig('visualisations_preparation/top_features_f_score.png')

# Sélection des k meilleures features
k = min(20, len(X_train_processed.columns))  # On limite à 20 ou moins
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train_processed, y_train)
X_test_selected = selector.transform(X_test_processed)

# Récupération des noms des features sélectionnées
selected_features = X_train_processed.columns[selector.get_support()]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

with open('resultats_texte_preparation/features_selectionnees.txt', 'w') as f:
    f.write(f"FEATURES SÉLECTIONNÉES (Top {k})\n")
    f.write("="*50 + "\n\n")
    f.write(str(selected_features.tolist()) + "\n\n")
    f.write(f"Nombre de features sélectionnées: {len(selected_features)}\n")

# Analyse en Composantes Principales (PCA)
pca = PCA()
pca.fit(X_train_processed)

# Calcul de la variance expliquée cumulée
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

with open('resultats_texte_preparation/analyse_pca.txt', 'w') as f:
    f.write("ANALYSE EN COMPOSANTES PRINCIPALES (PCA)\n")
    f.write("="*50 + "\n\n")
    f.write(f"Nombre de composantes expliquant 95% de la variance: {n_components_95}\n\n")
    f.write("Variance expliquée par composante:\n")
    for i, var in enumerate(pca.explained_variance_ratio_):
        f.write(f"Composante {i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulé)\n")

# Visualisation de la variance expliquée
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, label='Variance expliquée individuelle')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Variance expliquée cumulée')
plt.axhline(y=0.95, color='r', linestyle='--', label='Seuil de 95% de variance expliquée')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'Composantes nécessaires ({n_components_95})')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Ratio de variance expliquée')
plt.title('Analyse PCA: Variance expliquée par composante')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualisations_preparation/pca_variance_expliquee.png')

# Application de PCA avec 95% de variance
pca_95 = PCA(n_components=n_components_95)
X_train_pca = pca_95.fit_transform(X_train_processed)
X_test_pca = pca_95.transform(X_test_processed)

# Conversion en DataFrame
pca_cols = [f'PC{i+1}' for i in range(n_components_95)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)

# Analyse Discriminante Linéaire (LDA) si plus de 2 classes
n_classes = len(np.unique(y_train))
if n_classes > 2:
    n_components_lda = min(n_classes - 1, X_train_processed.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
    X_train_lda = lda.fit_transform(X_train_processed, y_train)
    X_test_lda = lda.transform(X_test_processed)
    
    # Conversion en DataFrame
    lda_cols = [f'LD{i+1}' for i in range(n_components_lda)]
    X_train_lda_df = pd.DataFrame(X_train_lda, columns=lda_cols, index=X_train.index)
    X_test_lda_df = pd.DataFrame(X_test_lda, columns=lda_cols, index=X_test.index)
    
    with open('resultats_texte_preparation/analyse_lda.txt', 'w') as f:
        f.write("ANALYSE DISCRIMINANTE LINÉAIRE (LDA)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Nombre de classes: {n_classes}\n")
        f.write(f"Nombre de composantes LDA: {n_components_lda}\n\n")
        f.write("Ratio de variance expliquée par composante:\n")
        explained_variance_ratio = lda.explained_variance_ratio_
        for i, var in enumerate(explained_variance_ratio):
            f.write(f"Composante LD{i+1}: {var:.4f}\n")
            
    le = LabelEncoder()
    y_train_numeric = le.fit_transform(y_train)  # Conversion des classes en numériques pour la visualisation  
    categories = le.classes_  # Récupérer les noms des catégories originales      
    
    # Visualisation LDA (2 premières composantes)
    if n_components_lda >= 2:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train_numeric, cmap='viridis', alpha=0.6)
        plt.title('Analyse LDA: Projection sur les 2 premières composantes')
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        # Ajout de la colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('League_Division')

        # Création de la légende manuelle
        handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                            markerfacecolor=scatter.cmap(scatter.norm(i)))
                for i in range(len(categories))]
        
        plt.legend(handles, categories, title="League_Division", loc="upper right")
        plt.tight_layout()
        plt.savefig('visualisations_preparation/lda_projection.png')

# SAUVEGARDE DES DATASETS PRÉTRAITÉS
print("\nSauvegarde des datasets prétraités...")

# Sauvegarde des ensembles originaux (avec target)
train_df = X_train.copy()
train_df['League_Division'] = y_train
test_df = X_test.copy()
test_df['League_Division'] = y_test

train_df.to_csv('datasets_preprocesses/train_original.csv', index=False)
test_df.to_csv('datasets_preprocesses/test_original.csv', index=False)

# Sauvegarde des ensembles prétraités
X_train_processed.to_csv('datasets_preprocesses/X_train_preprocessed.csv', index=False)
X_test_processed.to_csv('datasets_preprocesses/X_test_preprocessed.csv', index=False)
pd.Series(y_train).to_csv('datasets_preprocesses/y_train.csv', index=False)
pd.Series(y_test).to_csv('datasets_preprocesses/y_test.csv', index=False)

# Sauvegarde des ensembles avec features sélectionnées
X_train_selected_df.to_csv('datasets_preprocesses/X_train_selected.csv', index=False)
X_test_selected_df.to_csv('datasets_preprocesses/X_test_selected.csv', index=False)

# Sauvegarde des ensembles avec PCA
X_train_pca_df.to_csv('datasets_preprocesses/X_train_pca.csv', index=False)
X_test_pca_df.to_csv('datasets_preprocesses/X_test_pca.csv', index=False)

# Sauvegarde des ensembles avec LDA si applicable
if n_classes > 2:
    X_train_lda_df.to_csv('datasets_preprocesses/X_train_lda.csv', index=False)
    X_test_lda_df.to_csv('datasets_preprocesses/X_test_lda.csv', index=False)

print("\nTraitement terminé avec succès.")
print(f"Résultats textuels sauvegardés dans le dossier 'resultats_texte_preparation'")
print(f"Visualisations_preparation sauvegardées dans le dossier 'visualisations_preparation'")
print(f"Datasets prétraités sauvegardés dans le dossier 'datasets_preprocesses'")

# Récapitulatif du prétraitement
with open('resultats_texte_preparation/recapitulatif_pretraitement.txt', 'w') as f:
    f.write("RÉCAPITULATIF DU PRÉTRAITEMENT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Nombre total d'instances: {len(df)}\n")
    f.write(f"Instances d'entraînement: {len(X_train)} ({len(X_train)/len(df):.1%})\n")
    f.write(f"Instances de test: {len(X_test)} ({len(X_test)/len(df):.1%})\n\n")
    
    f.write(f"Features originales: {len(X.columns)}\n")
    f.write(f"Features après prétraitement: {X_train_processed.shape[1]}\n")
    f.write(f"Features sélectionnées (ANOVA): {len(selected_features)}\n")
    f.write(f"Composantes PCA (95% variance): {n_components_95}\n")
    if n_classes > 2:
        f.write(f"Composantes LDA: {n_components_lda}\n\n")
    
    f.write("Fichiers générés:\n")
    f.write("- train_original.csv, test_original.csv: Ensembles originaux avec target\n")
    f.write("- X_train_preprocessed.csv, X_test_preprocessed.csv: Données prétraitées\n")
    f.write("- y_train.csv, y_test.csv: Variables cibles\n")
    f.write("- X_train_selected.csv, X_test_selected.csv: Données avec features sélectionnées\n")
    f.write("- X_train_pca.csv, X_test_pca.csv: Données réduites par PCA\n")
    if n_classes > 2:
        f.write("- X_train_lda.csv, X_test_lda.csv: Données transformées par LDA\n")