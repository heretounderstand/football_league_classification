import pickle  
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import os  
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import f1_score

# 📂 Dossiers de sauvegarde
os.makedirs("visualisations_finalisation", exist_ok=True)
os.makedirs("resultats_texte_finalisation", exist_ok=True)

# 📥 Charger X_train
X_train_path = "datasets_preprocesses/X_train_selected.csv"
X_train = pd.read_csv(X_train_path)

# 📌 Liste des modèles et noms formatés
models = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost"
}
feature_importance_data = {}

# 🏆 Mapping des ligues vers les pays
ligue_pays = {
    'Premier League': 'Angleterre', 'EFL Championship': 'Angleterre',
    'Ligue 1': 'France', 'Ligue 2': 'France',
    'Serie A': 'Italie', 'Serie B': 'Italie',
    'Bundesliga': 'Allemagne', 'Bundesliga 2': 'Allemagne',
    'LaLiga': 'Espagne', 'LaLiga 2': 'Espagne'
}

# 🔄 Charger les modèles et extraire les importances
for model_key, model_name in models.items():
    model_path = f"modeles_entraines/{model_key}_optimized.pkl"

    # 📥 Charger le modèle
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # 🔢 Extraire l'importance des features
    feature_importance = model.feature_importances_
    feature_names = X_train.columns  # X_train contient déjà 20 features

    # 📊 Créer un DataFrame
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # 💾 Sauvegarder en CSV et TXT
    csv_path = f"resultats_texte_finalisation/{model_key}_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # Stocker les données pour analyse
    feature_importance_data[model_key] = importance_df

    # 📉 Visualisation
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top Features par Importance - {model_name}')
    plt.tight_layout()
    img_path = f"visualisations_finalisation/{model_key}_feature_importance.png"
    plt.savefig(img_path)

# 🔄 Analyse des erreurs pour chaque modèle
for model_key, model_name in models.items():
    pred_path = f"datasets_evalues/predictions_{model_key}.csv"

    # 📥 Charger les prédictions
    predictions_df = pd.read_csv(pred_path)

    # 🔎 Identifier les erreurs
    predictions_df['erreur'] = predictions_df['y_true'] != predictions_df['y_pred']
    erreurs_df = predictions_df[predictions_df['erreur'] == True]

    # 📌 Ajouter les pays associés aux ligues
    erreurs_df['pays_true'] = erreurs_df['y_true'].map(lambda x: ligue_pays.get(x, 'Autre'))
    erreurs_df['pays_pred'] = erreurs_df['y_pred'].map(lambda x: ligue_pays.get(x, 'Autre'))

    # 📊 Analyser les erreurs par pays
    erreurs_df['meme_pays'] = erreurs_df['pays_true'] == erreurs_df['pays_pred']
    erreurs_meme_pays = erreurs_df['meme_pays'].sum()
    total_erreurs = len(erreurs_df)
    pourcentage_meme_pays = (erreurs_meme_pays / total_erreurs * 100) if total_erreurs > 0 else 0

    # 📉 Visualisation des erreurs par pays
    plt.figure(figsize=(10, 6))
    sns.countplot(y=erreurs_df['pays_true'], order=erreurs_df['pays_true'].value_counts().index)
    plt.title(f"Répartition des erreurs par pays - {model_name}")
    plt.xlabel("Nombre d'erreurs")
    plt.ylabel("Pays (y_true)")
    plt.tight_layout()
    img_path = f"visualisations_finalisation/{model_key}_erreurs_par_pays.png"
    plt.savefig(img_path)

    # 💾 Sauvegarder les résultats
    erreurs_df.to_csv(f"resultats_texte_finalisation/{model_key}_erreurs.csv", index=False)
    with open(f"resultats_texte_finalisation/{model_key}_erreurs.txt", "w") as file:
        file.write(f"Modèle : {model_name}\n")
        file.write(f"Total erreurs: {total_erreurs}\n")
        file.write(f"Erreurs dans le même pays: {erreurs_meme_pays} ({pourcentage_meme_pays:.2f}%)\n")

# 📥 Charger les données
X_path = "datasets_preprocesses/X_train_selected.csv"
y_path = "datasets_preprocesses/y_train.csv"

X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()  # S'assurer que y est un vecteur

# 🔄 Tester la robustesse avec différentes seeds
seeds = list(range(5))
results = {}

for model_key, model_name in models.items():
    model_path = f"modeles_entraines/{model_key}_optimized.pkl"

    # 📥 Charger le modèle
    with open(model_path, "rb") as file:
        best_model = pickle.load(file)

    scores = []
    
    for seed in seeds:
        # 🔀 Split des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        
        # Initialiser et ajuster l'encodeur sur `y`
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # 🎯 Entraînement et prédiction
        best_model.fit(X_train, y_train_encoded)
        y_pred = best_model.predict(X_test)
        
        y_pred = label_encoder.inverse_transform(y_pred)

        # 🏆 Calcul du F1-score
        scores.append(f1_score(y_test, y_pred, average='weighted'))

    # 📊 Sauvegarder les résultats
    results[model_key] = scores
    csv_path = f"resultats_texte_finalisation/{model_key}_robustesse.csv"
    txt_path = f"resultats_texte_finalisation/{model_key}_robustesse.txt"

    pd.DataFrame({"Seed": seeds, "F1-score": scores}).to_csv(csv_path, index=False)

    with open(txt_path, "w") as file:
        file.write(f"Modèle : {model_name}\n")
        file.write(f"F1-scores avec différentes seeds: {scores}\n")
        file.write(f"Moyenne: {np.mean(scores):.4f}, Écart-type: {np.std(scores):.4f}\n")

    # 📉 Visualisation
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=seeds, y=scores, marker='o')
    plt.xlabel("Seed")
    plt.ylabel("F1-score")
    plt.title(f"F1-score par Seed - {model_name}")
    plt.xticks(seeds)
    plt.ylim(0, 1)
    plt.grid(True)

    img_path = f"visualisations_finalisation/{model_key}_robustesse.png"
    plt.savefig(img_path)
    
    # 🔄 Générer et sauvegarder les learning curves
for model_key, model_name in models.items():
    model_path = f"modeles_entraines/{model_key}_optimized.pkl"

    # 📥 Charger le modèle
    with open(model_path, "rb") as file:
        best_model = pickle.load(file)

    # Initialiser et ajuster l'encodeur sur `y`
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 📊 Calcul des learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y_encoded, cv=5, scoring='f1_weighted', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))

    # 📈 Calcul des moyennes et écarts-types
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 📊 Sauvegarde des résultats sous format CSV
    learning_curve_df = pd.DataFrame({
        "Training Size": train_sizes,
        "Train Mean": train_mean,
        "Train Std": train_std,
        "Test Mean": test_mean,
        "Test Std": test_std
    })
    csv_path = f"resultats_texte_finalisation/{model_key}_learning_curve.csv"
    learning_curve_df.to_csv(csv_path, index=False)

    # 📉 Visualisation
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation score')
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend(loc="best")
    plt.tight_layout()

    # 📸 Sauvegarde du graphique
    img_path = f"visualisations_finalisation/{model_key}_learning_curve.png"
    plt.savefig(img_path)



