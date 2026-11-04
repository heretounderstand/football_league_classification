import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
import warnings
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                            precision_recall_fscore_support, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize, LabelEncoder
from scipy.stats import randint, uniform

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Création des dossiers nécessaires s'ils n'existent pas
os.makedirs('resultats_texte_classification', exist_ok=True)
os.makedirs('visualisations_classification', exist_ok=True)
os.makedirs('modeles_entraines', exist_ok=True)
os.makedirs('datasets_evalues', exist_ok=True)

# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES
print("Chargement des datasets prétraités...")

# Nous allons utiliser les données avec features sélectionnées
X_train = pd.read_csv('datasets_preprocesses/X_train_selected.csv')
X_test = pd.read_csv('datasets_preprocesses/X_test_selected.csv')
y_train = pd.read_csv('datasets_preprocesses/y_train.csv').squeeze()
y_test = pd.read_csv('datasets_preprocesses/y_test.csv').squeeze()

print(f"Dimensions des datasets chargés:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Vérification du nombre de classes
classes = np.unique(y_train)
n_classes = len(classes)
print(f"Nombre de classes: {n_classes}")
print(f"Classes: {classes}")

# Préparation pour ROC AUC (si multiclass)
if n_classes > 2:
    y_test_bin = label_binarize(y_test, classes=classes)
else:
    y_test_bin = y_test

# 2. MODÈLES DE BASE ET VALIDATION CROISÉE
print("\nÉvaluation des modèles de base avec validation croisée...")

# Définition des modèles de base
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    'Ridge Classifier': RidgeClassifier(random_state=42),
    'SGD Classifier': SGDClassifier(max_iter=1000, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False, thread_count=-1, loss_function='MultiClass', bootstrap_type="Bernoulli"),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

# Configuration de la validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fichier pour sauvegarder les résultats
cv_results_file = open('resultats_texte_classification/validation_croisee.txt', 'w')
cv_results_file.write("ÉVALUATION DES MODÈLES DE BASE AVEC VALIDATION CROISÉE\n")
cv_results_file.write("="*70 + "\n\n")

# DataFrame pour stocker les résultats
cv_results = pd.DataFrame(columns=['Modèle', 'Précision CV Moyenne', 'Écart-type', 'Temps d\'exécution (s)'])

# Initialiser et ajuster l'encodeur sur `y`
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialiser le DataFrame AVANT la boucle
cv_results = pd.DataFrame(columns=['Modèle', 'Précision CV Moyenne', 'Écart-type', 'Temps d\'exécution (s)'])

# Évaluation de chaque modèle avec validation croisée
for name, model in base_models.items():
    start_time = time.time()
    
    cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Nouvelle ligne sous forme de DataFrame
    new_row = pd.DataFrame([{
        'Modèle': name,
        'Précision CV Moyenne': cv_scores.mean(),
        'Écart-type': cv_scores.std(),
        'Temps d\'exécution (s)': elapsed_time
    }])

    # Concaténer la nouvelle ligne
    cv_results = pd.concat([cv_results, new_row], ignore_index=True)
    
    # Écriture dans le fichier
    cv_results_file.write(f"Modèle: {name}\n")
    cv_results_file.write(f"Précision CV moyenne: {cv_scores.mean():.4f}\n")
    cv_results_file.write(f"Écart-type: {cv_scores.std():.4f}\n")
    cv_results_file.write(f"Temps d'exécution: {elapsed_time:.2f} secondes\n")
    cv_results_file.write("-"*50 + "\n\n")

# Trouver le meilleur modèle
best_model_name = cv_results.loc[cv_results['Précision CV Moyenne'].idxmax(), 'Modèle']
best_model = base_models[best_model_name]

# Sauvegarder les résultats en CSV
cv_results.to_csv('resultats_texte_classification/cv_results_summary.csv', index=False)

# Visualisation des résultats de validation croisée
plt.figure(figsize=(12, 6))
sns.barplot(x='Précision CV Moyenne', y='Modèle', data=cv_results, palette='viridis')
plt.xlim(max(0, cv_results['Précision CV Moyenne'].min() - 0.05), 1.0)
plt.title('Comparaison des modèles de base (Validation croisée 5-fold)')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('visualisations_classification/comparaison_modeles_base.png')
plt.close()

cv_results_file.write("\nSYNTHÈSE DES RÉSULTATS:\n")
best_model = cv_results.loc[cv_results['Précision CV Moyenne'].idxmax()]
cv_results_file.write(f"Meilleur modèle: {best_model['Modèle']}\n")
cv_results_file.write(f"Précision CV moyenne: {best_model['Précision CV Moyenne']:.4f}\n")
cv_results_file.write(f"Modèle le plus rapide: {cv_results.loc[cv_results['Temps d\'exécution (s)'].idxmin()]['Modèle']}\n")
cv_results_file.close()

# 3. OPTIMISATION DES HYPERPARAMÈTRES
print("\nOptimisation des hyperparamètres...")

# Sélection des meilleurs modèles pour l'optimisation (top 3)
top_models = cv_results.nlargest(3, 'Précision CV Moyenne')['Modèle'].tolist()
print(f"Optimisation pour les modèles: {top_models}")

# Définition des espaces de recherche pour les hyperparamètres
param_grids = {
    'Logistic Regression': {
        'C': uniform(0.1, 10),
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'max_iter': [1000, 2000, 3000]
    },
    'Random Forest': {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    },
    'XGBoost': {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 1)
    },
    'LightGBM': {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(20, 150),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    },
    'Ridge Classifier': {
        'alpha': uniform(0.1, 10),
        'solver': ['auto', 'svd', 'cholesky', 'sparse_cg'],
        'tol': uniform(1e-4, 1e-3)
    },
    'SGD Classifier': {
        'alpha': uniform(1e-5, 1e-3),
        'l1_ratio': uniform(0, 1),
        'loss': ['log_loss', 'modified_huber'],
        'learning_rate': ['optimal', 'adaptive'],
        'eta0': uniform(0.01, 0.1)
    },
    'Extra Trees': {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    },
    'CatBoost': {
        'iterations': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'depth': randint(3, 10),
        'l2_leaf_reg': uniform(1, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bylevel': uniform(0.5, 0.5),
    },
    'Decision Tree': {
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'criterion': ['gini', 'entropy']
    },
    'AdaBoost': {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 2.0),
        'algorithm': ['SAMME', 'SAMME.R']
    }
}

# Fichier pour les résultats d'optimisation
hp_results_file = open('resultats_texte_classification/optimisation_hyperparametres.txt', 'w')
hp_results_file.write("OPTIMISATION DES HYPERPARAMÈTRES\n")
hp_results_file.write("="*70 + "\n\n")

# Dictionnaire pour stocker les meilleurs modèles
best_models = {}

# Réalisation de l'optimisation pour les meilleurs modèles
for model_name in top_models:
    hp_results_file.write(f"Modèle: {model_name}\n")
    hp_results_file.write("-"*50 + "\n\n")
    
    model = base_models[model_name]
    param_grid = param_grids[model_name]
    
    # Configuration de la recherche aléatoire
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Nombre d'itérations limité pour les grands datasets
        cv=cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=2,  # Plus de détails sur l'erreur
        error_score='raise'
    )
    
    # Exécution de la recherche
    start_time = time.time()
    random_search.fit(X_train, y_train_encoded)
    end_time = time.time()
    
    # Enregistrement des résultats
    hp_results_file.write(f"Meilleurs hyperparamètres:\n{random_search.best_params_}\n\n")
    hp_results_file.write(f"Meilleur score de validation croisée: {random_search.best_score_:.4f}\n")
    hp_results_file.write(f"Temps d'exécution: {end_time - start_time:.2f} secondes\n\n")
    
    # Analyse des résultats de recherche
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    hp_results_file.write(f"Statistiques d'optimisation:\n")
    hp_results_file.write(f"Score moyen: {cv_results_df['mean_test_score'].mean():.4f}\n")
    hp_results_file.write(f"Score max: {cv_results_df['mean_test_score'].max():.4f}\n")
    hp_results_file.write(f"Score min: {cv_results_df['mean_test_score'].min():.4f}\n")
    hp_results_file.write(f"Écart-type: {cv_results_df['mean_test_score'].std():.4f}\n\n")
    
    # Visualisation des résultats d'optimisation
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cv_results_df.sort_values('mean_test_score'), 
                 y='mean_test_score', x=range(len(cv_results_df)))
    plt.xlabel('Itération (triée par score)')
    plt.ylabel('Score moyen de validation')
    plt.title(f'Progression du score pendant l\'optimisation - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'visualisations_classification/optimisation_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Stocker le meilleur modèle
    best_models[model_name] = random_search.best_estimator_
    
    hp_results_file.write("="*50 + "\n\n")

hp_results_file.close()

# 4. ÉVALUATION DES MODÈLES OPTIMISÉS
print("\nÉvaluation des modèles optimisés...")

# Fichier pour les résultats d'évaluation
eval_results_file = open('resultats_texte_classification/evaluation_modeles.txt', 'w')
eval_results_file.write("ÉVALUATION DES MODÈLES OPTIMISÉS\n")
eval_results_file.write("="*70 + "\n\n")

# DataFrame pour stocker les métriques d'évaluation
evaluation_results = pd.DataFrame(columns=['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

# Évaluation de chaque modèle optimisé
for model_name, model in best_models.items():
    eval_results_file.write(f"Modèle: {model_name}\n")
    eval_results_file.write("-"*50 + "\n\n")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    y_pred = label_encoder.inverse_transform(y_pred)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred)
    
    # Calcul de precision, recall, f1 (moyenne pondérée)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Calcul de ROC AUC (multi-class nécessite une approche one-vs-rest)
    if n_classes > 2:
        # Pour chaque classe, calculer ROC AUC one-vs-rest
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    # Nouvelle ligne sous forme de DataFrame
    new_row = pd.DataFrame([{
        'Modèle': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }])

    # Concaténer plutôt qu'utiliser .append()
    evaluation_results = pd.concat([evaluation_results, new_row], ignore_index=True)
    
    # Écrire les résultats détaillés
    eval_results_file.write(f"Accuracy: {accuracy:.4f}\n")
    eval_results_file.write(f"Precision (weighted): {precision:.4f}\n")
    eval_results_file.write(f"Recall (weighted): {recall:.4f}\n")
    eval_results_file.write(f"F1 Score (weighted): {f1:.4f}\n")
    eval_results_file.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    
    eval_results_file.write("Classification Report:\n")
    eval_results_file.write(class_report)
    eval_results_file.write("\n")
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.tight_layout()
    plt.savefig(f'visualisations_classification/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Visualisation ROC Curve (pour chaque classe si multiclass)
    plt.figure(figsize=(10, 8))
    
    if n_classes > 2:
        # One-vs-Rest ROC courbe pour chaque classe
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc_class = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Classe {class_name} (AUC = {roc_auc_class:.2f})')
    else:
        # Courbe ROC pour classification binaire
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc_class = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_class:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'visualisations_classification/roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Sauvegarder le modèle
    model_filename = f'modeles_entraines/{model_name.replace(" ", "_").lower()}_optimized.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Générer les prédictions pour analyse ultérieure
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    for i, class_name in enumerate(classes):
        predictions_df[f'prob_class_{class_name}'] = y_pred_proba[:, i]
    
    predictions_df.to_csv(f'datasets_evalues/predictions_{model_name.replace(" ", "_").lower()}.csv', index=False)
    
    eval_results_file.write("="*50 + "\n\n")

# Sauvegarder le résumé des évaluations
evaluation_results.to_csv('resultats_texte_classification/evaluation_summary.csv', index=False)

# Visualisation comparative des performances des modèles
plt.figure(figsize=(14, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
melted_df = pd.melt(evaluation_results, id_vars=['Modèle'], value_vars=metrics, 
                    var_name='Métrique', value_name='Score')

sns.barplot(x='Modèle', y='Score', hue='Métrique', data=melted_df)
plt.ylim(max(0, melted_df['Score'].min() - 0.05), 1.05)
plt.title('Comparaison des performances des modèles optimisés')
plt.xticks(rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualisations_classification/comparaison_performances_modeles.png')
plt.close()

# Identification du meilleur modèle global
best_model_name = evaluation_results.loc[evaluation_results['F1 Score'].idxmax()]['Modèle']
best_model_obj = best_models[best_model_name]

eval_results_file.write("\nSYNTHÈSE DES RÉSULTATS FINAUX:\n")
eval_results_file.write(f"Meilleur modèle global: {best_model_name}\n")
eval_results_file.write(f"F1 Score: {evaluation_results.loc[evaluation_results['Modèle'] == best_model_name, 'F1 Score'].values[0]:.4f}\n")
eval_results_file.write(f"Accuracy: {evaluation_results.loc[evaluation_results['Modèle'] == best_model_name, 'Accuracy'].values[0]:.4f}\n")
eval_results_file.write(f"ROC AUC: {evaluation_results.loc[evaluation_results['Modèle'] == best_model_name, 'ROC AUC'].values[0]:.4f}\n\n")

# Sauvegarder le meilleur modèle séparément
with open('modeles_entraines/best_model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

# Créer un fichier d'information sur le meilleur modèle
with open('resultats_texte_classification/best_model_info.txt', 'w') as f:
    f.write(f"INFORMATIONS SUR LE MEILLEUR MODÈLE\n")
    f.write("="*50 + "\n\n")
    f.write(f"Nom du modèle: {best_model_name}\n")
    f.write(f"Type: {type(best_model_obj).__name__}\n\n")
    f.write("Hyperparamètres:\n")
    for param, value in best_model_obj.get_params().items():
        f.write(f"- {param}: {value}\n")
    f.write("\nPerformances:\n")
    for metric in metrics:
        f.write(f"- {metric}: {evaluation_results.loc[evaluation_results['Modèle'] == best_model_name, metric].values[0]:.4f}\n")

eval_results_file.write("\nPROCESSUS TERMINÉ\n")
eval_results_file.write(f"Tous les modèles ont été entraînés, optimisés et évalués.\n")
eval_results_file.write(f"Les résultats détaillés sont disponibles dans le dossier 'resultats_texte_classification'.\n")
eval_results_file.write(f"Les visualisations_classification sont disponibles dans le dossier 'visualisations_classification'.\n")
eval_results_file.write(f"Les modèles entraînés sont sauvegardés dans le dossier 'modeles_entraines'.\n")
eval_results_file.close()

print("\nTraitement terminé avec succès.")
print(f"Meilleur modèle: {best_model_name}")
print(f"F1 Score: {evaluation_results.loc[evaluation_results['Modèle'] == best_model_name, 'F1 Score'].values[0]:.4f}")
print(f"Résultats sauvegardés dans les dossiers appropriés.")