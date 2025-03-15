# KNN Classification Challenge

## Description

Ce projet implémente un algorithme des k plus proches voisins (**K-Nearest Neighbors, KNN**) en utilisant **NumPy** et **Pandas**. 
Il est conçu pour effectuer des classifications sur des ensembles de données fournis sous forme de fichiers CSV. 
L'algorithme prend en charge plusieurs métriques de distance :

- Distance Euclidienne
- Distance de Manhattan
- Distance de Minkowski
- Distance de Hamming

Le modèle est optimisé pour fonctionner avec **K=3** et la distance de Manhattan, après une série d’expérimentations sur la plateforme **Kaggle**.

## Auteurs

- **Gabriel Maccione** (Pseudo Kaggle : Kirbybrave)
- **Cyrille Malongo** (Pseudo Kaggle : PLUTO)

## Fonctionnalités

- Chargement des données depuis des fichiers CSV
- Calcul des distances entre les points
- Prédiction de la classe des points de test
- Sauvegarde des prédictions au format CSV

## Installation

1. Assurez-vous d'avoir **Python 3** installé.
2. Installez les dépendances requises avec la commande suivante :
   ```bash
   pip install numpy pandas
   ```
3. Placez vos fichiers `train.csv` et `test.csv` dans le répertoire du projet.

## Utilisation

Exécutez le script **KNN (4).py** avec la commande suivante :

```bash
python KNN (4).py
```

Les prédictions seront enregistrées dans un fichier **predictions.csv**.

## Format des données

### `train.csv` (Données d'entraînement)

| Id  | Feature1 | Feature2 | Label |
|-----|---------|---------|-------|
| 1   | 3.2     | 1.5     | A     |

- **Id** : Identifiant unique
- **FeatureX** : Caractéristiques numériques
- **Label** : Classe associée

### `test.csv` (Données de test)

| Id  | Feature1 | Feature2 |
|-----|---------|---------|
| 101 | 3.5     | 1.6     |

- **Id** : Identifiant unique
- **FeatureX** : Caractéristiques numériques
- **Aucune colonne Label**, car c'est ce que nous devons prédire.

## Méthodologie

1. **Calcul des distances** entre chaque point de test et tous les points d’entraînement.
2. **Sélection des K plus proches voisins** en fonction de la distance choisie.
3. **Vote majoritaire** parmi les labels des voisins sélectionnés.
4. **Sauvegarde des résultats** sous forme de fichier CSV.

## Résultats et Optimisation

- Le choix de la **distance de Manhattan** et **K=3** donnait les meilleurs résultats.
- L’ajout de **vérifications d’erreurs** a permis d’améliorer la robustesse du code.
- L’algorithme initial a été **simplifié** pour éviter le sur-apprentissage.

## Améliorations possibles

- Implémentation d’une version plus optimisée avec **scikit-learn**
- Ajout d’une fonction d’évaluation des performances du modèle
- Test d’autres valeurs de K et d’autres distances

## Licence

Projet réalisé dans le cadre d'un challenge Kaggle.
