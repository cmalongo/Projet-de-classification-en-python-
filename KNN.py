import numpy as np
import pandas as pd

def calculate_distance(point1, point2, distance_type, p=3):

    point1 = np.sort(point1)
    point2 = np.sort(point2)

    if distance_type == "euclidean":
        return np.sqrt(np.sum((point1 - point2) **2)) 
    elif distance_type == "manhattan":
        return np.sum(np.abs(point1 - point2))
    elif distance_type == "minkowski":
        return np.sum(np.abs(point1 - point2) **p) ** (1 / p) 
    elif distance_type == "hamming":
        return np.sum(point1 != point2)
    else:
        raise ValueError("Type de distance non supporté. Choisissez parmi 'euclidean', 'manhattan', 'minkowski', ou 'hamming'.")

def knn_numpy(X_train, y_train, X_test, k, distance_type, p=3):

    predictions = []

    for test_point in X_test:
        distances = np.array([calculate_distance(test_point, train_point, distance_type, p) for train_point in X_train])

#Trouver les indices des K plus proches voisins
        k_indices = np.argpartition(distances, k)[:k]

        k_labels = y_train[k_indices]

#Trouver la classe majoritaire
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]

        predictions.append(majority_label)

    return np.array(predictions)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Vérification des colonnes
if "Id" not in train_data.columns or "Label" not in train_data.columns:
    raise ValueError("Le fichier train.csv doit contenir les colonnes 'Id' et 'Label'.")
if "Id" not in test_data.columns:
    raise ValueError("Le fichier test.csv doit contenir la colonne 'Id'.")

X_train = train_data.drop(columns=["Id", "Label"]).to_numpy()
y_train = train_data["Label"].to_numpy()
X_test = test_data.drop(columns=["Id"]).to_numpy()
test_ids = test_data["Id"].to_numpy()
if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        f"Les dimensions des caractéristiques ne correspondent pas : "
        f"X_train a {X_train.shape[1]} colonnes, mais X_test en a {X_test.shape[1]}."
    )

k = 3
distance_type = "manhattan"
# Paramètre par défault de la distance de Minkowski
p = 3

y_pred = knn_numpy(X_train, y_train, X_test, k, distance_type, p)

predictions = pd.DataFrame({'Id': test_ids, 'Label': y_pred})
predictions.to_csv("predictions.csv", index=False)

print("Prédictions sauvegardées dans le fichier 'predictions.csv'.")