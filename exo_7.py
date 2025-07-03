import numpy as np
from matplotlib import pyplot as plt

from perceptron_simple import PerceptronSimple


def generer_donnees_separables(n_points=100, noise=0.1):
    """
    Génère deux classes de points linéairement séparables
    """
    np.random.seed(42)  # Réduit l'aléatoire pour avoir des résultats lissé

    X1 = np.random.randn(n_points // 2, 2) * noise + np.array([2, 2])
    X2 = np.random.randn(n_points // 2, 2) * noise + np.array([-2, -2])
    X = np.vstack((X1, X2))  # Fusion des classes 1 et 2
    y1 = np.ones(n_points // 2)
    y2 = np.zeros(n_points // 2)
    y = np.hstack((y1, y2))  # Fusion des étiquettes 1 et 2
    # Mélange aléatoire des valeurs 
    indices = np.random.permutation(n_points)
    X = X[indices]
    y = y[indices]
    # Normalisation des données
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y


def visualiser_donnees(X, y, w=None, b=None, title="Données"):
    """
    Visualise les données et optionnellement la droite de séparation
    """
    plt.figure(figsize=(8, 6))
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='Classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='Classe -1')
    # Tracer la droite si les paramètres sont fourni
    if w is not None and b is not None:
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', label='Frontière de décision')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


X, y = generer_donnees_separables(n_points=200, noise=2.0)
# Entraînement abdos pour Léo
perceptron = PerceptronSimple(learning_rate=0.1)
perceptron.fit(X, y)
visualiser_donnees(X, y, title="Données séparables", w=perceptron.weights, b=perceptron.bias)
