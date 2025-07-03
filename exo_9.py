import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class PerceptronSimple:
    """
    Perceptron à seuil 0 : renvoie 1 si w·x + b >= 0, sinon 0.
    """
    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate
        self.w = None
        self.b = None
        self.err_history = []

    def _act(self, s: float) -> int:
        """Fonction d’activation : 0/1"""
        return 1 if s >= 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 100) -> None:
        """
        Apprentissage du perceptron.
        X : (n_samples, n_features) — y : 0/1
        """
        n_features = X.shape[1]
        self.w = np.random.randn(n_features)
        self.b = 0.0
        self.err_history.clear()

        for _ in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = self._act(self.w @ xi + self.b)
                if y_pred != yi:
                    delta = self.lr * (yi - y_pred)
                    self.w += delta * xi
                    self.b += delta
                    errors += 1
            self.err_history.append(errors)
            if errors == 0:      # convergence atteinte
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b >= 0).astype(int)

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

class PerceptronMultiClasse:
    """
    Un perceptron binaire par classe (One-vs-Rest).
    """
    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate
        self.classes = None
        self.perceptrons = {}      # dict  {classe: PerceptronSimple}

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 100) -> None:
        self.classes = np.unique(y)
        self.perceptrons.clear()

        for c in tqdm(self.classes, desc="Entraînement One-vs-Rest"):
            y_bin = (y == c).astype(int)                 # classe c → 1 ; autres → 0
            p = PerceptronSimple(learning_rate=self.lr)
            p.fit(X, y_bin, max_epochs=max_epochs)
            self.perceptrons[c] = p

    def _all_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Matrice (n_samples, n_classes) des w·x + b"""
        scores = np.empty((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            scores[:, i] = self.perceptrons[c].raw_score(X)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._all_raw_scores(X)
        indices = np.argmax(scores, axis=1)
        return self.classes[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._all_raw_scores(X)

def charger_donnees_iris_binaire():
    iris = load_iris()
    # longueur sépales, longueur pétales
    X = iris.data[:, [0, 2]]
    y = iris.target
    # on garde seulement les classes 0 et 1
    mask = y < 2
    # y reste 0/1 — surtout pas -1/+1
    return X[mask], y[mask]


def charger_donnees_iris_complete():
    iris = load_iris()
    # mêmes 2 features pour la visu
    X = iris.data[:, [0, 2]]
    y = iris.target
    return X, y, iris.target_names


def visualiser_iris(X, y, target_names=None, title="Dataset Iris"):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    markers = ['*', '+', 'o']

    for i in range(len(np.unique(y))):
        mask = (y == i)
        label = target_names[i] if target_names is not None else f'Classe {i}'
        plt.scatter(X[mask, 0], X[mask, 1],c=colors[i], marker=markers[i], s=90,label=label, alpha=0.7)

    plt.xlabel('Longueur des sépales (cm)')
    plt.ylabel('Longueur des pétales (cm)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # --- version multi-classe sur Iris complet -------------------
    X_full, y_full, nom_classes = charger_donnees_iris_complete()
    # normalisation (recommandée pour les perceptrons)
    scaler = StandardScaler().fit(X_full)
    X_full = scaler.transform(X_full)
    # split train / test
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.3, stratify=y_full, random_state=0)
    # entraînement
    clf = PerceptronMultiClasse(learning_rate=0.1)
    clf.fit(X_tr, y_tr, max_epochs=100)
    y_pred = clf.predict(X_te)
    print("Accuracy test :", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=nom_classes))
    visualiser_iris(X_full, y_full, nom_classes,title="Iris (2 features) – points + zones ambiguës")
