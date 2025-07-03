import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,)
        """
        # poids initiaux
        n_samples, n_features = X.shape 
        self.weights = np.random.randn(n_features)
        self.bias = 0.0
        # Liste pour suvre les erreurs par époque
        self.errors_ = [] 

        for _ in tqdm(range(max_epochs), desc="Entraînement du perceptron"):
            errors = 0
            for x, y_true in zip(X, y):
                # fonction linéaire
                linear_output = np.dot(x, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                error = y_true - y_pred
                # Si la prédiction est incorrecte mise à jour du poids et des biais
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * x
                    self.bias += update
                    errors += 1
            self.errors_.append(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        # retourne 0 ou 1
        return (linear_output >= 0).astype(int) 

    def score(self, X, y):
        # retourne le % de bonnes réponse
        return np.mean(self.predict(X) == y) 


#AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
#OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])
#XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

def train_and_show(X, y, title):
    perceptron = PerceptronSimple(learning_rate=0.1)
    perceptron.fit(X, y)
    acc = perceptron.score(X, y)
    print(f"Précision du perceptron ({title}) :", acc)

    plt.figure(figsize=(6, 6))
    for i, (x, target) in enumerate(zip(X, y)):
        if target == 1:
            plt.scatter(x[0], x[1], color='blue', marker='o', label='1 (True)' if i == 0 else "")
        else:
            plt.scatter(x[0], x[1], color='red', marker='x', label='0 (False)' if i == 0 else "")

    # Frontière de décision
    w, b = perceptron.weights, perceptron.bias
    x_vals = np.linspace(-0.2, 1.2, 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', label='décision')
    else:
        plt.axvline(-b / w[0], color='k', linestyle='--', label='décision')

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Perceptron avec le : {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

train_and_show(X_and, y_and, "AND")
train_and_show(X_or, y_or, "OR")
train_and_show(X_xor, y_xor, "XOR")
