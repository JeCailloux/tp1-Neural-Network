import numpy as np
from CoucheNeurones import CoucheNeurones

class PerceptronMultiCouches:
    def __init__(self, architecture, activation='sigmoid', learning_rate=0.01):
        # architecture = liste du nombre de neurones par couche ex [2, 3, 1] veut dire 2 en entrée, 3 cachés, 1 en sortie
        self.layers = []
        self.learning_rate = learning_rate

        # on crée chaque couche une par une
        for i in range(len(architecture) - 1):
            couche = CoucheNeurones(
                n_input=architecture[i],
                n_neurons=architecture[i + 1],
                activation=activation,
                learning_rate=learning_rate
            )
            self.layers.append(couche)

    def forward(self, X):

        # on passe l entrée dans toutes les couches une par une
        for layer in self.layers:
            X = layer.forward(X)
        return X  # c est la sortie finale du réseau

    def backward(self, gradient):
        # on fait la rétropropagation depuis la dernière couche jusqu à la première
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def entraine(self, X, y, epochs=1000):
        # on entraîne le réseau plusieurs fois sur les données
        for epoch in range(epochs):
            output = self.forward(X)  # passe avant
            loss = np.mean((y - output) ** 2)  # perte quadratique simple
            if epoch % 100 == 0:
                print(f"epoch {epoch} perte {loss:.4f}")
            gradient = 2 * (output - y)  # dérivée de la perte MSE
            self.backward(gradient)  # rétropropagation

    def calcule_sortie(self, X):
        # juste un alias plus lisible pour faire une prédiction
        return self.forward(X)

if __name__ == "__main__":
    # exemple simple pour XOR
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  # 4 exemples
    y = np.array([[0, 1, 1, 0]])  # sortie XOR

    mlp = PerceptronMultiCouches([2, 3, 1], activation='tanh', learning_rate=0.1)
    mlp.entraine(X, y, epochs=10000)

    pred = mlp.calcule_sortie(X)
    print("sorties prédites :", pred)
