
import numpy as np
import matplotlib.pyplot as plt
from PerceptronMultiCouches import PerceptronMultiCouches

def entrainer_et_visualiser(architecture, epochs=10000):
    # Données XOR
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])

    # Création du réseau
    reseau = PerceptronMultiCouches(architecture=architecture, activation='tanh', learning_rate=0.1)
    loss_history = []

    for epoch in range(epochs):
        output = reseau.forward(X)
        loss = np.mean((y - output) ** 2)
        loss_history.append(loss)
        gradient = 2 * (output - y)
        reseau.backward(gradient)

    # Courbe de loss
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Courbe de loss - Archi: {architecture}")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Surface de décision
    def plot_decision_surface(model):
        h = 0.01
        x_min, x_max = X[0].min() - 0.1, X[0].max() + 0.1
        y_min, y_max = X[1].min() - 0.1, X[1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()].T
        Z = model.calcule_sortie(grid)
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, levels=50, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[0], X[1], c=y[0], cmap=plt.cm.binary, edgecolors='k')
        plt.title(f"Surface de décision - Archi: {architecture}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.show()

    plot_decision_surface(reseau)

    # Visualisation des poids
    for i, couche in enumerate(reseau.layers):
        plt.figure()
        plt.imshow(couche.weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"Poids - Couche {i+1}")
        plt.xlabel("Entrées")
        plt.ylabel("Neurones")
        plt.show()

# Comparaison de performances selon l’architecture
architectures = [[2, 2, 1], [2, 3, 1], [2, 4, 1]]
for arch in architectures:
    entrainer_et_visualiser(arch)
