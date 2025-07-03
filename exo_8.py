from matplotlib import pyplot as plt
from exercice_7 import generer_donnees_separables, PerceptronSimple


def analyser_convergence(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]):
    """
    Analyse la convergence pour différents taux d'apprentissage
    """
    plt.figure(figsize=(12, 8))
    for i, lr in enumerate(learning_rates):
        perceptron = PerceptronSimple(learning_rate=lr)
        perceptron.fit(X, y)

        plt.plot(perceptron.errors_, label=f"Taux d'apprentissage: {lr}", linestyle='-', marker='o')
        # TODO: Tracer les courbes de convergence
        pass
    plt.xlabel('Époques')
    plt.ylabel("Erreurs")
    plt.title("Convergence des différents taux d'apprentissage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
X, y = generer_donnees_separables(n_points=200, noise=2.0)
analyser_convergence(X, y)
