import numpy as np

class CoucheNeurones:
    def __init__(self, n_input, n_neurons, activation='sigmoid', learning_rate=0.01):
        # ici on prépare la couche avec le nombre d entrées et de neurones qu on veut
        # on choisit aussi la fonction d activation et le taux d apprentissage

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.learning_rate = learning_rate

        # ici on initialise les poids de façon aléatoire avec la méthode xavier pour que ça parte pas dans tous les sens
        limit = np.sqrt(6 / (n_input + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_input))
        self.bias = np.zeros((n_neurons, 1))  # les biais on les met à zéro au début

        # on garde en mémoire les dernières valeurs pour pouvoir les utiliser dans le backward
        self.last_input = None
        self.last_z = None
        self.last_activation = None

        # on prend la bonne fonction d activation depuis l autre fichier
        from activation_functions import ActivationFunction
        self.activation_func = ActivationFunction(activation)

    def forward(self, X):
        # ici c est le passage en avant on calcule la sortie des neurones
        self.last_input = X  # on garde l entrée pour plus tard
        self.last_z = np.dot(self.weights, X) + self.bias  # on fait le produit des poids avec l entrée et on ajoute le biais
        self.last_activation = self.activation_func.apply(self.last_z)  # on applique la fonction d activation

        return self.last_activation  # c est ça la sortie de la couche

    def backward(self, gradient_from_next_layer):
        # ici on fait la rétropropagation pour corriger les poids

        dz = gradient_from_next_layer * self.activation_func.derivative(self.last_z)  # on dérive la fonction d activation

        grad_weights = np.dot(dz, self.last_input.T)  # calcul du gradient des poids
        grad_bias = np.sum(dz, axis=1, keepdims=True)  # calcul du gradient du biais

        grad_input = np.dot(self.weights.T, dz)  # c est le gradient qu on renvoie à la couche d avant

        # maintenant on met à jour les poids et le biais
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_input  # on renvoie ce gradient pour la couche précédente

if __name__ == "__main__":
    import numpy as np

    # ici on fait un petit test avec une entrée toute simple
    X = np.array([[1.0], [2.0]])

    # on crée une couche avec 2 entrées et 1 neurone
    couche = CoucheNeurones(n_input=2, n_neurons=1, activation='sigmoid', learning_rate=0.1)

    # on fixe les poids et biais à la main pour voir ce qu il se passe
    couche.weights = np.array([[0.5, -1.0]])
    couche.bias = np.array([[0.0]])

    # on fait passer l entrée dans la couche
    output = couche.forward(X)
    print("Sortie après forward :", output)

    # on fait semblant de recevoir une erreur de 1 venant de la couche suivante
    gradient = np.array([[1.0]])
    grad_input = couche.backward(gradient)

    print("\n=== Test backward avec sigmoid ===")
    print("Poids mis à jour :", couche.weights)
    print("Biais mis à jour :", couche.bias)
    print("Gradient à propager :", grad_input)
