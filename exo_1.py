import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(z >= 0, 1, 0)
        elif self.name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.name == "tanh":
            return np.tanh(z)
        elif self.name == "relu":
            return np.where(z >= 0, z, 0)
        elif self.name == "leaky_relu":
            return np.where(z >= 0, z, self.alpha * z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
            return np.zeros_like(z)
        elif self.name == "sigmoid":
            sig = self.apply(z)
            return sig * (1 - sig)
        elif self.name == "tanh":
            tanh_z = self.apply(z)
            return 1 - tanh_z ** 2
        elif self.name == "relu":
            return np.where(z > 0, 1, 0)
        elif self.name == "leaky_relu":
            return np.where(z > 0, 1, self.alpha)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")


#Affichage de la fonction
z = np.linspace(-10, 10, 100)
activation = ActivationFunction("heaviside", alpha=0.01)

plt.figure(figsize=(10, 6))
plt.plot(z, activation.apply(z), label='Heaviside', linewidth=2)
plt.plot(z, activation.derivative(z), '--', label='Dérivée', color='orange')
plt.title('Fonction d\'activation de Heaviside et de sa dérivée')
plt.xlabel('z')
plt.ylabel('Valeur')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
