import numpy as np
from PerceptronMultiCouches import PerceptronMultiCouches

# données d'entrée pour XOR
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])  # 4 exemples avec 2 entrées chacun

# sorties attendues pour XOR
y = np.array([[0, 1, 1, 0]])  # 1 sortie binaire par exemple

# on crée le réseau avec une couche cachée (2 neurones suffisent normalement)
reseau = PerceptronMultiCouches(architecture=[2, 2, 1], activation='tanh', learning_rate=0.1)

# on entraîne le réseau
reseau.entraine(X, y, epochs=10000)

# on teste le réseau après entraînement
sorties = reseau.calcule_sortie(X)

print("résultats prédits :")
print(sorties)

# on arrondit les sorties pour avoir 0 ou 1
print("sorties arrondies :")
print(np.round(sorties))
