## Objectifs du TP

- Comprendre les limites du perceptron simple et la nécessité des réseaux multicouches  
- Implémenter l'algorithme de rétropropagation du gradient  
- Analyser l'impact de l'architecture du réseau sur les performances  
- Appliquer les réseaux multicouches à des problèmes non-linéairement séparables  
- Étudier les phénomènes de sur-apprentissage et de sous-apprentissage  

---

## Livrables Attendus

### Code

- Implémentation complète de la classe `PerceptronMultiCouches`  
- Implémentation de la classe `CoucheNeurones`  
- Scripts de test et de visualisation avancés  
- Comparaisons avec le perceptron simple  
- Compléter le rapport précédent  

### Rapport

#### Introduction
- Du perceptron simple au réseau multicouches  

#### Méthodes
- Architecture des réseaux multicouches  
- Algorithme de rétropropagation  
- Fonctions de coût et d'optimisation  

#### Résultats
- Résolution du problème XOR  
- Tests sur datasets synthétiques et réels  
- Analyse de l'impact de l'architecture  
- Courbes d'apprentissage et de validation  

#### Discussion
- Avantages et inconvénients des réseaux multicouches  
- Problèmes de sur-apprentissage  
- Stratégies de régularisation  

#### Conclusion
- Bilan et perspectives  

---

## Visualisations

- Surfaces de décision en 2D  
- Courbes d'apprentissage (loss et accuracy)  
- Visualisation des poids appris  
- Comparaisons de performances selon l'architecture  
- Analyse du sur-apprentissage  



------------RAPPORT-----------

## 1. Introduction
Le perceptron simple ne peut résoudre que des problèmes où les données sont séparables par une ligne droite. Par exemple, il fonctionne bien pour AND ou OR, mais pas pour XOR. Pour aller plus loin, on utilise un perceptron multicouche, qui peut résoudre des problèmes non linéaires. Pour l’entraîner, on utilise un algorithme appelé rétropropagation du gradient.

## 2. Méthodes
#### 2.1 Architecture du réseau
Un réseau est composé de couches : une couche d’entrée, une ou plusieurs couches cachées, et une couche de sortie. Chaque neurone reçoit des valeurs, les combine avec des poids et applique une fonction (comme ReLU ou sigmoïde) pour produire une sortie.

#### 2.2 Rétropropagation
C’est l’algorithme qui permet de corriger les poids. Il calcule l’erreur du réseau, puis fait « remonter » cette erreur en ajustant petit à petit les poids pour que le réseau s’améliore.

#### 2.3 Fonction de coût et optimisation
On mesure l’erreur du réseau avec une fonction de coût (par exemple l’erreur quadratique moyenne). L’optimisation consiste à modifier les poids pour réduire cette erreur, souvent avec des méthodes comme SGD ou Adam.

## 3. Résultats
#### 3.1 Problème XOR
On a testé le réseau sur le problème XOR. Contrairement au perceptron simple, le réseau multicouche arrive à bien le résoudre.

#### 3.2 Jeux de données
On a aussi testé le réseau sur des données plus complexes, synthétiques et parfois réelles. Les performances dépendent beaucoup de l’architecture choisie.

#### 3.3 Impact de l’architecture
Si le réseau est trop petit, il apprend mal (sous-apprentissage). S’il est trop gros, il peut apprendre « par cœur » sans bien généraliser (sur-apprentissage). Il faut trouver un bon équilibre.

#### 3.4 Courbes d’apprentissage
On a tracé les courbes de loss (erreur) et d’accuracy pendant l’entraînement. Ces courbes aident à voir si le réseau apprend bien ou s’il commence à sur-apprendre.

## 4. Discussion
#### Avantages
Le réseau multicouche peut apprendre des choses plus complexes que le perceptron simple.

Il est plus flexible et peut être adapté selon le problème.

#### Inconvénients
Il peut être long à entraîner.

Il faut choisir beaucoup de paramètres (nombre de neurones, taux d’apprentissage…).

#### Régularisation
Pour éviter le sur-apprentissage, on peut utiliser :

La régularisation (pénaliser les gros poids),

Le dropout (désactiver des neurones aléatoirement),

Le early stopping (arrêter l’apprentissage au bon moment).

## 5. Conclusion
Les réseaux multicouches sont essentiels pour résoudre des problèmes non linéaires. Leur fonctionnement repose sur la rétropropagation, et leur performance dépend beaucoup de leur configuration. Avec de bonnes pratiques, ils peuvent très bien généraliser sur de nouvelles données.
