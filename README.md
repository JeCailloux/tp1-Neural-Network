lien du tp : https://sleek-think.ovh/enseignement/neural-network-perceptron/ 

## Objectifs du TP

- Comprendre le fonctionnement du perceptron simple  
- Implémenter l’algorithme du perceptron  
- Analyser les limites du perceptron sur des problèmes non-linéairement séparables  
- Appliquer le perceptron à des données réelles  

## Livrables attendus

### 1. Code

- Implémentation complète de la classe **PerceptronSimple**  
- Implémentation de la classe **PerceptronMultiClasse**  
- Scripts de test et de visualisation  

### 2. Rapport

- **Introduction** : Contexte et objectifs  
- **Méthodes** : Description des algorithmes implémentés  
- **Résultats** :  
  - Tests sur fonctions logiques  
  - Analyse de convergence  
  - Évaluation sur données réelles  
- **Discussion** :  
  - Limites du perceptron  
  - Cas d’usage appropriés  
- **Conclusion** : Synthèse des apprentissages  

### 3. Visualisations

- Graphiques de convergence  
- Visualisation des droites de séparation  
- Matrices de confusion  
- Comparaisons de performances  


Voir ci dessous le rapport demandé et voir dans REP_QUESTIONS.txt les réponses au questions des exercices


-----------------------------
---------- RAPPORT ----------
-----------------------------

### Introduction

Le fichier README.md rappelle les objectifs pédagogiques du projet : comprendre le fonctionnement du perceptron, l’implémenter et analyser ses limites, puis appliquer l’algorithme sur des données réelles.

### Méthodes

Le perceptron simple initialise un vecteur de poids aléatoire, puis itère sur les exemples pour ajuster les poids et le biais en fonction de l’erreur de prédiction. La prédiction utilise la fonction de Heaviside pour renvoyer une sortie binaire (0 ou 1).

Le perceptron multi-classe applique la stratégie « un-contre-tous » : un perceptron binaire est entraîné pour chaque classe, et la classe prédite est celle ayant le score maximal parmi tous les modèles.

Des scripts annexes permettent de générer des données linéairement séparables et d’étudier la convergence de l’apprentissage selon différents taux d’apprentissage.

L’évaluation avancée repose sur la séparation des données en ensembles d’entraînement, de validation et de test. Elle utilise un mécanisme d’early stopping basé sur partial_fit et la sauvegarde des meilleurs poids.

### Résultats

Le fichier PerceptronSimple.py inclut un jeu de tests sur les fonctions logiques AND, OR et XOR, affichant le score final et le nombre d’époques nécessaires à la convergence.

Le script de génération de données séparables affiche la performance finale du perceptron et visualise la frontière de décision apprise.

Le perceptron multi-classe dispose d’un test de base mesurant l’accuracy sur un jeu de données synthétiques à trois classes.

Le script d’évaluation génère des matrices de confusion et un rapport de classification détaillé (train/val/test), mais certaines méthodes nécessaires (partial_fit, get_weights, set_weights) ne sont pas encore implémentées.

### Discussion

Les résultats confirment les limites théoriques du perceptron : il échoue lorsque les données ne sont pas linéairement séparables (comme le montre l'exemple de XOR), et il est sensible au bruit ainsi qu’au déséquilibre des classes. Pour pallier ces limites, plusieurs pistes peuvent être envisagées :

La normalisation des données en entrée

L’ajout de couches cachées pour traiter des problèmes non linéaires (ex. : réseau de neurones multicouche) 

Une attention particulière portée à l’initialisation aléatoire des poids et au choix du taux d’apprentissage.

### Conclusion

Ce projet a permis de comprendre le perceptron, ses principes et ses limites. S’il fonctionne bien sur des données linéaires, il échoue dès que le problème devient non linéaire ou bruité. L’analyse a aussi souligné l’importance du taux d’apprentissage et de l’évaluation sur plusieurs ensembles. Pour aller plus loin, l’ajout de couches cachées ou l’usage d’outils plus avancés comme Scikit-learn pourrait être envisagé.
