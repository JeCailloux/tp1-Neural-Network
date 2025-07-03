import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from exercice_9 import PerceptronMultiClasse

def _afficher_cm(cm, classes, titre):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes, yticklabels=classes)
    plt.title(titre)
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.show()

def evaluer_perceptron_multiclasse(X, y,target_names=None,test_size=0.3,val_size=0.5,lr=0.1,max_epochs=100,random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state)
    print("Répartition des données :")
    print(f"  train       : {len(y_train)}")
    print(f"  validation  : {len(y_val)}")
    print(f"  test        : {len(y_test)}")

    perceptron_mc = PerceptronMultiClasse(learning_rate=lr)
    perceptron_mc.fit(X_train, y_train, max_epochs=max_epochs)

    y_pred_train = perceptron_mc.predict(X_train)
    y_pred_val   = perceptron_mc.predict(X_val)
    y_pred_test  = perceptron_mc.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val   = accuracy_score(y_val,   y_pred_val)
    acc_test  = accuracy_score(y_test,  y_pred_test)

    print("\nAccuracy :")
    print(f"  train      : {acc_train:.3f}")
    print(f"  validation : {acc_val:.3f}")
    print(f"  test       : {acc_test:.3f}")
    print("\nRapport test :")
    print(classification_report(y_test, y_pred_test,target_names=target_names))

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_val   = confusion_matrix(y_val,   y_pred_val)
    cm_test  = confusion_matrix(y_test,  y_pred_test)
    labels = target_names if target_names is not None else np.unique(y)
    _afficher_cm(cm_train, labels, "CM train")
    _afficher_cm(cm_val, labels, "CM val")
    _afficher_cm(cm_test, labels, "CM test")
    return perceptron_mc

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X = StandardScaler().fit_transform(X)
    evaluer_perceptron_multiclasse(X, y,target_names=iris.target_names,test_size=0.3,val_size=0.5,lr=0.1,max_epochs=100)
