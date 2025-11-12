import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# --- 1. Завантаження та підготовка даних ---
iris = load_iris()
X, y = iris.data, iris.target

# Розбиття даних
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)

# --- 2. Створення та навчання моделі ---
clf = RidgeClassifier(tol = 1e-2, solver = "sag")
clf.fit(Xtrain,ytrain)

# Прогноз
ypred = clf.predict(Xtest) # ВИПРАВЛЕНО: X_test -> Xtest

# --- 3. Розрахунок метрик якості ---
print("--- Показники якості класифікатора Ridge ---")
print('Accuracy:', np.round(metrics.accuracy_score(ytest,ypred),4))
print('Precision:', np.round(metrics.precision_score(ytest,ypred,average = 'weighted'),4))
print('Recall:', np.round(metrics.recall_score(ytest,ypred,average = 'weighted'),4))
print('F1 Score:', np.round(metrics.f1_score(ytest,ypred,average = 'weighted'),4))
print("-" * 20)
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest,ypred),4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest,ypred),4))
print("-" * 20)

# ВИПРАВЛЕНО: ytest та ypred поміняно місцями для коректного звіту
print('\t\tClassification Report:\n', metrics.classification_report(ytest,ypred))


# --- 4. Побудова Матриці Плутанини ---
mat = confusion_matrix(ytest, ypred)

# ВИПРАВЛЕНО: Прибрано транспонування та виправлено підписи осей
sns.heatmap(mat, square = True, annot = True, fmt = 'd', cbar = False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label (Прогнозована Мітка)')
plt.ylabel('True Label (Справжня Мітка)');

plt.savefig("Confusion.jpg")
print("\nМатрицю плутанини збережено у файл Confusion.jpg")

f = BytesIO()
plt.savefig(f, format = "svg")