import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Завантаження датасету Iris
iris = load_iris()
X = iris.data
y = iris.target

# Ініціалізація моделі K-Means.
# n_clusters=3 => існує 3 види ірисів.
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)

# Навчання моделі на вхідних даних X
kmeans.fit(X)

# Прогнозування кластерів
y_kmeans = kmeans.predict(X)

# Візуалізація результатів (за довжиною та шириною листка)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Дані')

# центри кластерів
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, label='Центроїди')
plt.title('K-Means clustering on Iris Dataset (sklearn)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()

def find_clusters(X, n_clusters, rseed=2):
    # генератор випадкових чисел із заданим зерном для відтворюваності
    rng = np.random.RandomState(rseed)

    # random початкових центрів кластерів із точок даних
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # знаходження найближчого центру для кожної точки
        # обчисленняя відстані та повертає індекс найближчого центру
        labels = pairwise_distances_argmin(X, centers)

        # обчислення середнього значення (mean) для всіх точок кожного кластера
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # перевірка на збіжність
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

centers, labels = find_clusters(X, 3, rseed=0)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('K-Means clustering (Manual Implementation)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()