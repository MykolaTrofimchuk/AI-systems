import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# завантаження Boston, else - California
try:
    housing_data = datasets.load_boston()
except ImportError:
    print("Boston dataset removed from sklearn. Using California housing instead.")
    housing_data = datasets.fetch_california_housing()

X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# AdaBoost Regressor
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400, random_state=7
)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nADABOOST REGRESSOR")
print(f"Mean squared error = {round(mse, 2)}")
print(f"Explained variance score = {round(evs, 2)}")

# Важливість ознак
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Нормалізація
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування
index_sorted = np.flipud(np.argsort(feature_importances))
pos = np.arange(index_sorted.shape[0]) + 0.5

# Графік
plt.figure(figsize=(10, 6))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, np.array(feature_names)[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Оцінка важливості ознак (AdaBoost)')
plt.show()