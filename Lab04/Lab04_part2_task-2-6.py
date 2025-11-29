import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
noise = np.random.uniform(-0.5, 0.5, (m, 1))
y = 3 + np.sin(X) + noise


# побудовa кривих навчання
def plot_learning_curves(model, X, y, ax, title):
    """
    криві навчання: RMSE на тренувальному та валідаційному сетах
    залежно від розміру навчальної вибірки.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []

    # тренуємо модель
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчання (train)")
    ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірка (val)")
    ax.set_title(title)
    ax.set_xlabel("Розмір навчального набору")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 3)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- Графік 1: Лінійна регресія ---
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, ax1, "Лінійна регресія (Underfitting)")

# --- Графік 2: Поліноміальна регресія (ступінь 2) ---
poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg_2, X, y, ax2, "Поліном (ступінь 2)")

# --- Графік 3: Поліноміальна регресія (ступінь 10) ---
poly_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg_10, X, y, ax3, "Поліном (ступінь 10) - Overfitting")

plt.tight_layout()
plt.show()