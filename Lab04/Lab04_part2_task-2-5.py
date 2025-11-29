import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1) # Перетворення у двовимірний масив
noise = np.random.uniform(-0.5, 0.5, (m, 1)) # Випадковий шум
y = 3 + np.sin(X) + noise

# лінійна регресія (Проста пряма)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# ступінь 3, оскільки синусоїда на інтервалі [-3, 3] (розкладання Тейлора: x - x^3/6)
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(f"X[0] (оригінальний): {X[0]}")
print(f"X_poly[0] (розширений: x, x^2, x^3): {X_poly[0]}")

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

print("\n--- Результати ---")
print("Лінійна регресія R2:", round(r2_score(y, y_lin_pred), 3))
print("Поліноміальна регресія R2:", round(r2_score(y, y_poly_pred), 3))

print("\n--- Коефіцієнти Поліноміальної моделі ---")
print(f"Intercept (Вільний член, b): {poly_reg.intercept_[0]:.4f}")
print(f"Coefficients (Ваги w1, w2, w3): {poly_reg.coef_[0]}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Випадкові дані (Variant 9)')
plt.plot(X, y_lin_pred, color='red', linewidth=2, label='Лінійна регресія (degree=1)')
plt.plot(X, y_poly_pred, color='green', linewidth=3, label='Поліноміальна регресія (degree=3)')

plt.title('Порівняння Лінійної та Поліноміальної регресії: y = 3 + sin(x) + noise')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()