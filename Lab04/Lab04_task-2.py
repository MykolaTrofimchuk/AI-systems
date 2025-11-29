import numpy as np
import matplotlib.pyplot as plt

x = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0])
y = np.array([-5.0, -4.0, 0.7, 1.25, 3.0, 5.0])

# розрахунок коефіцієнтів методом найменших квадратів
coefficients = np.polyfit(x, y, 1)
beta_1 = coefficients[0]
beta_0 = coefficients[1]

# функція апроксимації
polynomial = np.poly1d(coefficients)
y_pred = polynomial(x)

print(f"Отримане рівняння регресії: y = {beta_0:.4f} + {beta_1:.4f}x")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Експериментальні точки')
plt.plot(x, y_pred, color='blue', label=f'Апроксимація: y={beta_0:.2f}+{beta_1:.2f}x') # Лінія регресії

plt.title('Лінійна регресія (Варіант 14)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()