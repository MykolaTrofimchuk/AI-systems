import numpy as np
import matplotlib.pyplot as plt

x_points = np.array([0.1, 3.2, 0.3, 3.0, 0.4])
y_points = np.array([1.0, 0.6, 1.8, 0.7, 1.9])

# коефіцієнти інтерполяційного полінома (ступінь 4 для 5 точок)
degree = 4
coeffs_interp = np.polyfit(x_points, y_points, degree)
poly_interp = np.poly1d(coeffs_interp)

print("Коефіцієнти полінома (від найвищого ступеня):")
print(coeffs_interp)

# плавні дані для лінії графіка
x_line = np.linspace(min(x_points), max(x_points), 100)
y_line = poly_interp(x_line)

plt.figure(figsize=(10, 6))
plt.scatter(x_points, y_points, color='red', s=100, label='Вузли інтерполяції')
plt.plot(x_line, y_line, color='green', label='Інтерполяційний поліном')

# значення у проміжних точках 0.2 і 0.5
target_x = [0.2, 0.5]
target_y = poly_interp(target_x)

for tx, ty in zip(target_x, target_y):
    print(f"Значення функції в точці x={tx}: y={ty:.4f}")
    plt.plot(tx, ty, 'bo', label=f'Точка ({tx}, {ty:.2f})')

plt.title('Інтерполяція поліномом Лагранжа (4-го ступеня)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()