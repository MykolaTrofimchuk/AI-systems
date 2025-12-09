import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Параметри навчання
n_samples = 1000
batch_size = 100
num_steps = 20000
display_step = 1000
learning_rate = 0.0001  # Швидкість навчання

print("Генерація даних...")
X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1)) # y = 2x + 1 + шум

# входи в граф обчислень
X = tf.placeholder(tf.float32, shape=(batch_size, 1), name="Input_X")
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="Input_y")

# параметри, які модель буде вчити
with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1)), name='slope')  # Нахил (вага)
    b = tf.Variable(tf.zeros((1,)), name='bias')  # Зсув

# Модель
y_pred = tf.matmul(X, k) + b

# Сума квадратів помилок
loss = tf.reduce_sum((y - y_pred) ** 2)

# Алгоритм градієнтного спуску
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Запуск сесії
print("Початок навчання...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        # Запуск кроку оптимізації
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b],
                                             feed_dict={X: X_batch, y: y_batch})

        if (i + 1) % display_step == 0:
            print('Епоха %d: Loss=%.4f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_val, b_val))

    print("\nНавчання завершено!")
    print(f"Фінальні значення: k = {k_val[0][0]:.4f} (очікувалось ~2), b = {b_val[0]:.4f} (очікувалось ~1)")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_data, y_data, s=10, label='Дані (з шумом)')

    y_line = k_val[0][0] * X_data + b_val[0]
    plt.plot(X_data, y_line, color='red', linewidth=2, label=f'Лінійна регресія: y={k_val[0][0]:.2f}x + {b_val[0]:.2f}')

    plt.title('Лінійна регресія з TensorFlow')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()