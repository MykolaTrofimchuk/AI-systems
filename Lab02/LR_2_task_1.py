import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ігноруємо попередження про збіжність, оскільки дані не масштабовані
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        # ВИПРАВЛЕНО: .strip() надійніше, ніж [:-1]
        data = line.strip().split(', ')

        # ВИПРАВЛЕНО: 'X' містив і ознаки, і мітку.
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)  # Додаємо весь рядок (включаючи мітку)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)  # Додаємо весь рядок (включаючи мітку)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

# Розділяємо закодовані дані на ознаки (X) та мітку (y)
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Створення SVМ-класифікатора
# ВИПРАВЛЕНО: Додано dual=False та max_iter, щоб уникнути помилок збіжності
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, max_iter=5000))

# Навчання класифікатора
# ВИПРАВЛЕНО: замінено на 'y'
classifier.fit(X, y)

# Перехресна перевірка (розбиття даних)
# ВИПРАВЛЕНО: використовуємо 'train_test_split'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створюємо та навчаємо НОВИЙ класифікатор
classifier_test = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, max_iter=5000))
classifier_test.fit(X_train, y_train)
y_test_pred = classifier_test.predict(X_test)

# --- Обчислення F-міри ---
# ВИПРАВЛЕНО: 'train_test_split.cross_val_score' -> 'cross_val_score'
f1_cv = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("--- Метрики Cross-Validation (cv=3) ---")
print("F1 score (weighted, 3-fold CV): " + str(round(100 * f1_cv.mean(), 2)) + "%")

# --- Обчислення інших показників (для звіту) ---
# Використовуємо результати 'classifier_test' (навченого на 80/20)
print("\n--- Метрики для Train/Test Split (80/20) ---")
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, average='weighted')
rec = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Accuracy (Точність): {round(100 * acc, 2)}%")
print(f"Precision (Чіткість): {round(100 * prec, 2)}%")
print(f"Recall (Повнота): {round(100 * rec, 2)}%")
print(f"F1 Score (F-міра): {round(100 * f1, 2)}%")

# --- Передбачення для однієї точки ---
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0',
              '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        # ВИПРАВЛЕНО: .transform() очікує 1D-масив (навіть для одного елемента)
        input_data_encoded[i] = label_encoder[count].transform([item])[0]
        count += 1

input_data_encoded = np.array(input_data_encoded)

# ВИПРАВЛЕНО: Модель очікує 2D-масив (список зразків
input_data_encoded = input_data_encoded.reshape(1, -1)

# Використання класифікатора для кодованої точки даних
predicted_class = classifier.predict(input_data_encoded)

# ВИПРАВЛЕНО: Додано [0] для отримання самого значення зі списку
predicted_label = label_encoder[-1].inverse_transform(predicted_class)[0]

print("\n--- Прогноз для тестової точки ---")
print(f"Тестові дані: {input_data}")
print(f"Прогнозований клас доходу: {predicted_label}")