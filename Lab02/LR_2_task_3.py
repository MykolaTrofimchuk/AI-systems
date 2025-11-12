# КРОК 1: Завантаження бібліотек
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder # <-- Важливе доповнення!
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
from sklearn.exceptions import ConvergenceWarning

# --- Завантаження та вивчення даних (Pandas) ---

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("--- КРОК 1: Вивчення даних (Pandas) ---")
# shape
print("Shape (форма):")
print(dataset.shape)
print("-" * 30)

# Зріз даних head (перші 20)
print("Head (перші 20 рядків):")
print(dataset.head(20))
print("-" * 30)

# Статистичне зведення (describe)
print("Describe (статистика):")
print(dataset.describe())
print("-" * 30)

# Розподіл за класами
print("Розподіл за класами:")
print(dataset.groupby('class').size())
print("-" * 30)


# --- КРОК 2: Візуалізація даних ---
print("--- КРОК 2: Візуалізація (чекайте на вікна з графіками) ---")

# Одновимірні графіки:
# Діаграма розмаху (Box and Whiskers)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.suptitle("Діаграма розмаху (Box-plot) для кожної ознаки")
pyplot.show()

# Гістограма
dataset.hist()
pyplot.suptitle("Гістограма розподілу для кожної ознаки")
pyplot.show()

# Багатовимірні графіки:
# Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.suptitle("Матриця діаграм розсіювання (Scatter-matrix)")
pyplot.show()

print("Візуалізацію завершено.")
print("-" * 30)

# --- КРОК 3: Створення навчального та тестового наборів ---

array = dataset.values
X = array[:,0:4]
y_text = array[:,4]

# ВИПРАВЛЕННЯ: Перетворюємо текстові мітки на числові (0, 1, 2)
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

# Розділення X і y на обучаючу та контрольну вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print("--- КРОК 3: Дані розділено на (80/20) ---")
print(f"Розмір X_train: {X_train.shape}")
print(f"Розмір X_validation: {X_validation.shape}")
print("-" * 30)

# --- КРОК 4: Класифікація (Побудова та оцінка моделей) ---

warnings.filterwarnings('ignore', category=FutureWarning)

print("--- КРОК 4: Порівняння алгоритмів (10-кратна CV) ---")

# алгоритми моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# оцінюємо модель на кожній ітерації
results = []
names = []
print("Результати (Середнє, Стандартне відхилення):")

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Графічне порівняння алгоритмів
pyplot.boxplot(results, tick_labels=names)

pyplot.title('Порівняння точності (Accuracy) алгоритмів')
pyplot.ylabel('Точність (Accuracy)')
pyplot.show()

print("-" * 30)

# --- КРОК 5: Оптимізація параметрів ---
print("--- КРОК 5 Оптимізація параметрів ---")
print("-" * 30)

# --- КРОК 6: Отримання прогнозу (на тестовому наборі) ---
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("--- КРОК 6: Прогноз на тестовій вибірці (X_validation) зроблено ---")
print("-" * 30)

# --- КРОК 7: Оцінка якості моделі ---
print("--- КРОК 7: Оцінка якості моделі (SVM) ---")
# оцінюємо прогноз
print("Accuracy (Точність):")
print(accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix (Матриця помилок):")
print(confusion_matrix(Y_validation, predictions))
print("\nClassification Report (Звіт класифікації):")
print(classification_report(Y_validation, predictions, target_names=encoder.classes_))
print("-" * 30)

# --- КРОК 8: Отримання прогнозу (Нова квітка) ---
print("--- КРОК 8: Прогноз для нової квітки ---")
X_new = np.array([[5, 2.9, 1, 0.2]])
print(f"Форма масиву X_new: {X_new.shape}")

# прогноз, використовуючи навчену модель SVM
prediction_numeric = model.predict(X_new)

# конвертуємо числову мітку (0, 1, 2) назад у текстову назву
prediction_name = encoder.inverse_transform(prediction_numeric)

print(f"Прогноз (числовий): {prediction_numeric}")
print(f"Спрогнозована мітка (назва сорту): {prediction_name[0]}")
print("-" * 30)