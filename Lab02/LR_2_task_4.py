import numpy as np
import warnings
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# --- КРОК 1: Завантаження та обробка даних---
input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

print("--- КРОК 1: Йде зчитування та обробка 50,000 рядків... ---")

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')

        if len(data) < 2:
            continue

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X_features = X_encoded[:, :-1].astype(int)
y_target = X_encoded[:, -1].astype(int)

print("Дані завантажено та оброблено.")
print(f"Форма X (ознаки): {X_features.shape}")
print(f"Форма y (ціль): {y_target.shape}")
print("-" * 30)

# --- КРОК 2: Порівняння алгоритмів ---
print("--- КРОК 2: Запуск порівняння моделей (10-fold CV)... ---")

# створення "конвеєрів" (Pipelines)
pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression(max_iter=1000))])))
pipelines.append(('LDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('NB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))

pipelines.append(('SVM', Pipeline([('Scaler', StandardScaler()), ('SVM', LinearSVC(dual=False, max_iter=2000))])))

results = []
names = []
scoring = 'accuracy'

for name, model_pipeline in pipelines:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model_pipeline, X_features, y_target, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

print("-" * 30)
print("Порівняння завершено.")

# --- КРОК 3: Візуалізація результатів ---
print("--- КРОК 3: Відображення графіку порівняння... ---")
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Порівняння точності (Accuracy) алгоритмів')
pyplot.ylabel('Точність (Accuracy)')
pyplot.show()