import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
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
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(float)

# --- Масштабування ознак ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = X_encoded[:, -1].astype(int)

# Поліноміальне ядро
classifier = SVC(kernel='poly', degree=3, C=1, gamma='scale', random_state=0)

# Cross-validation
f1_cv = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("--- Поліноміальне ядро (kernel='poly') ---")
print(f"F1 score (weighted, 3-fold CV): {round(100 * f1_cv.mean(), 2)}%")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {round(100 * acc, 2)}%")
print(f"Precision: {round(100 * prec, 2)}%")
print(f"Recall: {round(100 * rec, 2)}%")
print(f"F1 Score: {round(100 * f1, 2)}%")
