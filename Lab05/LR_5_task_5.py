import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')
        data.append(items)
data = np.array(data)

# Кодування нечислових ознак
label_encoders = []
X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Регресор
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Оцінка
y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# Тестування
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0

for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        # використ. відповідний енкодер
        try:
             test_datapoint_encoded[i] = int(label_encoders[count].transform([test_datapoint[i]])[0])
             count += 1
        except:
             # Якщо час '10:20' не знайдено в навчанні, може виникнути помилка
             # Для простоти, якщо значення нове, ставимо 0 або обробляємо інакше
             print(f"Value {item} not seen in training.")
             test_datapoint_encoded[i] = 0
             count += 1

test_datapoint_encoded = np.array(test_datapoint_encoded).reshape(1, -1)

print("Input point:", test_datapoint)
print("Predicted traffic:", int(regressor.predict(test_datapoint_encoded)[0]))