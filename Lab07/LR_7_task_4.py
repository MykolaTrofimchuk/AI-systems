import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import yfinance as yf
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

input_file = 'company_symbol_mapping.json'
try:
    with open(input_file, 'r') as f:
        company_symbols_map = json.loads(f.read())
except FileNotFoundError:
    print(f"Помилка: Файл {input_file} не знайдено.")
    sys.exit(1)

symbols, names = np.array(list(company_symbols_map.items())).T

# архівні дані котирувань
start_date = "2003-07-03"
end_date = "2007-05-04"

quotes = []
valid_names = []

print(f"Завантаження даних для {len(symbols)} компаній...")

# Завантажуємо дані пакетом
data_all = yf.download(list(symbols), start=start_date, end=end_date, progress=True, auto_adjust=True)

# Обробка даних
for symbol, name in zip(symbols, names):
    try:
        # наявність даних
        if symbol in data_all['Close'] and symbol in data_all['Open']:
            opens = data_all['Open'][symbol].values
            closes = data_all['Close'][symbol].values

            if np.isnan(opens).all() or np.isnan(closes).all():
                print(f"Пропуск {symbol}: відсутні дані.")
                continue

            quotes.append({'Open': opens, 'Close': closes})
            valid_names.append(name)
        else:
            print(f"Пропуск {symbol}: дані не знайдено.")

    except Exception as e:
        print(f"Помилка обробки {symbol}: {e}")

names = np.array(valid_names)

if len(quotes) < 2:
    print("Недостатньо валідних даних для кластеризації.")
    sys.exit(1)

# Обчислення різниці (Variation)
opening_quotes = np.array([q['Open'] for q in quotes]).astype(np.float64)
closing_quotes = np.array([q['Close'] for q in quotes]).astype(np.float64)

# Різниця
quotes_diff = closing_quotes - opening_quotes

# Нормалізація
X = quotes_diff.copy().T
X /= X.std(axis=0)

# модель (Lasso)
print("\nНавчання моделі GraphLassoCV...")
edge_model = covariance.GraphicalLassoCV(cv=5)

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

print("Кластеризація...")
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print("\n=== РЕЗУЛЬТАТИ КЛАСТЕРИЗАЦІЇ ===")
for i in range(num_labels + 1):
    cluster_names = names[labels == i]
    print(f"Cluster {i + 1}: {', '.join(cluster_names)}")