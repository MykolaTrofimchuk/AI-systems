import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
try:
    print("Завантаження та обробка даних...")
    df = pd.read_csv(url)
    df = df.dropna()
    df = df.head(10000)

    le_type = LabelEncoder()
    df['train_type_n'] = le_type.fit_transform(df['train_type'])

    le_class = LabelEncoder()
    df['train_class_n'] = le_class.fit_transform(df['train_class'])

    # ціна та тип потяга для прогнозу класу
    X = df[['price', 'train_type_n']]
    y = df['train_class_n']

    # навчальна та тестова вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nТочність моделі (Accuracy): {acc:.2f}")
    print("\nЗвіт класифікації:")
    print(classification_report(y_test, y_pred, zero_division=0))

except Exception as e:
    print(f"Помилка: {e}")