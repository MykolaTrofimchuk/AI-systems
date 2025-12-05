import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

le_outlook = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['Outlook_n'] = le_outlook.fit_transform(df['Outlook'])
df['Humidity_n'] = le_humidity.fit_transform(df['Humidity'])
df['Wind_n'] = le_wind.fit_transform(df['Wind'])
df['Play_n'] = le_play.fit_transform(df['Play'])

X = df[['Outlook_n', 'Humidity_n', 'Wind_n']]
y = df['Play_n']

# --- НАВЧАННЯ ---
model = GaussianNB()
model.fit(X, y)

# Outlook=Sunny, Humidity=Normal, Wind=Strong
variant_data = ['Sunny', 'Normal', 'Strong']

outlook_val = le_outlook.transform([variant_data[0]])[0]
humidity_val = le_humidity.transform([variant_data[1]])[0]
wind_val = le_wind.transform([variant_data[2]])[0]

prediction = model.predict([[outlook_val, humidity_val, wind_val]])
result = le_play.inverse_transform(prediction)[0]

proba = model.predict_proba([[outlook_val, humidity_val, wind_val]])
yes_prob = proba[0][1] * 100
no_prob = proba[0][0] * 100

print(f"Вхідні умови: {variant_data}")
print("-" * 30)
print(f"Прогноз: Чи відбудеться гра? -> {result}")
print(f"Ймовірність 'Yes': {yes_prob:.2f}%")
print(f"Ймовірність 'No':  {no_prob:.2f}%")