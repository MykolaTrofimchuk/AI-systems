import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# test_size = 0.5 (50% даних для тестування), random_state = 0 (для відтворюваності)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Regression Coefficients (Коефіцієнти регресії):")
print(regr.coef_)
print("\nIntercept (Вільний член):", regr.intercept_)

print("\nMetrics:")
print("R2 score =", round(r2_score(ytest, ypred), 2))
print("Mean Absolute Error (MAE) =", round(mean_absolute_error(ytest, ypred), 2))
print("Mean Squared Error (MSE) =", round(mean_squared_error(ytest, ypred), 2))

fig, ax = plt.subplots(figsize=(8, 6))

# розкид: Реальні значення (X) проти Передбачених (Y)
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.7, label='Дані')

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеал')

ax.set_xlabel('Виміряно (Observed)')
ax.set_ylabel('Передбачено (Predicted)')
ax.set_title('Лінійна регресія: Прогресування діабету')
ax.legend()
plt.grid(True)
plt.show()