# Polynomial regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Train simple linear regression model

lr = LinearRegression()
lr.fit(X, y)

# Polynomial features

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)
pr = LinearRegression()
pr.fit(X_poly, y)

# Visualizing the data

y_hat = lr.predict(X)
y_hat_poly = pr.predict(pf.fit_transform(X))
plt.scatter(X, y, color='red')
plt.plot(X, y_hat, color='blue')
plt.plot(X, y_hat_poly, color='green')
plt.title('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Prediction for other values

# Making a single prediction

position = 6.5
prediction = pr.predict(pf.fit_transform([[position]]))[0]
print(f'Prediction for position {position}: {prediction}')

"""
# Getting the equation

coeffs = pr.coef_
intercept = pr.intercept_

ecuacion = ''
index = 0
for coef in coeffs:
  ecuacion = ecuacion + f'+ {coef} * x ** {index}'
  index += 1

ecuacion = f'{ecuacion} + {intercept}'
print(f'Equation: {ecuacion}')
"""
