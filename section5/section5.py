# Multivariable linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Data encoding

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Dataset separation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print('X train')
print(X_train)

print ('y train')
print (y_train)

print('X test')
print(X_test)

print ('y test')
print (y_test)

# Train simple linear regression model on the training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results

y_hat = regressor.predict(X_test)
y_hat_train = regressor.predict(X_train)

# Visualizing the training set results

np.set_printoptions(precision=2)

print("Training data:")
print(np.concatenate((y_hat_train.reshape(len(y_hat_train), 1), y_train.reshape(len(y_train), 1)), 1))

# Visualizing the test set results

print("Test data:")
print(np.concatenate((y_hat.reshape(len(y_hat), 1), y_test.reshape(len(y_test), 1)), 1))

# Backward Elimination

import statsmodels.api as sm
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
X_opt = X[:, list(range(X.shape[1]))]
X_opt = X_opt.astype(np.float64)

SL = 0.05
while True:
  regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
  if np.amax(regressor_OLS.pvalues) > SL:
    X_opt = np.delete(X_opt,np.argmax(regressor_OLS.pvalues),axis=1)
    print("Deleted index ", np.argmax(regressor_OLS.pvalues), "p-value ", np.amax(regressor_OLS.pvalues))
  else:
    break

print(regressor_OLS.summary())
# Prediction for other values

# Making a single prediction (for example the profit of a startup with:
#   R&D Spend = 160000
#   Administration Spend = 130000
#   Marketing Spend = 300000
#   State = 'California'

x_input = np.array([[1, 0, 0, 160000, 130000, 300000]])
x_input[:, 3:] = sc.transform(x_input[:, 3:])
prediction = regressor.predict(x_input)[0]
print(f'Prediction for R&D Spend = 160000; Administration Spend = 130000; Marketing Spend = 300000; State = California: {prediction}')

# Getting the equation

coeffs = regressor.coef_
intercept = regressor.intercept_

ecuacion = ''
index = 0
for coef in coeffs:
  ecuacion = ecuacion + f'+ {coef} * x_{index}'
  index += 1

ecuacion = f'{ecuacion} + {intercept}'
print(f'Equation: {ecuacion}')
