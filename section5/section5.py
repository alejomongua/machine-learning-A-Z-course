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

# Prediction for other values, for example 12 years of experience
"""
years = 12
prediction = regressor.predict([[years]])[0]
print(f'Prediction for {years} years of experience: {prediction}')

# Getting the equation

slope = regressor.coef_[0]
intercept = regressor.intercept_

print(f'Equation: {slope} * x + {intercept}')
"""