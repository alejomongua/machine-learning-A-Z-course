# SVR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y_ = dataset.iloc[:, -1].values

# Feature scaling

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y_.reshape(len(y_), 1))

print("Scaled X")
print(X)

print("Scaled y")
print(y)

# Train simple linear regression model

regressor = SVR(kernel='rbf')
regressor.fit(X, y.reshape(len(y), ))

# Making a single prediction

position = 6.5
prediction = regressor.predict(sc_X.transform([[position]]))
print(f'Prediction for position {position}: {sc_y.inverse_transform(prediction)}')

# Visualizing the data

X_original = sc_X.inverse_transform(X)
y_original = sc_y.inverse_transform(y)
y_hat = sc_y.inverse_transform(regressor.predict(X))
plt.scatter(X_original, y_original, color='red')
plt.plot(X_original, y_hat, color='green')
plt.title('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Smoother plot

x_grid = np.arange(min(X_original), max(X_original), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
y_hat_grid = sc_y.inverse_transform(regressor.predict(sc_X.transform(x_grid)))
plt.scatter(X_original, y_original, color='red')
plt.plot(x_grid, y_hat_grid, color='green')
plt.title('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Evaluate model performance:

from sklearn.metrics import r2_score
score = r2_score(y_, y_hat)
print(f'r2 score: {score}')
