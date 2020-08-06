# Decition tree regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Train simple linear regression model

regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Making a single prediction

position = 6.5
prediction = regressor.predict([[position]])
print(f'Prediction for position {position}: {prediction}')

# Visualizing the data

x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
y_hat_grid = regressor.predict(x_grid)
plt.scatter(X, y, color='red')
plt.plot(x_grid, y_hat_grid, color='green')
plt.title('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()