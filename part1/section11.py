# Random Forest regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Train simple linear regression model

regressor = RFR(n_estimators=10, random_state=0)
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

# Evaluate model performance:

from sklearn.metrics import r2_score
y_hat = regressor.predict(X)
score = r2_score(y, y_hat)
print(f'r2 score: {score}')
