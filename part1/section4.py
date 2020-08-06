import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train simple linear regression model on the training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results

y_hat = regressor.predict(X_test)

# Visualizing the training set results

y_hat_train = regressor.predict(X_train)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_hat_train, color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results

y_hat_test = regressor.predict(X_test)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_hat, color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

# Prediction for other values, for example 12 years of experience

years = 12
prediction = regressor.predict([[years]])[0]
print(f'Prediction for {years} years of experience: {prediction}')

# Getting the equation

slope = regressor.coef_[0]
intercept = regressor.intercept_

print(f'Equation: {slope} * x + {intercept}')

# Evaluate model performance:

from sklearn.metrics import r2_score
score = r2_score(y_test, y_hat_test)
print(f'r2 score: {score}')
