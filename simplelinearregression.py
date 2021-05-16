Importing The Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Importing the Dataset

dataset = pd.read_csv('ex1data1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

Splitting the Dataset into Test set and Training set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/2, random_state = 0)

Training the Simple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

Predicting Test set result
y_pred = regressor.predict(X_test)


Visulaization of Training Set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Profit vs Population (Training set)')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()


Visualization of Test Set 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Profit vs Population (Training set)')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()
