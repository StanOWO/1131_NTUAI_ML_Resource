# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:17:16 2024

@author: Stan Wang
"""
# In[] Preprocessing
import pandas as pd

# Read CSV file
dataset = pd.read_csv("../dataset/Salary Data.csv")

# Decomposition the dataset into Independent & Dependent Variables
#X = pd.DataFrame(dataset.iloc[:, 0].values,columns=['age'])
X = pd.DataFrame(dataset.iloc[:, -2].values)
Y = pd.DataFrame(dataset.iloc[:, -1].values)

# Missing Data
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

X = pd.DataFrame(imputer.fit_transform(X))
Y = pd.DataFrame(imputer.fit_transform(Y))

# Split Training & Testing set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# In[] Linear regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
Y_pred=regressor.fit(X_train, Y_train).predict(X_test)

# In[] Performance
R_Score = regressor.score(X_test, Y_test)

print("R-Squared Score:",R_Score)

# In[] Visualization
import matplotlib.pyplot as plt

plt.plot(X_train, regressor.predict(X_train), linestyle='-')
plt.scatter(X_train, Y_train, color='red')
plt.title('data')
plt.xlabel('year')
plt.ylabel('salary')
plt.show()