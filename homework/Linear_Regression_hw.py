# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:17:16 2024

@author: Stan Wang
"""
# In[] Preprocessing
import HappyML.preprocessor as pp

# Read CSV file
dataset = pp.dataset(file="../dataset/Salary.csv")

# Decomposition the dataset into Independent & Dependent Variables
X, Y = pp.decomposition(dataset, x_columns=[0], y_columns=[1])

# Missing Data
X = pp.missing_data(X, strategy="mean")
Y = pp.missing_data(Y, strategy="mean")

# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)

# In[] Linear regression
from HappyML.regression import SimpleRegressor

regressor = SimpleRegressor()
Y_pred = regressor.fit(X_train, Y_train).predict(X_test)

# In[] Performance
print("R-Squared Score:", regressor.r_score(X_test, Y_test))

# In[] Visualization
import HappyML.model_drawer as md

sample_data=(X_train, Y_train)
model_data=(X_train, regressor.predict(X_train))
md.sample_model(sample_data=sample_data, model_data=model_data,xlabel="year",ylabel="salary",
                title="data", font="DFKai-sb")