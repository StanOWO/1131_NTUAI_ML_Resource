# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 01:26:07 2024

@author: Stan Wang
"""
# In[] Import & Load Data Set
import HappyML.preprocessor as pp

dataset = pp.dataset(file="../dataset/Salary_Data.csv")

# In[] Decomposition the dataset into Independent & Dependent Variables
X, Y = pp.decomposition(dataset, x_columns=[-2], y_columns=[-1])

# In[] Missing Data
X = pp.missing_data(X, strategy="mean")
Y = pp.missing_data(Y, strategy="mean")

# In[] Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)

# In[] Linear regression
from HappyML.regression import SimpleRegressor

regressor = SimpleRegressor()
Y_pred = regressor.fit(X_train, Y_train).predict(X_test)
print("R-Squared Score:", regressor.r_score(X_test, Y_test))

# In[] Draw
import HappyML.model_drawer as md

sample_data=(X_train, Y_train)
model_data=(X_train, regressor.predict(X_train))
md.sample_model(sample_data=sample_data, model_data=model_data,xlabel="year",ylabel="salary",
                title="data", font="DFKai-sb")