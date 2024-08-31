# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:07:44 2024

@author: Stan Wang
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Read CSV file
dataset = pp.dataset(file="../dataset/Social_Network_Ads.csv")

# Decomposition the dataset into Independent & Dependent Variables
X, Y = pp.decomposition(dataset, x_columns=[0,1], y_columns=[2])

# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Logistic Regression
from HappyML.regression import LogisticRegressor
model = LogisticRegressor()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from HappyML.performance import ClassificationPerformance

pfm = ClassificationPerformance(Y_test, Y_pred)
print("Confusion Matrix:\n", pfm.confusion_matrix())
print(f"Accuracy: {pfm.accuracy():.2%}")
print(f"Recall: {pfm.recall():.2%}")
print(f"Precision: {pfm.precision():.2%}")

# In[] Visualization
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=model.regressor, 
                   title="訓練集樣本點 vs. 模型", font="DFKai-sb")
md.classify_result(x=X_test, y=Y_test, classifier=model.regressor, 
                   title="測試集樣本點 vs. 模型", font="DFKai-sb")