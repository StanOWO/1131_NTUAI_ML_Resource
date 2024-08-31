# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:07:44 2024

@author: Stan Wang
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Read CSV file
dataset = pp.dataset(file="../dataset/car_data.csv")

# Decomposition the dataset into Independent & Dependent Variables
X, Y = pp.decomposition(dataset, x_columns=[2,3], y_columns=[4])

# Missing Data
X = pp.missing_data(X, strategy="mean")
Y = pp.missing_data(Y, strategy="mean")

# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler().fit(X_train)
# X_train = sc_X.transform(X_train)
# X_test = sc_X.transform(X_test)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Logistic Regression
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(solver="lbfgs", random_state=int(0))
# model.fit(X_train, Y_train.values.ravel())
# Y_pred = model.predict(X_test)
from HappyML.regression import LogisticRegressor
model = LogisticRegressor()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

# In[] Performance
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

# print("Confusion Matrix:\n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))
# print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2%}")
# print(f"Recall: {recall_score(Y_test, Y_pred):.2%}")
# print(f"Precision: {precision_score(Y_test, Y_pred):.2%}")

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