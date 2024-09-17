# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:18:35 2024

@author: StanOWO
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Read CSV file
dataset = pp.dataset(file="../dataset/housing.csv")

# Drop the rows of data which contains null
dataset = dataset.dropna(axis=0, how='any')

# Decomposition the dataset into Independent & Dependent Variables
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(9)], y_columns=[9])

# Label Encoding
Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# Missing Data
#X = pp.missing_data(X, strategy="mean")
#Y = pp.missing_data(Y, strategy="mean")

# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)

# In[] SVM
# # from sklearn.svm import SVC
# # import time

# # classifier = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=int(time.time()))
# # classifier.fit(X_train, Y_train.values.ravel())
# # Y_pred = classifier.predict(X_test)

from HappyML.classification import SVM

classifier = SVM()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# from sklearn.model_selection import cross_val_score

# k_fold = 10
# accuracies = cross_val_score(estimator=classifier.classifier, X=X, y=Y.values.ravel(), scoring="accuracy", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean Accuracy: {}".format(k_fold, accuracies.mean()))

# recalls = cross_val_score(estimator=classifier.classifier, X=X, y=Y.values.ravel(), scoring="recall", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean Recall: {}".format(k_fold, recalls.mean()))

# precisions = cross_val_score(estimator=classifier.classifier, X=X, y=Y.values.ravel(), scoring="precision", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean Precision: {}".format(k_fold, precisions.mean()))

# f_scores = cross_val_score(estimator=classifier.classifier, X=X, y=Y.values.ravel(), scoring="f1", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean F1-Score: {}".format(k_fold, f_scores.mean()))
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K, verbose=False)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# import time

# classifier = DecisionTreeClassifier(criterion="entropy", random_state=int(time.time()))
# classifier.fit(X_train, Y_train)
# Y_pred = classifier.predict(X_test)
from HappyML.classification import DecisionTree

classifier = DecisionTree()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Decision Tree Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Random Forest
# from sklearn.ensemble import RandomForestClassifier
# import time

# classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=int(time.time()))
# classifier.fit(X_train, Y_train.values.ravel())
# Y_pred = classifier.predict(X_test)

# With HappyML's Class   
from HappyML.classification import RandomForest

classifier = RandomForest(n_estimators=10, criterion="entropy")
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Random Forest Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))