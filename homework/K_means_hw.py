# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 00:44:21 2024

@author: Stan Wang
"""
# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="../dataset/Family Income and Expenditure.csv")

# Decomposition
X = pp.decomposition(dataset, x_columns=[i for i in range(28)])

# One-Hot Encoding
X = pp.onehot_encoder(ary=X, columns=[1, 3, 25, 27], remove_trap=True)

# Feature Scaling (for PCA Feature Selection)
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# In[] K-Means Clustering
from HappyML.clustering import KMeansCluster

cluster = KMeansCluster(best_k=4, max_k=10)
Y_pred = cluster.fit(x_ary=X).predict(x_ary=X, y_column="Group number")

# In[] Save as csv file
file_name = "group.csv"

dataset = pp.combine(dataset, Y_pred)
dataset.to_csv(file_name, index=False)

print("Generate", file_name,"Successfully")