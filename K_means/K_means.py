# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 00:44:21 2024

@author: Stan Wang
"""
# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="../dataset/Hobby_Data.csv")

# Decomposition
X = dataset

# One-Hot Encoding
# import pandas as pd
# import numpy as np
# columns_to_encode = [i for i in range(X.shape[1]) if i not in [5, 6, 12]]
# ary_dummies = pd.get_dummies(X.iloc[:, columns_to_encode])
# column_else=[5,6,12]
# X = np.concatenate((ary_dummies, X.iloc[:, column_else]), axis=1).astype("float64")
X = pp.onehot_encoder(ary=X, columns=[i for i in range(14) if i != 5 or i!=6 or i!=12], remove_trap=True)

# Feature Scaling (for PCA Feature Selection)
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# In[] K-Means Clustering
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4, init="k-means++", random_state=0)
# Y_pred = kmeans.fit_predict(X)

from HappyML.clustering import KMeansCluster

cluster = KMeansCluster(best_k=4,max_k=10)
Y_pred = cluster.fit(x_ary=X).predict(x_ary=X, y_column="Group number")

# In[] Save as csv file
file_name = "group.csv"

dataset = pp.combine(dataset, Y_pred)
dataset.to_csv(file_name, index=False)

print("Generate", file_name,"Successfully")