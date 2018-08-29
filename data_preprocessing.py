# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:00:05 2018

@author: akash
"""
# Data Processing Templates
#Machine Learning
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Spliting dataset in Test and Training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""