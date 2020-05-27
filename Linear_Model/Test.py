# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import make_regression
import matplotlib
from Linear_Regression import OLS

def data_make():
    
    X, y = make_regression(n_samples=100, n_features=2, n_targets=1)
    
    return X, y
X, y = data_make()
y = (y - np.mean(y)) / (np.std(y))
ols_m = OLS(Normal = False)
beta = ols_m.fit(X, y)
