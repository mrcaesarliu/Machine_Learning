# -*- coding: utf-8 -*-

import numpy as np


class OLS:
    # 最小二乘法回归模型
    # arg min sum{ (Y -y_i)^2  }
    def __init__(self, 
                 intercept = True,
                 Normal = False
                 ):
        self.intercept = intercept
        self.Normal = Normal
        pass
    def _normal(self, X):
        
        for col in range(X.shape[1]):
            mu = np.mean( X[:, col] )
            sigma = np.std( X[:, col] )
            Sample = ( X[:, col] - mu )/sigma
            X[:, col] = Sample
            pass
        
        return X
    
    def fit(self, X, Y):
        #  arg min{(Y -<w, X>)^2}  
        # L = min[(Y-X*W).T(Y-X*W)]
        # W* = (X.T*X).inv * X.T*Y
        # 最小二乘法求解过程
        if self.Normal:
            X = self._normal(X)
            pass
        if self.intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            pass
        pre_param = np.dot( np.linalg.inv( np.dot(X.T, X ) ) , X.T)
        self.Beta = np.dot( pre_param, Y )
        
        return self.Beta
    
    def predict(self, X):
        # Y = W*X
        if self.intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            pass
        Y = X @ self.Beta
        return Y
    
    pass



class Ridge_Rgegession:
    # arg min{ (Y-W*X)+lambda*||W|| }
    #加上一个惩罚系数
    def __init__(self,
                 intercept=True,
                 alpha,
                 Normal=False):
        self.intercept = intercept
        self.alpha = alpha
        self.Normal = Normal
        pass
    
    def _normal(self, X):
        
        for col in range(X.shape[1]):
            mu = np.mean( X[:, col] )
            sigma = np.std( X[:, col] )
            Sample = ( X[:, col] - mu )/sigma
            X[:, col] = Sample
            pass
        
        return X
    
    def fit(self, X, Y):
        # arg min{ (Y-W*X).T*(Y-W*X) + lambda*W.T*W  }
        # W* = (X.T*X + lambda*I).inv *(X.T*Y)
        # Ridge Regress回归的解
        if self.intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            pass
        if self.Normal:
            X = self._normal
            pass
        Regular = np.eye(X.shape[0]) * self.alpha
        pre_para = np.dot( np.linalg.inv(X.T @ X + Regular), X.T )
        self.beta = np.dot(pre_para, Y)
        
        return self.beta
    
    def predict(self, X):
        if self.intercept:
            X = np.c_[np.ones(X.shape[0]), X] 
            pass
        
        Y = X @ self.Beta
        
        return Y
    pass
