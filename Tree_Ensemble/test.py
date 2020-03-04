# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
import numpy as np
from decision_tree import Decision_Tree
from random_forest import Random_Forets
from GBDT import GradientTreeBoost
from sklearn.datasets.samples_generator import make_blobs


def random_forest_test():
    X, Y = make_blobs(n_samples=100, centers=10, n_features=10, random_state=5)
    RF = Random_Forets(n_trees=10,
                       max_depth = 3,
                       classifier = True,
                       Loss="Gini"
                       
                       )
    RF.fit(X, Y)
    Y_P = RF.predict(X)
    
    return Y_P, Y
def decision_tree_test():
    data=load_boston()
    X=data.data
    Y=data.target
    X=X[:,[0,2,4,5,7,11]]
    dt = Decision_Tree(max_depth = 5)
    dt.fit(X, Y)
    Y_pre = dt.predict(X[-20:])
    
    return Y_pre

def GBDT_test():
    X, Y = make_blobs(n_samples=100, centers=10, n_features=10, random_state=5)
    gbdt = GradientTreeBoost(n_iter=20,
                             max_depth=3,
                             learn_rate=1,
                             type_boost="gradient_boost"
                             )
    gbdt.fit(X, Y)
    Y_P = gbdt.predict(X)
    
    
    return Y_P, Y

def decisionTree_class():
    X, Y = make_blobs(n_samples=100, centers=10, n_features=10, random_state=5)
    dt = Decision_Tree(max_depth = 5,
                       classifier = True,
                       Loss = "Gini")
    dt.fit(X, Y)
    Y_P = dt.predict(X)
    return Y_P, Y
Y_P, Y = GBDT_test()

