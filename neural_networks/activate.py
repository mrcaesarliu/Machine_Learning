# -*- coding: utf-8 -*-
import numpy as np


    
def act_choice(act_name):
    
    if act_name=='sigmod':
        acf = sigmod()
    elif act_name=='tanh':
        acf = tanh()
    elif act_name=='ReLU':
        acf = ReLU()
    elif act_name=='Affine':
        acf = Affine()
    elif act_name=='ELU':
        acf = ELU()
    elif not(act_name):
        acf = N()
    return acf


class sigmod:
    def __init__(self):
        
        pass
    def fn(self, Z):
        
        return 1 / (1 + np.exp(-Z))
    
    def grad(self, X):
        
        fn_x = self.fn(X)
        return fn_x * (1 - fn_x)
    
    pass

class tanh:
    
    def __init__(self):
        
        pass
    def fn(self, Z):
        
        return np.tanh(Z)
    
    def grad(self, X):
        
        tanh_x = np.tanh(X)
        return -2 * tanh_x * (1 - tanh_x ** 2)
    pass

class ReLU:
    
    def __init__(self):
        
        pass
    
    def fn(self, Z):
        
        return np.clip(Z, 0, np.inf)
    
    def grad(self, x):
        
        return (x > 0).astype(int)
    pass

class Affine:
    
    pass

class ELU:
    
    pass

class N:
    def __init__(self):
        
        pass
    def fn(self, Z):
        
        return Z
    
    def grad(self, X):
        
        return np.ones_like(X)
    pass


