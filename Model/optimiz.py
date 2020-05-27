# -*- coding: utf-8 -*-

import numpy as np


def optimizer_choice(optimizer_name):
    
    if optimizer_name=='SGD':
        optimizer = SGD()
    elif optimizer_name == 'AdaGrad':
        optimizer = AdaGrad()
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop()
    elif optimizer_name == 'Adam':
        optimizer = Adam()
        pass

    return optimizer


class SGD:
    
    # W = W' - n*G
    #momentum formular:
    # W = W' - v  ; v = beta*v' + n*G 
    # 带动量的SGD算法
    
    def __init__(self, lr=0.01, momentum=0):
        self.cache = {}
        self.lr = lr
        self.momentum = momentum
        self.mt_init = False
        pass
    
    def update(self, param, grad, param_name):
        
        C = self.cache
        if param_name not in C:
            C[param_name] = np.zeros_like(param)
            pass
        
        update = self.momentum*C[param_name] + self.lr*grad
        self.cache[param_name] = update
        
        return param - update
    
    pass


class AdaGrad:
    
    def __init__(self):
        
        pass
    
    def update(self, param, grad, beta):
        
        pass
    
    pass

class RMSprop:
    
    def __init__(self):
        
        pass
    
    def update(self):
        
        pass
    
    pass

class Adam:
    
    def __init__(self):
        
        pass
    
    def update(self):
        
        pass
    
    pass