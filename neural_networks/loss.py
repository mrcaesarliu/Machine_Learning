# -*- coding: utf-8 -*-
import numpy as np


def loss_choice(loss_name):
    
    if loss_name == "MSE":
        loss_fn = MSE()
    elif loss_name == "cross_entropy":
        loss_fn = cross_entroy()
        pass
    return loss_fn

class MSE:
    
    # loss =  0.5 || y- y_pre ||^2
    # y' = y - y_pre
    
    def __init__(self):
        
        pass
    
    def loss(self, Y, y_pre):
        
        return 0.5 * np.linalg.norm(y_pre - Y) ** 2
    
    def grad(self, Y, y_pre):
        
        return y_pre-Y
    
    pass

class cross_entroy:
    
    # 在 cross_entropy + softmax 
    # 分类函数 + 损失函数 
    
    def __init__(self):
        #
        
        
        pass
    
    def loss(self, Y, y_pre):
        # Y [ 0 0 1 0 0]
        # y_pre 处理后是和为1 的概率分布
        # 先 进行 softmax 处理
        p1 = np.exp(y_pre)
        p2 = p1.sum(axis=1)
        p2 = p2.reshape([p2.shape[0],1])
        p = p1/p2
        # 防止 log(0) 出现
        eps = np.finfo(float).eps
        # 计算损失函数
        cross_entropy = -np.sum(Y * np.log(p + eps))
        
        return cross_entropy
    
    def grad(self, Y, y_pre):
        # y*ln(a) + softmax 一起求导后结果
        p1 = np.exp(y_pre)
        p2 = p1.sum(axis=1)
        p2 = p2.reshape([p2.shape[0],1])
        p = p1/p2
        
        return p-Y
    
    pass