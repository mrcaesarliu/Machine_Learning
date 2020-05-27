# -*- coding: utf-8 -*-
#from layers import Full_connected
from loss import loss_choice
from optimiz import optimizer_choice
import numpy as np

class MLP:
    
    def __init__(self, epoch,loss_fn='cross_entropy', classif = False, optimizer='SGD'):
        self.epoch = epoch
        self.layers = []
        self.loss = loss_choice(loss_fn)
        self.classif = classif
        self.optimizer = optimizer
        pass
    
    def add(self, layer):
        # 为神经网络添加网络层数
        self.layers.append(layer)
        pass
    
    def smmary(self):
        # 打印神经网络每层网络的状态
        
        pass
    
    def foward(self, X):
        # 整个网络前向传播
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
            pass
        
        return Z
    
    def backward(self, dLdY):
        # 整个网络反向传播 
        # 并给网络层更新梯度
        for i in range(0, len(self.layers))[::-1]:
            dLdY = self.layers[i].backward(dLdY)
            pass
        
        pass
    
    def fit(self, X, Y_in):
        
        if self.classif:
            Y = self._process_Y(Y_in)
        model_loss = []
        acc=[]
        for i in range(self.epoch):
            Z = self.foward(X)
            loss_step = self.loss.loss(Y, Z)
            model_loss.append(loss_step)
            dL = self.loss.grad(Y, Z)
            self.backward(dL)
            self._update()
            Y_pre, acc_r = self.predict(X, Y_in)
            acc.append(acc_r)
            print(i)
            pass
        return model_loss,acc
    
    def _update(self):
        # 对每层神经网络中 
        for layer in self.layers:
            D = layer.update()
            pass
        pass
    
    def _process_Y(self, Y):
        #对于 lable 进行分类
        #
        class_value = np.unique(Y)
        self.class_value = class_value
        n_class = class_value.shape[0]
        Y_P = np.zeros([Y.shape[0], n_class])
        for i in range(Y.shape[0]):
            index = int( np.argwhere(Y[i] == class_value) )
            Y_P[i, index] =1
            pass
        
        return Y_P
    
    def predict(self, X, Y_in):
        Y = self.foward(X)
        Y_pre = np.zeros_like(Y_in)
        for i in range(Y.shape[0]):
            sample_p = Y[i,:]
            index = np.argmax(sample_p)
            Y_pre[i] = self.class_value[index]
            pass
        acc = 100*sum(Y_pre == Y_in)/Y.shape[0]
        return Y_pre, acc
    pass

'''

class softmax:
    def __init__(self, loss_fn='cross_entropy'):
        self.layers=[]
        self.loss_fn = loss_choice(loss_fn)
        pass
    
    def add(self, layer):
        self.layers.append(layer)
        
        pass
    
    def forward(self, X):
        for i in range(len(self.layers)):
            X = self.layers[i].forward(X) 
            pass
        
        return X
    
    def backward(self, dLdY):
        
        for i in range(len(self.layers)):
            self.layers[i].backward()
            pass
        pass
    
    def update(self):
        
        pass
    
    def fit(self, X, Y, epoch):
        
        loss = []
        for i in range(epoch):
            X = self.forward()
            loss.append(self.loss_fn.loss(X, Y))
            dL = self.loss_fn.grad(X, Y)
            self.backward(dL)
            self.update()
            pass
        loss = np.array(loss)
        
        return loss
    pass
'''