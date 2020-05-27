# -*- coding: utf-8 -*-
import numpy as np
from activate import act_choice
from initializ import weight_ch
from optimiz import optimizer_choice
from util import conv2D
from abc import ABC

class layer_base(ABC):
    def __init__(self, optimizer = 'SGD'):
        super().__init__()
        self.optimizer = optimizer_choice(optimizer)
        pass
    
    def update(self):
        A = sum(self.W)
        self.W = self.optimizer.update(self.W, self.grad_W, 'W')
        self.b = self.optimizer.update(self.b, self.grad_b, 'b')
        return A - sum(self.W)
    
    def flush_grad(self):
        self.grad_W = np.zeros_like(self.grad_W)
        self.grad_b = np.zeros_like(self.grad_b)
        pass
    
    pass

class Full_connected(layer_base):
    '''
    全连接神经网络层
    Full connected layer
    y = F(<W|X>+b)
    '''
    def __init__(self, n_out, acf_fn=None, init_type='he_normal', optimizer = 'SGD'):
        self.X = []
        self.n_out = n_out
        self.n_in = None
        self.acf_fn = acf_fn
        self.init_type = init_type
        self.is_init = False
        super().__init__(optimizer)
        pass
    
    def _param_init(self):
        # 初始化权重参数 激活函数
        data_shape = (self.n_in, self.n_out)
        W = weight_ch(self.init_type, data_shape)
        self.acf = act_choice(self.acf_fn)
        b = np.zeros([self.n_out])
        self.W = W
        self.b = b
        #self.grad_W = []
        #self.grad_b = []
        self.is_init = True
        pass
    
    def forward(self, X):
        # 训练网络是第一次FOWARD执行
        # 初始化网络 W权重 输入参数的个数 梯度矩阵的维度
        self.X = X
        if not self.is_init:
            self.n_in = X.shape[1]
            self._param_init()
            pass
        #正向传播本次网络
        Y, Z = self._fwd(X)
        
        return Y
    
    def _fwd(self, X):
        # 对于传入的数据 进行 权重 和激活函数
        W = self.W
        b = self.b
        Z = X@W + b
        Y = self.acf.fn(Z)
        
        return Y, Z
    
    def backward(self, dLdY):
        # 反向传播计算网络权重的梯度 
        X = self.X
        dX, dW, dB = self._bwd(dLdY, X)
        # 保存此次反向传播计算 W， B 的梯度
        self.grad_W = dW
        self.grad_b = dB
        # 返还本层传播误差
        # 本层传播误差将被当做上一层反向传播的输入
        return dX
    
    def _bwd(self, dLdY, X):
        
        W = self.W
        b = self.b
        # 正向传播是 激活函数的输入
        Z = X@W +b
        # 误差传播经过激活函数
        dZ = dLdY*self.acf.grad(Z)
        # 本层误差乘以W 准备传递到下一层去
        dX = dZ@W.T
        # 误差对W 的偏微分
        dW = X.T@dZ
        # 误差对bias的偏微分
        dB = dZ.sum(axis=0)
        
        return dX, dW, dB
    pass

class RNN(layer_base):
    
    
    
    pass

class Conv1D(layer_base):
    '''
    卷积层
    y = f( <pad(X)|W> + b) 
    '''
    def __init__(self, n_out, acf_fn=None,
                 init_type='he_normal', 
                 optimizer = 'SGD',
                 kernel_shape=(5,5),
                 stride =2,
                 pad = 0,
                 dilation = 0
                 ):
        self.X = []
        self.n_out = n_out
        self.n_in = None
        self.acf_fn = acf_fn
        self.init_type = init_type
        self.kernel_shape = kernel_shape
        self.is_init = False
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        super().__init__(optimizer)
        pass
    
    def _param_init(self):
        # 初始化权重参数 激活函数
        fr, fc = self.kernel_shape
        data_shape = (fr, fc, self.in_ch,self.n_out)
        W = weight_ch(self.init_type, data_shape)
        self.acf = act_choice(self.acf_fn)
        b = np.zeros((1, 1, 1, self.n_out))
        self.W = W
        self.b = b
        #self.grad_W = []
        #self.grad_b = []
        self.is_init = True        
        
        pass
    
    
    def forward(self, X):
        # 图片的维度
        if not self.is_init:
            self.in_ch = X.shape[3]
            self._param_init()
            pass        
        # X_shape = X.shape
        W = self.W
        b = self.b
        # 卷积函数
        s, p, d = self.stride, self.pad, self.dilation
        Z = conv2D(X, W, s, p, d)
        Y = self.acf_fn(Z)
        
        return Y
    
    def _fwd(self, X):
        
        pass
    
    def backward(self):
        
        pass
    
    def _bwd(self):
        
        pass
    
    pass

class BatchNorm(layer_base):
    
    
    pass

class RNN(layer_base):
    
    def __init__(self):
        
        pass
    
    
    pass


