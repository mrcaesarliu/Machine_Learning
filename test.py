# -*- coding: utf-8 -*-
from initializ import weight_ch
from layer import Full_connected, Conv1D
from activate import act_choice
import numpy as np
from keras.datasets import mnist 
from model import MLP
from loss import loss_choice
from keras.datasets import cifar10


def init_test():
    # 测试权重初始化
    W = weight_ch('he_uniform', (100,20))
    return W
#W =  init_test()

def activate_test():
    # 测试激活函数
    act = act_choice('tanh')
    Data = np.array([[1,10,50,60,70,80,100],[1,10,50,60,70,80,100]])
    Z = act.fn(Data)
    G = act.grad(Z)
    return Z, G
Z, G = activate_test()

def layer_test():
    # 测试网络连接层函数
    # 构造训练数据
    (train_data,train_lable),(test_data,test_lable) = mnist.load_data()
    train_data = train_data[:1000,:,:]
    # reshape 教学
    train_data = train_data.reshape([train_data.shape[0], train_data.shape[1]*train_data.shape[2]])
    train_lable = train_lable[:1000]
    
    layer = Full_connected(n_out = 10)
    Z = layer.forward(train_data)
    
    dW, dB = layer.backward(Z)
    
    return dW, dB

def loss_test():
    
    (train_data,train_lable),(test_data,test_lable) = mnist.load_data()
    train_data = train_data[:1000,:,:]
    # reshape 教学
    train_data = train_data.reshape([train_data.shape[0], train_data.shape[1]*train_data.shape[2]])
    train_lable = train_lable[:1000]
    
    layer = Full_connected(n_out = 10)
    Z = layer.forward(train_data)
    Y = np.zeros([train_lable.shape[0],10])
    
    for i in range(train_lable.shape[0]):
        Y[i,train_lable[i]] = 1
        
    l = loss_choice('cross_entropy')
    los = l.loss(Y, Z)
    grad = l.grad(Y, Z)
    return grad

def model_test():

    # 构造训练数据
    (train_data,train_lable),(test_data,test_lable) = mnist.load_data()
    train_data = train_data[:1000,:,:]
    # reshape 教学
    train_data = train_data.reshape([train_data.shape[0], train_data.shape[1]*train_data.shape[2]])/255
    train_lable = train_lable[:1000]
    ANN = MLP(epoch=300, classif=True)
    ANN.add(Full_connected(n_out=300,acf_fn ='sigmod'))
    ANN.add(Full_connected(n_out=100,acf_fn ='sigmod'))
    ANN.add(Full_connected(n_out=10,acf_fn ='sigmod'))
    model_loss,acc = ANN.fit(train_data, train_lable)
        
    return model_loss,acc


def CNN_test():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # CNN = MLP(epoch=300, classif=True)
    X = x_train[:200]
    layer_T = Conv1D(n_out = 8,acf_fn ='ReLU' )
    layer_T.forward(X)
    pass
CNN_test()


