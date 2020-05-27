# -*- coding: utf-8 -*-
import numpy as np


    
def weight_ch(init_type, data_shape):
    Weight_init = WeightInit(data_shape)
    if init_type=='he_normal':
        W = Weight_init.he_normal()
    elif init_type =='he_uniform':
        W = Weight_init.he_uniform()
    elif init_type =='glorot_normal':
        W = Weight_init.glorot_normal()
    elif init_type =='glorot_uniform':
        W = Weight_init.glorot_uniform()
    elif init_type =='trun_normal':
        W = Weight_init.trun_normal(0, 1, data_shape)
    
    return W

class WeightInit:
    
    def __init__(self, data_shape):
        self.data_shape = data_shape
        pass
    
    def parm_calu(self):
        # 计算 input ,output_size
        if len(self.data_shape)==2:
            in_size, out_size = self.data_shape
        elif len(self.data_shape) in [3,4]:
            in_ch, out_ch = self.data_shape[-2:]
            kernel_size = np.prod(self.data_shape[:-2])
            in_size, out_size = in_ch*kernel_size, out_ch* out_ch
        return in_size, out_size
    
    def he_normal(self):
        # he 高斯分布
        in_size, out_size = self.parm_calu()
        std = np.sqrt(2/in_size)
        Weights = self.trun_normal(0, std, self.data_shape)
        return Weights
    
    def he_uniform(self):
        # he 均匀分布
        in_size, out_size = self.parm_calu()
        limt = np.sqrt(6 / in_size)
        return np.random.uniform(-limt, limt, size=self.data_shape)
    
    def glorot_normal(self, P=1.0):
        # glorot 高斯分布
        in_size, out_size = self.parm_calu()
        std = P * np.sqrt(2 / (in_size + out_size))
        Weights = self.trun_normal(0, std, self.data_shape)
        return Weights
    
    def glorot_uniform(self, P=1.0):
        # glorot 均匀分布
        in_size, out_size = self.parm_calu()
        limt = P * np.sqrt(6 / (in_size + out_size))
        return np.random.uniform(-limt, limt, size=self.data_shape)
    
    def trun_normal(self, mean, std, data_shape):
        # 截尾高斯分布
        # 对于2倍标准差之外的数据重新生成 直到所有数据都在标准差范围内
        samples = np.random.normal(loc=mean, scale=std, size=data_shape)
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
        while any(reject.flatten()):
            resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
            samples[reject] = resamples
            reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
        return samples
    
    pass
