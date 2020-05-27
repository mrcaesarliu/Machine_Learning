# -*- coding: utf-8 -*

import numpy as np


class Convolution:
    # 卷积网络所使用的函数集合
    def __init__(self):
        
        pass
    
    def Conv2D(self, X, W, pad=0, stride=1):
        
        fr, fc, out_ch = W.shape
        
        n_sam, in_row, in_col = X.shape
        
        
        
        
        pass
    
    def Conv1D(self, X, W, stride):
        
        # 卷积核行 ， 列 ， 输入数据通道， 输出数据通道
        fr, fc, out_ch = W.shape
        W_col = W.transpose([2,0,1])
        W_col = W_col.reshape([out_ch, -1]) 
        #输入图片的数据结构
        n_sam, in_row, in_col, in_ch = X.shape
        
        # 
        x_step = ( (fc - in_col)/stride )+1
        y_step = ( (fr - in_row)/stride )+1
        
        #
        Y = np.zeros([n_sam, row, col, out_ch])
        
        for x_step in range(r):
            
            for y_step in range(col):
                
                Y[:,y_step, x_step] = X[:, y_step:y_step+fr, x_step:x_step+fc].reshape([n_sam, -1])@W_col 
                
                pass
            pass
            
                        

    
    def _img2col(self, X, W_shape):
        # 将输入的图片数据变成二维数字矩阵
        
        
        
        pass
    
    def pading(self):
        
        pass
    
    def Pooling(self):
        
        pass
    
    pass

###################################################
            # 卷积层
#####################################################
    
def conv2D(X, W, stride, pad, dilation=0):

    s, d = stride, dilation
    _, p = pad2D(X, pad, W.shape[:2], s, dilation=dilation)

    pr1, pr2, pc1, pc2 = p
    fr, fc, in_ch, out_ch = W.shape
    n_ex, in_rows, in_cols, in_ch = X.shape

    
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

   
    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    
    X_col, _ = im2col(X, W.shape, p, s, d)
    W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)

    Z = (W_col @ X_col).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

    return Z


##############################################################
    #卷积向量化
##############################################################

def im2col(X, W_shape, pad, stride, dilation=0):

    fr, fc, n_in, n_out = W_shape
    s, p, d = stride, pad, dilation
    n_ex, in_rows, in_cols, n_in = X.shape

    
    X_pad, p = pad2D(X, p, W_shape[:2], stride=s, dilation=d)
    pr1, pr2, pc1, pc2 = p

   
    X_pad = X_pad.transpose(0, 3, 1, 2)

    
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, p, s, d)

    X_col = X_pad[:, k, i, j]
    X_col = X_col.transpose(1, 2, 0).reshape(fr * fc * n_in, -1)
    return X_col, p

def _im2col_indices(X_shape, fr, fc, p, s, d=0):

    pr1, pr2, pc1, pc2 = p
    n_ex, n_in, in_rows, in_cols = X_shape

   
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution: "
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )


    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in) * (d + 1)
    i1 = s * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * n_in) * (d + 1)
    j1 = s * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (fr * fc * n_in, out_height * out_width)
    # j.shape = (fr * fc * n_in, out_height * out_width)
    # k.shape = (fr * fc * n_in, 1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j

def col2im(X_col, X_shape, W_shape, pad, stride, dilation=0):

    if not (isinstance(pad, tuple) and len(pad) == 4):
        raise TypeError("pad must be a 4-tuple, but got: {}".format(pad))

    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = pad
    fr, fc, n_in, n_out = W_shape
    n_ex, in_rows, in_cols, n_in = X_shape

    X_pad = np.zeros((n_ex, n_in, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, pad, s, d)

    X_col_reshaped = X_col.reshape(n_in * fr * fc, -1, n_ex)
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return X_pad[:, :, pr1:pr2, pc1:pc2]

################################################################################
    #填充工具函数
###############################################################################

def pad2D(X, pad, kernel_shape=None, stride=None, dilation=0):

    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # compute the correct padding dims for a 'same' convolution
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_2D(
            X.shape, X.shape[1:3], kernel_shape, stride, dilation=dilation
        )
        X_pad, p = pad2D(X, p)
    return X_pad, p

def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride, dilation=0):

    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    pr = int((stride * (out_rows - 1) + _fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + _fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - _fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - _fc) / stride)


    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError

    if any(np.array([pr1, pr2, pc1, pc2]) < 0):
        raise ValueError(
            "Padding cannot be less than 0. Got: {}".format((pr1, pr2, pc1, pc2))
        )
    return (pr1, pr2, pc1, pc2)

