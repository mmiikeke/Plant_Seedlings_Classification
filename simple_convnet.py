# coding: utf-8
import sys, os
sys.path.append(os.pardir)  #設置文件父目錄
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import math


class SimpleConvNet:
    """捲積神經網路

    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 輸入尺寸
    hidden_size_list : 隱藏層中神經元數量的列表（e.g. [100, 100, 100]）
    output_size : 輸出尺寸
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定Weight的標準差（e.g. 0.01）
        如果指定"relu"或"he"，則設置"He的初始值"
        如果指定"sigmoid"或"xavier"，則設置"Xavier的初始值"
    """
    def __init__(self, input_dim=(3,128,128), 
                 conv_param={'filter_num':[32,64], 'filter_size':[3,3], 'pad':[1,1], 'stride':[1,1]},
                 hidden_size=100, output_size=12, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size1 = (input_size - filter_size[0] + 2*filter_pad[0]) / filter_stride[0] + 1
        conv_output_size2 = ((conv_output_size1/2) - filter_size[1] + 2*filter_pad[1]) / filter_stride[1] + 1
        pool_output_size2 = int(filter_num[1] * (conv_output_size2/2) * (conv_output_size2/2))

        #初始化權重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num[0], input_dim[0], filter_size[0], filter_size[0])
        self.params['b1'] = np.zeros(filter_num[0])
		
        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter_num[1], filter_num[0], filter_size[1], filter_size[1])
        self.params['b2'] = np.zeros(filter_num[1])

        self.params['W3'] = weight_init_std * \
                            np.random.randn(pool_output_size2, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)

        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        #生成層數
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'][0], conv_param['pad'][0])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'][1], conv_param['pad'][1])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """查找損失函數
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """求梯度（數值導數）
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（反向傳播方法）
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]