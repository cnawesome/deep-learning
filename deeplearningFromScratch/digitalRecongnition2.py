import sys, os
#导入父目录
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from collections import OrderedDict

def softmax(a):
    c = np.max(a)
    #溢出对策
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#批处理交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+ 1e-7))/batch_size

'''
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test
'''

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        #损失
        self.loss = None 
        #softmax的输出
        self.y = 0
        #监督数据
        self.t = 0

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

    
class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size,
                 output_size, weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn( hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relus'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def Accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    #x:输入数据 t:监督数据
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grad = {}
        grad['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grad['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grad['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grad['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

    def gradient(self, x, t):
        #forward
        self.loss(x, t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        #设定
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch =100

for i in range(iters_num):
    #Generates a random sample from a given 1-D array
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #使用误差反向传播求梯度
    grad = network.gradient(x_batch,t_batch)

    #更新
#    for key in ('w1', 'b1', 'w2', 'b2'):
#        network.params[key] -= learning_rate * grad[key] 
    
    network.params['w1'] -= learning_rate * grad['w1']
    network.params['b1'] -= learning_rate * grad['b1']
    network.params['w2'] -= learning_rate * grad['w2']
    network.params['b2'] -= learning_rate * grad['b2']

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.Accuracy(x_train, t_train)
        test_acc = network.Accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
    
