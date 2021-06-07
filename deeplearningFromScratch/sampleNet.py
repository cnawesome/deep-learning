import sys, os
#导入父目录
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np

def softmax(a):
    c = np.max(a)
    #溢出对策
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+ 1e-7))/batch_size

#对loss求梯度
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def f(W):
    return net.loss(x, t)

class sampleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x): 
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

##对一层的参数求梯度  
net = sampleNet()
#print(net.W)
x = np.array([2.0, 3.0])
p = net.predict(x)
#print(p)
t = np.array([0, 0, 1])
print(net.loss(x, t))

dw = numerical_gradient(f, net.W)
print(dw)

