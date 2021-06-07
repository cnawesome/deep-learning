import sys, os
#导入父目录
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x) )


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
        fxh1 = f(x) 
        # f(x+h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val 
        # 还原值
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


class TwolayerNet():

    def __init__(self, input_size, hidden_size, output_size,
                    weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = softmax(a2)

        return x

    # x:输入数据 t：监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y =self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    #x:输入数据 ，t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['bw'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


class SGD:

    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 


net = TwolayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['w1'].shape)
x = np.random.rand(100, 784)
#y = net.predict(x)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
numerical_gradient(x_train, t_train )
