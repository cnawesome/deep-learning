import numpy as np
import matplotlib.pylab as plt
#阶跃函数
def step_fuction1(x):
    if x > 0:
        return 1
    else:
        return 0

#适用于Numpy数组的阶跃函数
def step_fuction2(x):
    #判断数组中每一个值是否大于0
    y = x > 0
    #将得到的数组中的布尔型转化为int型
    return y.astye(np.int)

#阶跃函数
def step_fuction(x):
    return np.array(x>0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_fuction(x)
plt.plot(x, y)
#plt.ylim(-0.1, 1.1)
plt.show()
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x) )

x1 = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x1)
plt.plot(x1, y1)
plt.show() 