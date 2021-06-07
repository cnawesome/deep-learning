import numpy as np
#感知机and函数
#输入为0,1，参数的值决定输出的大小
#输出为0,1
def AND(x1, x2):
    w1,w2,theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0)) 
print(AND(1,1))

#与门函数
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#与非门函数
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    #*
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#或门函数
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    #权重与偏置与AND不一样
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#感知机不能表示异或门
#感知机局限在于它只能表示由直线分割的空间

#多层感知机实现异或函数
def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y