import sys, os
#导入父目录
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
# load_mnist 三个参
# 1.normalize 将输入图像正规化为0.1~1.0 
# 2.flatten 展开输入图像，变为一维数组
# 3.one_hot_lable 将标签表示为one_hot，仅正确的位置置为1，如 2：【0，0，1，0，0，0，0，0，0，0】
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

#输出各个数据的形状
print(x_train.shape)