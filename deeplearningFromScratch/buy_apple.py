import numpy as np

class MulLayer:

    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
    
    def backward(self, dout):
        dx = self.y
        dy = self.x

        return dx, dy
    
apple = 100
applenum = 2

mul = MulLayer()
result = mul.forward(100, 2)
dx, dy = mul.backward(result)
print(dx,dy)