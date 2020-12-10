import numpy as np
import math


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

        

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        self.x = x
        self.batch, __ , self.width = x.shape # (batch_size, in_channel, width)
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)

        out_width = int((self.width - self.kernel_size) / self.stride + 1) 
        out = np.zeros((self.batch, self.out_channel, out_width))

        for i in range(self.batch):
           for j in range(self.out_channel):
                s = 0
                for k in range(out_width):
                    out[i][j][k] = np.sum(self.W[j] * x[i][:, s:s + self.kernel_size]) + self.b[j]
                    s += self.stride

        return out # (self.batch, self.out_channel, out_width)
        
    def backward(self, delta): # delta : (self.batch, self.out_channel, out_width)
        
        ## Your codes here
        dx = np.zeros(self.x.shape) # (batch, in_channel, width)
        for i in range(self.batch):
            for j in range(self.out_channel):
                s = 0
                for k in range(delta.shape[2]):
                    self.dW[j] += self.x[i][:, s:s + self.kernel_size] * delta[i][j][k]
                    self.db[j] += 1.0 * delta[i][j][k]
                    dx[i][:,s:s + self.kernel_size] += self.W[j] * delta[i][j][k]
                    s += self.stride

        return dx

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x): # shape (batch, in_channel, width)
        
        self.in_channel = x.shape[1]
        self.width = x.shape[2]
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    def backward(self, delta):
        
        return delta.reshape(delta.shape[0],delta.shape[1],delta.shape[2])


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x>=0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta