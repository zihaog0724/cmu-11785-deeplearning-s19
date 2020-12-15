import torch
import torch.nn as nn
import numpy as np
import itertools

class Sigmoid:
    """docstring for Sigmoid"""
    def __init__(self):
        pass
    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res
    def backward(self):
        return self.res * (1-self.res)
    def __call__(self, x):
        return self.forward(x)


class Tanh:
    def __init__(self):
        pass
    def forward(self, x):
        self.res = np.tanh(x)
        return self.res
    def backward(self):
        return 1 - (self.res**2)
    def __call__(self, x):
        return self.forward(x)


class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)



        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        
    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        # 
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.h_t1 = h # h[t-1]

        # Update gate
        self.z_t = self.z_act(self.Wzh @ h + self.Wzx @ x) # shape(hidden dim)
        # Reset gate
        self.r_t = self.r_act(self.Wrh @ h + self.Wrx @ x) # shape(hidden dim)

        self.h_prime = self.Wh @ (self.r_t * h) # shape(hidden dim)

        # self.h_bar mainly contains data in x (current state)
        self.h_bar = self.h_act(self.h_prime + self.Wx @ x) # shape(hidden dim)

        h_t = (1.0 - self.z_t) * h + self.z_t * self.h_bar
        return h_t

    def backward(self, delta):
        # input:
        #   - delta:    shape(hidden dim), summation of derivative wrt loss from next layer at 
        #           same time-step and derivative wrt loss from same layer at
        #           next time-step
        #
        # output:
        #   - dx:   Derivative of loss wrt the input x
        #   - dh:   Derivative of loss wrt the input hidden h
        
        """
        Eqn (4) in the writeup
        """
        # dloss_dht = delta  ?
        dloss_dzt = delta * (self.h_bar - self.h_t1) # shape(hidden dim)
        dloss_dh_bar = delta * self.z_t # shape(hidden dim)
        dht_dht1 = 1.0 - self.z_t # shape(hidden dim)

        
        """
        Eqn (3) in the writeup
        """
        dloss_dh_act = dloss_dh_bar * self.h_act.backward() # shape(hidden dim)
        self.dWx = np.dot(dloss_dh_act.reshape(self.h,1), self.x.reshape(1,self.d)) # shape(hidden dim, input dim)

        # dh_dbar_dx is actually the derivative from the final loss
        dh_bar_dx = np.dot(dloss_dh_act.reshape(1,self.h), self.Wx) # shape(1, input dim)
        dh_bar_dx = dh_bar_dx.reshape(self.d) # shape(input dim)
        
        self.dWh = np.dot(dloss_dh_act.reshape(self.h,1), (self.r_t * self.h_t1).reshape(1,self.h)) # shape(hidden dim, hidden dim)

        # dh_dbar_dhr is actually the derivative from the final loss, it is dloss/d(r_t * h[t-1])
        dh_bar_dhr = np.dot(dloss_dh_act.reshape(1,self.h), self.Wh) # shape(1, hidden dim)
        dh_bar_dhr = dh_bar_dhr.reshape(self.h) # shape(hidden dim)

        # dh_dbar_drt and dh_dbar_dht1 are actually the derivatives from the final loss
        dh_bar_drt = dh_bar_dhr * self.h_t1 # shape(hidden dim)
        dh_bar_dht1 = dh_bar_dhr * self.r_t # shape(hidden dim)


        """
        Eqn (2) in the writeup
        """
        dloss_drt_dact = dh_bar_drt * self.r_act.backward() # shape(hidden dim)
        self.dWrh = np.dot(dloss_drt_dact.reshape(self.h,1), self.h_t1.reshape(1,self.h)) # shape(hidden dim, hidden dim)

        # drt_dht1 is actually the derivative from the final loss
        drt_dht1 = np.dot(dloss_drt_dact.reshape(1,self.h), self.Wrh) # shape(1, hidden dim)
        drt_dht1 = drt_dht1.reshape(self.h) # shape(hidden dim)

        self.dWrx = np.dot(dloss_drt_dact.reshape(self.h,1), self.x.reshape(1,self.d)) # shape(hidden dim, input dim)

        # drt_dx is actually the derivative from the final loss
        drt_dx = np.dot(dloss_drt_dact.reshape(1,self.h), self.Wrx) # shape(1, input dim)
        drt_dx = drt_dx.reshape(self.d) # shape(input dim)


        """
        Eqn (1) in the writeup
        """
        dloss_dzt_dact = dloss_dzt * self.z_act.backward()
        self.dWzh = np.dot(dloss_dzt_dact.reshape(self.h,1), self.h_t1.reshape(1,self.h)) # shape(hidden dim, hidden dim)

        # dzt_dht1 is actually the derivative from the final loss
        dzt_dht1 = np.dot(dloss_dzt_dact.reshape(1,self.h), self.Wzh) # shape(1, hidden dim)
        dzt_dht1 = dzt_dht1.reshape(self.h) # shape(hidden dim)

        self.dWzx = np.dot(dloss_dzt_dact.reshape(self.h,1), self.x.reshape(1,self.d)) # shape(hidden dim, input dim)

        # dzt_dx is actually the derivative from the final loss
        dzt_dx = np.dot(dloss_dzt_dact.reshape(1,self.h), self.Wzx) # shape(1, input dim)
        dzt_dx = dzt_dx.reshape(self.d) # shape(input dim)

        dx = dh_bar_dx + drt_dx + dzt_dx
        dht1 = delta * dht_dht1 + dh_bar_dht1 + drt_dht1 + dzt_dht1
        dx = dx.reshape(1,self.d)
        dht1 = dht1.reshape(1,self.h)

        return dx, dht1



if __name__ == '__main__':
    test()









