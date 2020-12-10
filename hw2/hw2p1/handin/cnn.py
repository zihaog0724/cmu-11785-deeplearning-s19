from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24,8,8,4),ReLU(),Conv1D(8,16,1,1),ReLU(),Conv1D(16,4,1,1),Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        self.layers[0].W = weights[0].T.reshape(8,8,24).swapaxes(1,2)
        self.layers[2].W = weights[1].T.reshape(16,8,1)
        self.layers[4].W = weights[2].T.reshape(4,16,1)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24,8,8,4),ReLU(),Conv1D(8,16,1,1),ReLU(),Conv1D(16,4,1,1),Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        self.layers[0].W = weights[0].T.reshape(8,8,24).swapaxes(1,2)
        self.layers[2].W = weights[1].T.reshape(16,8,1)
        self.layers[4].W = weights[2].T.reshape(4,16,1)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
