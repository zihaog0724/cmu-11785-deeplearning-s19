"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        self.state = 1./(1.+np.exp(-x))
        return self.state

    def derivative(self):

        # Maybe something we need later in here...

        return self.state * (1. - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - self.state**2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(x, 0.)
        return self.state

    def derivative(self):
        self.state = 1. * (self.state > 0.)
        return self.state

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y

        softmax = np.ones((x.shape[0],x.shape[1]))
        for i in range(len(x)):
            exp = np.exp(x[i])
            softmax[i] = exp/np.sum(exp) 

        self.sm = softmax

        log_x = np.log(softmax)
        log_xy = log_x * y
        cross_entropy = np.ones((log_xy.shape[0],1))

        for j in range(len(log_xy)):
            cross_entropy[j] = -1. * np.sum(log_xy[j])

        cross_entropy = np.squeeze(cross_entropy)
        
        return cross_entropy

    def derivative(self):

        # self.sm might be useful here...
        dcedsm = -1. * (self.labels/self.sm)
        dsmdx = np.ones((self.logits.shape[0],self.logits.shape[1],self.logits.shape[1]))
     
        for i in range(dsmdx.shape[0]):
            for j in range(dsmdx.shape[1]):
                for k in range(dsmdx.shape[2]):
                    if j == k:
                        dsmdx[i][j][k] = self.sm[i][j] * (1. - self.sm[i][j])
                    else:
                        dsmdx[i][j][k] = -1. * self.sm[i][j] * self.sm[i][k]

        gradients = np.ones((self.logits.shape[0],self.logits.shape[1]))
        for i in range(gradients.shape[0]):
            gradients[i] = np.dot(dcedsm[i],dsmdx[i])

        return gradients


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        if eval:
            self.mean = self.running_mean
            self.var = self.running_var
            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta
            
        else:
            self.x = x
            self.mean = np.mean(x, axis=0).reshape(1,self.x.shape[1])
            self.var = np.var(x, axis=0).reshape(1,self.x.shape[1])
            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta

            # update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1. - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out

    def backward(self, delta): # delta shape(num_examples, hiddens)

        #delta = delta.reshape(delta.shape[0],delta.shape[1]) # (num_examples,hiddens) e.g (20,5)
        self.dgamma = np.sum(delta * self.norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        dloss_dnorm = delta * self.gamma # (num_examples,hiddens) e.g (20,5)
        numerator = self.x - self.mean
        denominator = 1. / np.sqrt(self.var + self.eps)

        dnorm_dvar = numerator * -0.5 * denominator**3
        dloss_dvar = np.sum(dloss_dnorm * dnorm_dvar,axis=0) # (1, hiddens) e.g.(1,5)

        dloss_dmean =  np.sum(dloss_dnorm * -1.0 * denominator, axis=0) + dloss_dvar * np.mean(-2.0 * numerator, axis=0) # (1, hiddens) e.g.(1,5)

        dloss_dx = dloss_dnorm * denominator + dloss_dvar * 2.0 * numerator / self.x.shape[0] + dloss_dmean / self.x.shape[0]

        return dloss_dx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    
    return np.random.normal(size=(d0,d1))


def zeros_bias_init(d):
    
    return np.zeros(d)

class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
        if len(hiddens) != 0:
            for i in range(len(hiddens)):
                if i == 0:
                    self.W.append(weight_init_fn(input_size,hiddens[i]))
                    self.dW.append(np.zeros((input_size,hiddens[i])))
                else:
                    self.W.append(weight_init_fn(hiddens[i-1],hiddens[i]))
                    self.dW.append(np.zeros((hiddens[i-1],hiddens[i])))
                self.b.append(bias_init_fn(hiddens[i]))
                self.db.append(zeros_bias_init(hiddens[i]))
            self.W.append(weight_init_fn(hiddens[-1],output_size))
            self.dW.append(np.zeros((hiddens[-1],output_size)))
            self.b.append(bias_init_fn(output_size))
            self.db.append(zeros_bias_init(output_size))
        else:
            self.W.append(weight_init_fn(input_size,output_size))
            self.dW.append(np.zeros((input_size,output_size)))
            self.b.append(bias_init_fn(output_size))
            self.db.append(zeros_bias_init(output_size))

        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = []
            for i in range(num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.x = None # Original inputs
        self.z = None # outputs before softmax
        self.w_velocity = [np.zeros_like(w) for w in self.W]
        self.b_velocity = [np.zeros_like(b) for b in self.b]

    def forward(self, x):
     
        self.x = x
        self.neurons = []
        self.neurons.append(x)

        for i in range(len(self.W)):
            if i == 0:
                self.z = np.dot(x,self.W[i]) + self.b[i]
                if self.bn:
                    if self.train_mode:
                        self.z = self.bn_layers[0].forward(self.z,eval=False)
                    else:
                        self.z = self.bn_layers[0].forward(self.z,eval=True)
                self.z = self.activations[i].forward(self.z)
                self.neurons.append(self.z)
            else:
                self.z = np.dot(self.z,self.W[i]) + self.b[i]
                if self.bn and i < len(self.bn_layers):
                    if self.train_mode:
                        self.z = self.bn_layers[i].forward(self.z,eval=False)
                    else:
                        self.z = self.bn_layers[i].forward(self.z,eval=True)
                self.z = self.activations[i].forward(self.z)
                self.neurons.append(self.z)
        return self.z # a list consisting all outputs after activation of every layer, including original inputs

    def zero_grads(self):
 
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b] # self.b[i] shape(1,hiddens[i]) but self.db[i] should be a vector
        for i in range(len(self.db)):
            self.db[i] = self.db[i][0]

        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].dgamma = np.zeros_like(self.bn_layers[i].dgamma)
                self.bn_layers[i].dbeta = np.zeros_like(self.bn_layers[i].dbeta)

        return self.dW, self.db

    def step(self):
        
        self.w_velocity = [self.momentum * self.w_velocity[i] + self.lr * self.dW[i] for i in range(len(self.w_velocity))]
        self.b_velocity = [self.momentum * self.b_velocity[i] + self.lr * self.db[i] for i in range(len(self.b_velocity))]
        self.W = [self.W[i] - self.w_velocity[i] for i in range(len(self.W))]
        self.b = [self.b[i] - self.b_velocity[i] for i in range(len(self.b))]
        
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta            
        return self.W, self.b

    def backward(self, labels):

        sm_ce = self.criterion
        cross_entropy = sm_ce.forward(self.z, labels) #(num_examples,) vector
        dcedz = sm_ce.derivative() # (num_examples, output_size)
        dcedz = dcedz * self.activations[-1].derivative()
    
        dW = np.matmul(self.neurons[-2].reshape(self.neurons[-2].shape[0],self.neurons[-2].shape[1],1), dcedz.reshape(dcedz.shape[0],1,dcedz.shape[1]))        
        db = dcedz

        dy = np.matmul(np.repeat(self.W[-1][np.newaxis,:,:],self.x.shape[0],axis=0),dcedz.reshape(dcedz.shape[0],dcedz.shape[1],1)) # shape(num_examples,num_hiddens,1)
       
        self.dW[-1] = np.mean(dW,axis=0)
        self.db[-1] = np.mean(db,axis=0)

        dloss_dw = dW
        dloss_db = db
        for j in reversed(range(len(self.dW) - 1)):
            d_activation = self.activations[j].derivative() #shape(num_examples,hiddens[1])
            dloss_dact = d_activation * dy.reshape(dy.shape[0],dy.shape[1])
            if self.bn and j < len(self.bn_layers):
                dloss_dbn = self.bn_layers[j].backward(dloss_dact)
                dloss_db = dloss_dbn
                self.db[j] = np.mean(dloss_db,axis=0)
                dloss_dw = np.matmul(self.neurons[j].reshape(self.neurons[j].shape[0],self.neurons[j].shape[1],1),dloss_dbn.reshape(dloss_dbn.shape[0],1,dloss_dbn.shape[1]))
                self.dW[j] = np.mean(dloss_dw,axis=0)
                dy = np.matmul(np.repeat(self.W[j][np.newaxis,:,:],self.x.shape[0],axis=0),dloss_dbn.reshape(dloss_dbn.shape[0],dloss_dbn.shape[1],1))

            else:
                dloss_db = dloss_dact
                self.db[j] = np.mean(dloss_db,axis=0)
                dloss_dw = np.matmul(self.neurons[j].reshape(self.neurons[j].shape[0],self.neurons[j].shape[1],1),dloss_dact.reshape(dloss_dact.shape[0],1,dloss_dact.shape[1]))
                self.dW[j] = np.mean(dloss_dw,axis=0)
                dy = np.matmul(np.repeat(self.W[j][np.newaxis,:,:],self.x.shape[0],axis=0),dloss_dact.reshape(dloss_dact.shape[0],dloss_dact.shape[1],1))

        return cross_entropy

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))
    train_num_batches = len(trainx) / batch_size
    val_num_batches = len(valx) / batch_size

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    for e in range(nepochs):

        print('Epoch:'+str(e+1))
        mlp.train()
        epoch_train_loss = 0
        epoch_val_loss = 0  

        for b in range(0, len(trainx), batch_size):

            mlp.zero_grads()
            outputs = mlp(trainx[b:b+batch_size])
            cross_entropy_loss = mlp.backward(trainy[b:b+batch_size])
            mlp.step()
            epoch_train_loss += cross_entropy_loss

        training_losses.append(np.mean(epoch_train_loss / train_num_batches))

        mlp.eval()
        output_train = mlp(trainx)
        pred_train = (output_train == output_train.max(axis=1)[:,None]).astype(float)
        correct_train = 0
        for i in range(len(trainy)):
            if (trainy[i] == pred_train[i]).all():
                correct_train += 1
        training_errors.append(1. - correct_train / len(trainy))

        for b in range(0, len(valx), batch_size):

            val_output = mlp(valx[b:b+batch_size])
            val_loss = SoftmaxCrossEntropy().forward(val_output,valy[b:b+batch_size])
            epoch_val_loss += val_loss

        validation_losses.append(np.mean(epoch_val_loss / val_num_batches))
        
        output_val = mlp(valx)
        pred_val = (output_val == output_val.max(axis=1)[:,None]).astype(float)
        correct_val = 0
        for i in range(len(valy)):
            if (valy[i] == pred_val[i]).all():
                correct_val += 1
        validation_errors.append(1. - correct_val / len(valy))       



        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

    #raise NotImplemented

