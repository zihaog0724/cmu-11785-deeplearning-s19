import numpy as np
import os
import time
import torch    
    


def sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array.
    y is a 1-dimensional int numpy array.
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
    return result


def vectorize_sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array. Shape of x is (N, ).
    y is a 1-dimensional int numpy array. Shape of y is (N, ).
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> vectorize_sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    # Write the vecotrized version here
    return x.sum() * y.sum()


def Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
    return result

def vectorize_Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    return np.maximum(x, 0)


def ReluPrime(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result


def vectorize_PrimeRelu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    return np.where(x >= 0, 1, 0)


def slice_fixed_point(x, l, start_point):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should have.
    start_point is an integer representing the point at which the final utterance should start in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    sliced_arr = []
    for i in range(len(x)):
        x[i] = np.array(x[i])[start_point:start_point+l,:].tolist()
        sliced_arr.append(x[i])

    return np.array(sliced_arr)


def slice_last_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    sliced_arr = []
    for i in range(len(x)):
        row = list(x[i])
        sliced_arr.append(row[len(row)-l:len(row)])

    return np.array(sliced_arr)


def slice_random_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    sliced_arr = []
    for i in range(len(x)):
        start_point = np.random.randint(0,len(x[i])-l+1)
        x[i] = np.array(x[i])[start_point:start_point+l,:].tolist()
        sliced_arr.append(x[i])

    return np.array(sliced_arr)


def pad_pattern_end(x):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    padded_arr = []
    max_len = len(max(list(x),key=len))
    for i in range(len(x)):
        diff = max_len-len(x[i])
        padded = np.pad(x[i],(0,diff),'symmetric')
        padded_cut = padded[:,0:len(padded[0])-diff].tolist()
        padded_arr.append(padded_cut)

    return np.array(padded_arr)
    


def pad_constant_central(x, c_):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    padded_arr = []
    max_len = len(max(list(x),key=len))
    features = len(x[0][0])
    for i in range(len(x)):
        if (max_len - len(x[i])) % 2 == 0:
            left = (max_len - len(x[i])) / 2
            right = (max_len - len(x[i])) / 2
        else:
            left = (max_len - len(x[i])) / 2 - 0.5
            right = (max_len - len(x[i])) / 2 + 0.5
        
        left_padded = int(left) * [features * [c_]]
        original = list(x[i])
        right_padded = int(right) * [features * [c_]]
        padded = left_padded + original + right_padded
        padded_arr.append(padded)   

    return np.array(padded_arr)



def numpy2tensor(x):
    """
    x is an numpy nd-array. 

    Return a pytorch Tensor of the same shape containing the same data.
    """
    return torch.from_numpy(x)

def tensor2numpy(x):
    """
    x is a pytorch Tensor. 

    Return a numpy nd-array of the same shape containing the same data.
    """
    return x.numpy()

def tensor_sumproducts(x,y):
    """
    x is an n-dimensional pytorch Tensor.
    y is an n-dimensional pytorch Tensor.

    Return the sum of the element-wise product of the two tensors.
    """
    return torch.sum(x*y)

def tensor_ReLU(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the ReLU function: 
    RELU(i) = 0 if i < 0 else i

    Return a pytorch Tensor of the same shape as x containing RELU(x)
    """
    return torch.max(x,torch.zeros(x.shape))        

def tensor_ReLU_prime(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the RELU_PRIME function: 
    RELU_PRIME(i) = 0 if i < 0 else 1

    Return a pytorch Tensor of the same shape as x containing RELU_PRIME(x)
    """
    x_np = np.where(x.numpy() >= 0, 1.000, 0.000)
    x_torch = torch.from_numpy(x_np)
    x_torch = x_torch.type(torch.FloatTensor)
    return x_torch



