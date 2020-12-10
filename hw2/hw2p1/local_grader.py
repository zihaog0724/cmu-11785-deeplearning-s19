import torch
import torch.nn as nn
import numpy as np
from  torch.autograd import Variable
import multiprocessing as mtp
import traceback


np.random.seed(11785)


def test_cnn_correctness_once(idx):
    from layers import Conv1D

    scores_dict = [0,0,0,0]

    ############################################################################################
    #############################   Initialize hyperparameters    ##############################
    ############################################################################################
    rint = np.random.randint
    norm = np.linalg.norm
    in_c, out_c = rint(5,15), rint(5,15)
    kernel, stride =  rint(1,10), rint(1,10)
    batch, width = rint(1,4), rint(20,300)



    def info():
        print('\nTesting model:')
        print('    in_channel: {}, out_channel: {},'.format(in_c,out_c))
        print('    kernel size: {}, stride: {},'.format(kernel,stride))
        print('    batch size: {}, input size: {}.'.format(batch,width))


    ##############################################################################################
    ##########    Initialize the CNN layer and copy parameters to a PyTorch CNN layer   ##########
    ##############################################################################################
    try:
        net = Conv1D(in_c, out_c, kernel, stride)
    except:
        info()
        print('Failed to pass parameters to your Conv1D function!')
        return scores_dict

    model = nn.Conv1d(in_c, out_c, kernel, stride)
    model.weight = nn.Parameter(torch.tensor(net.W))
    model.bias = nn.Parameter(torch.tensor(net.b))


    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x = np.random.randn(batch, in_c, width)
    x1 = Variable(torch.tensor(x),requires_grad=True)
    y1 = model(x1)
    b, c, w = y1.shape
    delta = np.random.randn(b,c,w)
    y1.backward(torch.tensor(delta))


    #############################################################################################
    ##########################    Get your forward results and compare ##########################
    #############################################################################################
    y = net(x)
    assert y.shape == y1.shape
    if not(y.shape == y1.shape): print("FAILURE")


    forward_res = y - y1.detach().numpy()
    forward_res_norm = abs(forward_res).max()


    if forward_res_norm < 1e-12:
        scores_dict[0] =  1
    else:
        info()
        print('Fail to return correct forward values')
        return scores_dict

    #############################################################################################
    ##################   Get your backward results and check the tensor shape ###################
    #############################################################################################
    dx = net.backward(delta)

    assert dx.shape == x.shape    
    assert net.dW.shape == model.weight.grad.detach().numpy().shape
    assert net.db.shape == model.bias.grad.detach().numpy().shape
    #############################################################################################
    ################   Check your dx, dW and db with PyTorch build-in functions #################
    #############################################################################################
    dx1 = x1.grad.detach().numpy()
    delta_res_norm = abs(dx - dx1).max()

    dW_res = net.dW - model.weight.grad.detach().numpy()
    dW_res_norm = abs(dW_res).max()

    db_res = net.db - model.bias.grad.detach().numpy()
    db_res_norm = abs(db_res).max()


    if delta_res_norm < 1e-12:
        scores_dict[1] = 1
    
    if dW_res_norm < 1e-12:
        scores_dict[2] = 1

    if db_res_norm < 1e-12:
        scores_dict[3] = 1

    if min(scores_dict) != 1:
        info()
        if scores_dict[1] == 0:
            print('Fail to return correct backward values dx')
        if scores_dict[2] == 0:
            print('Fail to return correct backward values dW')
        if scores_dict[3] == 0:
            print('Fail to return correct backward values db')
    return scores_dict





def test_cnn_correctness():
    scores = []
    worker = min(mtp.cpu_count(),4)
    p = mtp.Pool(worker)
    
    for __ in range(15):
        scores_dict = test_cnn_correctness_once(__) 
        scores.append(scores_dict)
        if min(scores_dict) != 1:
            break
    
    scores = np.array(scores).min(0)
    return list(scores)

import cnn as cnn_solution

def test_part_b():
    data = np.loadtxt('data/data.asc').T.reshape(1, 24, -1)
    cnn = cnn_solution.CNN_B()
    weights = np.load('weights/mlp_weights_part_b.npy')
    cnn.init_weights(weights)

    expected_result = np.load('autograde/res_b.npy')
    result = cnn(data)

    try:
        assert(type(result)==type(expected_result))
        assert(result.shape==expected_result.shape)
        #print(expected_result)
        assert(np.allclose(result,expected_result))

        return True
    except Exception as e:
        traceback.print_exc()
        return False


def test_part_c():
    data = np.loadtxt('data/data.asc').T.reshape(1, 24, -1)
    cnn = cnn_solution.CNN_C()
    weights = np.load('weights/mlp_weights_part_c.npy')
    cnn.init_weights(weights)

    expected_result = np.load('autograde/res_c.npy')
    result = cnn(data)

    try:
        assert(type(result)==type(expected_result))
        assert(result.shape==expected_result.shape)
        assert(np.allclose(result,expected_result))

        return True
    except Exception as e:
        traceback.print_exc()
        return False










if __name__ == '__main__':
    a, b, c, d = test_cnn_correctness()
    string = 'CNN layer forward: {}; backward dx: {}, dW: {}, db: {}.'.format(a,b,c,d)

    print('Conv1D Forward:', 'PASS' if a == 1 else 'FAIL')
    print('Conv1D dX:', 'PASS' if b == 1 else 'FAIL')
    print('Conv1D dW:', 'PASS' if c == 1 else 'FAIL')
    print('Conv1D db:', 'PASS' if d == 1 else 'FAIL')

    b = test_part_b()
    print("PART B:", "PASS" if b else "FAIL")
    c = test_part_c()
    print("PART C:", "PASS" if c else "FAIL")    



    





