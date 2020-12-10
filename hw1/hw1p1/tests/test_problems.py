import numpy as np
import os
from helpers.helpers import *
import sys
import pickle

autolab = bool(int(os.environ['AUTOLAB'])) if 'AUTOLAB' in os.environ.keys() else False
saved_data = pickle.load(open("data.pkl", 'rb'))
rtol = 1e-4
atol = 1e-04
TOLERANCE = 1e-4

SEED = 2019
if autolab:
    print("We are on Autolab")
    TRAINDATAPATH = "/datasets/11785/mnist_train.csv"
    TESTDATAPATH = "/datasets/11785/mnist_test.csv"
    sys.path.append('handin/')
else:
    print("We are on local")
    TRAINDATAPATH = "tests/data/mnist_train.csv"
    TESTDATAPATH = "tests/data/mnist_test.csv"

if os.path.exists(TRAINDATAPATH):
    print("Train data exists")
if os.path.exists(TESTDATAPATH):
    print("Test data exists")

from hw1 import hw1


def raw_mnist(path):
    return (cleaned_mnist(path))


def cleaned_mnist(path):
    data = np.genfromtxt(path, delimiter=',')
    X = data[:, 1:]
    Y = data[:, 0]
    Y = Y.astype(int)
    return X, Y


def reset_prng():
    np.random.seed(11785)


def weight_init(x, y):
    return np.random.randn(x, y)


def bias_init(x):
    return np.zeros((1, x))


def test_softmax_cross_entropy_forward():
    data = saved_data[0]
    x = data[0]
    y = data[1]
    sol = data[2]

    ce = hw1.SoftmaxCrossEntropy()
    closeness_test(ce(x, y), sol, "ce(x, y)")


def test_softmax_cross_entropy_derivative():
    data = saved_data[1]
    x = data[0]
    y = data[1]
    sol = data[2]
    ce = hw1.SoftmaxCrossEntropy()
    ce(x, y)
    closeness_test(ce.derivative(), sol, "ce.derivative()")


def test_linear_classifier_forward():
    data = saved_data[2]
    x = data[0]
    gt = data[1]
    reset_prng()
    mlp = hw1.MLP(784, 10, [], [hw1.Identity()], weight_init, bias_init,
                  hw1.SoftmaxCrossEntropy(), 0.008, momentum=0.0,
                  num_bn_layers=0)
    pred = mlp.forward(x)
    closeness_test(pred, gt, "mlp.forward(x)")


def test_linear_classifier_backward():
    data = saved_data[3]
    x = data[0]
    y = data[1]
    soldW = data[2]
    soldb = data[3]
    reset_prng()
    mlp = hw1.MLP(784, 10, [], [hw1.Identity()], weight_init, bias_init,
                  hw1.SoftmaxCrossEntropy(), 0.008, momentum=0.0,
                  num_bn_layers=0)
    mlp.forward(x)
    mlp.backward(y)
    closeness_test(mlp.dW[0], soldW, "mlp.dW")
    closeness_test(mlp.db[0], soldb, "mlp.db")


def test_linear_classifier_step():
    data = saved_data[4]
    x = data[0]
    y = data[1]
    solW = data[2]
    solb = data[3]
    reset_prng()
    mlp = hw1.MLP(784, 10, [], [hw1.Identity()], weight_init, bias_init,
                  hw1.SoftmaxCrossEntropy(), 0.008, momentum=0.0,
                  num_bn_layers=0)
    num_test_updates = 5
    for u in range(num_test_updates):
        mlp.zero_grads()
        mlp.forward(x)
        mlp.backward(y)
        mlp.step()
    closeness_test(mlp.W[0], solW, "mlp.W[0]")
    closeness_test(mlp.b[0], solb, "mlp.b[0]")


def test_sigmoid_forward():
    data = saved_data[5]
    t0 = data[0]
    gt = data[1]
    student = hw1.Sigmoid()
    student(t0)
    closeness_test(student.state, gt, "sigmoid.state")


def test_sigmoid_derivative():
    data = saved_data[6]
    t0 = data[0]
    gt = data[1]
    student = hw1.Sigmoid()
    student(t0)
    closeness_test(student.derivative(), gt, "sigmoid.derivative()")


def test_relu_forward():
    data = saved_data[7]
    t0 = data[0]
    gt = data[1]
    student = hw1.ReLU()
    student(t0)
    closeness_test(student.state, gt, "relu.state")


def test_relu_derivative():
    data = saved_data[8]
    t0 = data[0]
    gt = data[1]
    student = hw1.ReLU()
    student(t0)
    closeness_test(student.derivative(), gt, "relu.derivative()")


def test_tanh_forward():
    data = saved_data[9]
    t0 = data[0]
    gt = data[1]
    student = hw1.Tanh()
    student(t0)
    closeness_test(student.state, gt, "tanh.state")


def test_tanh_derivative():
    data = saved_data[10]
    t0 = data[0]
    gt = data[1]
    student = hw1.Tanh()
    student(t0)
    closeness_test(student.derivative(), gt, "tanh.derivative()")


def test_single_hidden_forward():
    data = saved_data[11]
    x = data[0]
    gt = data[1]
    reset_prng()
    mlp = hw1.MLP(784, 10, [32], [hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)

    pred = mlp.forward(x)
    closeness_test(pred, gt, "mlp.forward(x)")


def test_single_hidden_backward():
    data = saved_data[12]
    assert len(data) == 6
    x = data[0]
    y = data[1]
    soldW = data[2:4]
    soldb = data[4:]
    reset_prng()
    mlp = hw1.MLP(784, 10, [32], [hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)
    mlp.forward(x)
    mlp.backward(y)

    for i, (pred, gt) in enumerate(zip(mlp.dW, soldW)):
        closeness_test(pred, gt, "mlp.dW[%d]" % i)

    for i, (pred, gt) in enumerate(zip(mlp.db, soldb)):
        closeness_test(pred, gt, "mlp.db[%d]" % i)


def test_mystery_hidden_forward1():
    data = saved_data[13]
    x = data[0]
    gt = data[1]
    reset_prng()
    mlp = hw1.MLP(784, 10, [64, 32], [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)

    pred = mlp.forward(x)
    closeness_test(pred, gt, "mlp.forward(x)")


def test_mystery_hidden_forward2():
    data = saved_data[14]
    x = data[0]
    gt = data[1]
    reset_prng()
    mlp = hw1.MLP(784, 10, [32, 32, 32, 32, 32],
                  [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Sigmoid(), hw1.Sigmoid(),
                   hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)

    pred = mlp.forward(x)
    closeness_test(pred, gt, "mlp.forward(x)")


def test_mystery_hidden_forward3():
    data = saved_data[15]
    x = data[0]
    gt = data[1]
    reset_prng()
    mlp = hw1.MLP(784, 10, [32], [hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)

    pred = mlp.forward(x)
    closeness_test(pred, gt, "mlp.forward(x)")


def test_mystery_hidden_backward1():
    data = saved_data[16]
    assert len(data) == 8
    x = data[0]
    y = data[1]
    soldW = data[2:5]
    soldb = data[5:]
    reset_prng()
    mlp = hw1.MLP(784, 10, [64, 32], [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)

    mlp.forward(x)
    mlp.backward(y)

    for i, (pred, gt) in enumerate(zip(mlp.dW, soldW)):
        closeness_test(pred, gt, "mlp.dW[%d]" % i)

    for i, (pred, gt) in enumerate(zip(mlp.db, soldb)):
        closeness_test(pred, gt, "mlp.db[%d]" % i)


def test_mystery_hidden_backward2():
    data = saved_data[17]
    assert len(data) == 14
    x = data[0]
    y = data[1]
    soldW = data[2:8]
    soldb = data[8:]
    reset_prng()
    mlp = hw1.MLP(784, 10, [32, 32, 32, 32, 32],
                  [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Sigmoid(), hw1.Sigmoid(),
                   hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)
    mlp.forward(x)
    mlp.backward(y)

    for i, (pred, gt) in enumerate(zip(mlp.dW, soldW)):
        closeness_test(pred, gt, "mlp.dW[%d]" % i)

    for i, (pred, gt) in enumerate(zip(mlp.db, soldb)):
        closeness_test(pred, gt, "mlp.db[%d]" % i)


def test_mystery_hidden_backward3():
    data = saved_data[18]
    assert len(data) == 6
    x = data[0]
    y = data[1]
    soldW = data[2:4]
    soldb = data[4:]
    reset_prng()
    mlp = hw1.MLP(784, 10, [32], [hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=0)
    mlp.forward(x)
    mlp.backward(y)

    for i, (pred, gt) in enumerate(zip(mlp.dW, soldW)):
        closeness_test(pred, gt, "mlp.dW[%d]" % i)

    for i, (pred, gt) in enumerate(zip(mlp.db, soldb)):
        closeness_test(pred, gt, "mlp.db[%d]" % i)


def test_batch_norm_train():
    data = saved_data[19]
    assert len(data) == 10
    x = data[0]
    y = data[1]
    soldW = data[2:5]
    soldb = data[5:8]
    soldbeta = data[8]
    soldgamma = data[9]

    reset_prng()

    mlp = hw1.MLP(784, 10, [64, 32], [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=1)

    mlp.forward(x)
    mlp.backward(y)

    for i, (pred, gt) in enumerate(zip(mlp.dW, soldW)):
        closeness_test(pred, gt, "mlp.dW[%d]" % i)

    for i, (pred, gt) in enumerate(zip(mlp.db, soldb)):
        closeness_test(pred, gt, "mlp.db[%d]" % i)

    closeness_test(mlp.bn_layers[0].dbeta, soldbeta, "mlp.bn_layers[0].dbeta")
    closeness_test(mlp.bn_layers[0].dgamma, soldgamma, "mlp.bn_layers[0].dgamma")


def test_batch_norm_inference():
    num_examples = 1000
    data = saved_data[20]
    assert len(data) == 15
    x = data[0]
    y = data[1]
    soldbeta = data[2]
    soldgamma = data[3]
    xs = data[4]
    solground = data[5:]
    reset_prng()
    mlp = hw1.MLP(784, 10, [64, 32], [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Identity()],
                  weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.0, num_bn_layers=1)

    batch_size = 100
    mlp.train()
    for b in range(0, 1):
        mlp.zero_grads()
        mlp.forward(x[b:b + batch_size])
        mlp.backward(y[b:b + batch_size])
        mlp.step()
        closeness_test(mlp.bn_layers[0].dbeta, soldbeta, "mlp.bn_layers[0].dbeta")
        closeness_test(mlp.bn_layers[0].dgamma, soldgamma, "mlp.bn_layers[0].dgamma")

    for b in range(0, num_examples, batch_size):
        mlp.eval()
        student = mlp.forward(xs[b:b + batch_size])
        ground = solground[b//batch_size]
        closeness_test(student, ground, "mlp.forward(x)")


def test_momentum():
    data = saved_data[21]
    assert len(data) == 8
    x = data[0]
    y = data[1]
    solW = data[2:5]
    solb = data[5:]
    reset_prng()
    mlp = hw1.MLP(784, 10, [64, 32], [hw1.Sigmoid(), hw1.Sigmoid(), hw1.Identity()], weight_init, bias_init, hw1.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.856, num_bn_layers=0)

    num_test_updates = 5
    for u in range(num_test_updates):
        mlp.zero_grads()
        mlp.forward(x)
        mlp.backward(y)
        mlp.step()
    mlp.eval()

    for i, (pred, gt) in enumerate(zip(mlp.W, solW)):
        closeness_test(pred, gt, "mlp.W[%d]" % i)

    for i, (pred, gt) in enumerate(zip(mlp.b, solb)):
        closeness_test(pred, gt, "mlp.b[%d]" % i)


def failed_test_names(names, preds, gts, status):
    values = [(preds[i], gts[i]) for i, s in enumerate(status) if not s]
    names = [n for n, s in zip(names, status) if not s]
    return names, values


def union(xs, ys):
    return [x or y for x, y in zip(xs, ys)]


def assert_any_zeros(nparr):
    for i in range(len(nparr)):
        assert (np.all(nparr[i], 0))
