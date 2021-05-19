from modules import *

class LossMSE(Module):
    '''
    Creates the MSE Loss module.

    Methods
    -------
    loss :
        computes the MSE loss, and the gradient of the loss
    '''
    def __init__(self):
        super().__init__()

    def loss(self, target, output):
        loss= (target - output).norm(dim = 1).pow(2).mean()
        grad = -2 * (target - output)
        return loss, grad

class CrossEntropy(Module):
    '''
    Creates the Cross Entropy Loss module.

    Methods
    -------
    loss :
        computes the Cross Entropy loss, and the gradient of the loss
    '''
    def __init__(self):
        super().__init__()

    def loss(self, target, output):
        loss = (- output.log_softmax(dim = 1) * target).sum(1).mean()
        grad = output.softmax(dim = 1) - target
        return loss, grad
