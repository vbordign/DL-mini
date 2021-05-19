from modules import *

class ReLU(Module):
    '''
    Creates the ReLU activation module.

    Methods
    -------
    forward :
        runs a forward pass
    backward :
        accumulates gradient for backward pass

    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        output = x.clamp(min=0)
        return output

    def backward(self, gradwrtoutput):
        return self.x.clamp(min=0) * gradwrtoutput


class Tanh(Module):
    '''
    Creates the Tanh activation module.

    Methods
    -------
    forward :
        runs a forward pass
    backward :
        accumulates gradient for backward pass

    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return x.tanh()

    def backward(self, gradwrtoutput):
        return (1 - self.x.tanh().pow(2)) * gradwrtoutput


class Sigmoid(Module):
    '''
    Creates the Sigmoid activation module.

    Methods
    -------
    forward :
        runs a forward pass
    backward :
        accumulates gradient for backward pass

    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return x.sigmoid()

    def backward(self, gradwrtoutput):
        return self.x.sigmoid()*(1 - self.x.sigmoid()) * gradwrtoutput
