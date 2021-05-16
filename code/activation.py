from modules import *

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        output = x.clamp(min=0)
        return output

    def backward(self, gradwrtoutput):
        return self.x.clamp(min=0) * gradwrtoutput


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return x.tanh()

    def backward(self, gradwrtoutput):
        return (1 - self.x.tanh().pow(2)) * gradwrtoutput


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return x.sigmoid()

    def backward(self, gradwrtoutput):
        return self.x.sigmoid()*(1 - self.x.sigmoid()) * gradwrtoutput
