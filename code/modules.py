from torch import empty
from initialization import *

class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias_flag=True, init = 'uniform'):
        super().__init__()
        self.bias_flag = bias_flag

        if init == 'xavier':
            self.w = [initialize_xavier_normal(empty(out_features, in_features)),
                      empty(out_features, in_features).zero_()]
        else:
            self.w = [initialize_uniform(empty(out_features, in_features)),
                      empty(out_features, in_features).zero_()]
        if bias_flag:
            if init == 'xavier':
                self.bias = [initialize_xavier_normal(empty(out_features, 1)),
                             empty(out_features, 1).zero_()]
            else:
                self.bias = [initialize_uniform(empty(out_features, 1)),
                         empty(out_features, 1).zero_()]
        else:
            self.bias = [empty(out_features, 1).zero_(), empty(out_features, 1).zero_()]

    def forward(self, x):
        self.x = x
        output = self.w[0].mm(x.t())+ self.bias[0]
        return output.t()

    def backward(self, gradwrtoutput):
        if self.bias_flag:
            self.bias[1] = gradwrtoutput.sum(0)[:,None]

        self.w[1] = gradwrtoutput.t().mm(self.x)
        return gradwrtoutput.mm(self.w[0])

    def param(self):
        return [self.w, self.bias]


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.modules = []
        for module in args:
            self.modules.append(module)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, gradwrtoutput):
        for module in self.modules[::-1]:
            gradwrtoutput = module.backward(gradwrtoutput)

    def param(self):
        param_list = []
        for module in self.modules:
            param_list = param_list + module.param()
        return param_list

    def zero_grad(self):
        param_list = self.param()
        for p in param_list:
            p[1].zero_()

