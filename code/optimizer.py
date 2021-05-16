from torch import empty
import math

class SGD():
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p[0] -= self.lr * p[1]


class Adam():
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.counter = 0
        self.moments = [[empty(p[1].size()).zero_(), empty(p[1].size()).zero_()] for p in params]

    def step(self):
        self.counter += 1
        for i, param in enumerate(self.params):
            grad = param[1]
            exp_avg = self.moments[i][0]
            exp_avg_sq = self.moments[i][1]

            exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad
            exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * grad.pow(2)

            m1_corr = exp_avg/ (1 - math.pow(self.beta1, self.counter))
            m2_corr = exp_avg_sq/ (1 - math.pow(self.beta2, self.counter))

            delta = m1_corr / (m2_corr.sqrt() + self.eps)
            param[0] -= self.lr * delta


class Adagrad():
    def __init__(self, params, lr=0.001, eps = 1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.moments = [empty(p[1].size()).zero_() for p in params]

    def step(self):
        for i, param in enumerate(self.params):
            grad = param[1]
            exp_avg = self.moments[i]

            exp_avg = exp_avg + grad.pow(2)

            delta = grad / (exp_avg + self.eps).sqrt()
            param[0] -= self.lr * delta

class Adadelta():
    def __init__(self, params, beta=0.95, eps=1e-8):
        self.params = params
        self.beta = beta
        self.eps = eps
        self.moments = [[empty(p[1].size()).zero_(), empty(p[1].size()).zero_()] for p in params]

    def step(self):
        for i, param in enumerate(self.params):
            grad = param[1]
            exp_avg_sq = self.moments[i][0]
            delta = self.moments[i][1]

            exp_avg_sq = self.beta * exp_avg_sq + (1 - self.beta) * grad.pow(2)

            diff = -(delta + self.eps).sqrt() / (exp_avg_sq + self.eps).sqrt() * grad

            param[0] += diff

            c = diff

            delta = self.beta * delta + (1 - self.beta) * c.pow(2)


class RMSprop():
    def __init__(self, params, lr=0.001, beta=0.95, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.moments = [empty(p[1].size()).zero_()for p in params]

    def step(self):
        for i, param in enumerate(self.params):
            grad = param[1]
            exp_avg_sq = self.moments[i]

            exp_avg_sq = self.beta * exp_avg_sq + (1 - self.beta) * grad.pow(2)

            diff = grad / (exp_avg_sq.sqrt() + self.eps)

            param[0] -= self.lr* diff

