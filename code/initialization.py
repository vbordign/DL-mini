import math

def initialize_uniform(input):
    std = math.sqrt(1/ input.size(1))
    return input.uniform_(-std, std)

def initialize_xavier_normal(input, gain = 1.0):
    std = gain * math.sqrt(2.0 / (input.size(0) + input.size(1)))
    return input.normal_(0, std)