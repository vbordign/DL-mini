import math

def initialize_uniform(input):
    '''
    Initializes the input tensor from a uniform distribution.

    Parameters
    ----------
    input: FloatTensor
        input tensor

    Returns
    -------
    : FloatTensor
        initialized tensor
    '''
    std = math.sqrt(1/ input.size(1))
    return input.uniform_(-std, std)

def initialize_xavier_normal(input, gain = 1.0):
    '''
    Initializes the input tensor from a normal distribution.

    Parameters
    ----------
    input: FloatTensor
        input tensor

    Returns
    -------
    : FloatTensor
        initialized tensor
    '''
    std = gain * math.sqrt(2.0 / (input.size(0) + input.size(1)))
    return input.normal_(0, std)