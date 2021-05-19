from torch import empty
import math

def generate_data(data_size):
    '''
    Generates 2D dataset within [0,1]^2 with label 1 for data samples
    within a disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi) and
    label 0 outside.

    Parameters
    ----------
    data_size: int
        length of dataset

    Returns
    -------
    data: FloatTensor
        data samples
    target: FloatTensor
        labels
    '''
    data = empty(data_size, 2).uniform_()
    target = ((data[:,0] - 0.5).pow(2) + (data[:,1]-0.5).pow(2)) <= 1/(2*math.pi)
    target = target * 1
    return data, target

def generate_batches(data_tuple, batch_size, one_hot = True):
    '''
    Generates batches of normalized data. Labels can be encoded
    one-hot or not.

    Parameters
    ----------
    data_tuple: Tuple(FloatTensor)
        Tuple with data samples and labels
    batch_size: int
        Size of batch
    one_hot: bool
        Indicates whether the labels should be one-hot encoded

    Returns
    -------
    data_batch: Tuple(FloatTensor)
        Tuple with data batches
    target_batch: Tuple(FloatTensor)
        Tuple with target batches
    '''
    data_batch = normalize_data(data_tuple[0]).split(batch_size)
    if one_hot:
        target_batch = labels_to_one_hot(data_tuple[1]).split(batch_size)
    else:
        target_batch = data_tuple[1][:,None].split(batch_size)

    return data_batch, target_batch

def normalize_data(data):
    '''
    Normalizes data.

    Parameters
    ----------
    data: FloatTensor
        data samples

    Returns
    -------
    : FloatTensor
        normalized data
    '''
    return (data - data.mean(0))/data.std(0)

def labels_to_one_hot(targets):
    '''
    Converts tensor with integers to one-hot encoding.

    Parameters
    ----------
    targets: FloatTensor
        tensor of labels

    Returns
    -------
    : FloatTensor
        tensor of one-hot encoded labels
    '''
    return empty(len(targets), targets.max() + 1).zero_().scatter_(1, targets[:,None], 1.)

