from torch import empty
import math

def generate_data(data_size):
    data = empty(data_size, 2).uniform_()
    target = ((data[:,0] - 0.5).pow(2) + (data[:,1]-0.5).pow(2)) <= 1/(2*math.pi)
    target = target * 1
    return data, target

def generate_batches(data_tuple, batch_size):
    data_batch = normalize_data(data_tuple[0]).split(batch_size)
    target_batch = labels_to_one_hot(data_tuple[1]).split(batch_size)
    return data_batch, target_batch

def normalize_data(data):
    return (data - data.mean(0))/data.std(0)

def labels_to_one_hot(targets):
    return empty(len(targets), targets.max() + 1).zero_().scatter_(1, targets[:,None], 1.)

