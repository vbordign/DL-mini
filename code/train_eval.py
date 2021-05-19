from loss import *
from activation import *
from optimizer import *
from training_parameters import *
from generate_data import *

def train_model(model, data_train, target_train, data_val,
                target_val, optimizer, criterion, num_epochs = 100, verbose = False, one_hot = True):
    ''' Trains a model using an optimization criterion for a certain number of epochs.
    Returns validation and training performance indicators over epochs.

    Parameters
    ----------
    model: modules.Module
        model to be trained
    data_train: Tuple(FloatTensor)
        Tuple with training data batches
    target_train: Tuple(FloatTensor)
        Tuple with training label batches
    data_val: Tuple(FloatTensor)
        Tuple with validation data batches
    target_val: Tuple(FloatTensor)
        Tuple with validation label batches
    optimizer: SGD, Adam, Adagrad, Adadelta or RMSprop
        Optimizer
    criterion: LossMSE or CrossEntropy criterion
        Criterion
    num_epochs: int
        Number of training epochs
    verbose: bool
        Flag that indicates logging
    one_hot: bool
        Flag that indicates one-hot encoding
    Returns
    -------
    loss_list: list(FloatTensor)
        training loss over epochs
    loss_val_list: list(FloatTensor)
        validation loss over epochs
    acc_val_list: list(FloatTensor)
        validation accuracy over epochs
    '''
    loss_list, loss_val_list, acc_val_list = [], [], []
    for i in range(num_epochs):
        loss_batch = 0
        total = 0
        for j, batch in enumerate(data_train):
            output = model.forward(batch)
            loss, gradwrtoutput = criterion.loss(target_train[j], output)
            model.zero_grad()
            model.backward(gradwrtoutput)
            optimizer.step()
            loss_batch += loss
            total += batch.size(0)
        loss_val, acc_val = eval_model(model, data_val, target_val, criterion, one_hot = one_hot)
        loss_list.append(loss_batch/len(data_train))
        loss_val_list.append(loss_val)
        acc_val_list.append(acc_val)
        if verbose:
            print(f'Epoch {i}: Train Loss: {(loss_batch/len(data_train)).item():.4f}, '
                  f'Val. Loss: {loss_val.item():.4f}, Val Accuracy: {acc_val.item():.2f} %')
    return loss_list, loss_val_list, acc_val_list

def eval_model(model, data_val, target_val, criterion, one_hot = True):
    '''
    Evaluate the model on the validation dataset.

    Parameters
    ----------
    model: modules.Module
        model to be evaluated
    data_val: Tuple(FloatTensor)
        Tuple with validation data batches
    target_val: Tuple(FloatTensor)
        Tuple with validation label batches
    criterion: LossMSE or CrossEntropy criterion
        Criterion
    one_hot: bool
        Flag that indicates one-hot encoding

    Returns
    -------
    : FloatTensor
        validation loss
    acc: FloatTensor
        validation accuracy
    '''
    loss_list = []
    loss_batch, correct, total = 0, 0, 0
    for j, batch in enumerate(data_val):
        output = model.forward(batch)
        loss, _ = criterion.loss(target_val[j], output)
        if one_hot:
            _, predicted = output.max(1)
            correct += (predicted == target_val[j].max(1)[1]).sum()
        else:
            predicted = (output > 0.5)*1.
            correct += (predicted == target_val[j]).sum()
        loss_batch += loss
        total += batch.size(0)
    acc = 100 * correct/ total
    return loss_batch/len(data_val), acc
