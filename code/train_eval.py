from loss import *
from activation import *
from optimizer import *
from training_parameters import *
from generate_data import *

def train_model(model, data_train, target_train, data_val, target_val, optimizer, criterion, num_epochs = 100, verbose = False):
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
        loss_val, acc_val = eval_model(model, data_val, target_val, criterion)
        loss_list.append(loss_batch/len(data_train))
        loss_val_list.append(loss_val)
        acc_val_list.append(acc_val)
        if verbose:
            print(f'Epoch {i}: Train Loss: {(loss_batch/len(data_train)).item():.4f}, Val Loss: {loss_val.item():.4f}, Val Acc: {acc_val.item():.2f} %')
    return loss_list, loss_val_list, acc_val_list

def eval_model(model, data_val, target_val, criterion):
    loss_list = []
    loss_batch, correct, total = 0, 0, 0
    for j, batch in enumerate(data_val):
        output = model.forward(batch)
        loss, _ = criterion.loss(target_val[j], output)
        _, predicted = output.max(1)
        correct += (predicted == target_val[j].max(1)[1]).sum()
        loss_batch += loss
        total += batch.size(0)
    acc = 100 * correct/ total
    return loss_batch/len(data_val), acc
