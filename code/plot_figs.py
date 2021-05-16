from loss import *
from activation import *
from optimizer import *
from training_parameters import *
from generate_data import *
from train_eval import *
import torch
import os
if not os.path.exists('./stats/'):
    os.makedirs('./stats/')
if not os.path.exists('./figs/'):
    os.makedirs('./figs/')


torch.manual_seed(1)
data_train, target_train = generate_batches(generate_data(train_size), batch_size = batch_size)
data_val, target_val = generate_batches(generate_data(val_size), batch_size = val_size)
data_test, target_test = generate_batches(generate_data(test_size), batch_size = test_size)


loss_tr_repeat, loss_val_repeat, acc_val_repeat, loss_test_repeat, acc_test_repeat = [], [], [], [], []
for i in range(num_repeat):
    model = Sequential(
                Linear(2, 16, bias_flag=True, init = 'uniform'),
                ReLU(),
                Linear(16, 32, bias_flag=True, init='uniform'),
                ReLU(),
                Linear(32, 2, bias_flag=True, init = 'uniform'),
    )

    criterion = CrossEntropy()
    lr = 0.001
    optimizer = SGD(model.param())

    loss_train, loss_val, acc_val = train_model(model, data_train, target_train, data_val, target_val, optimizer, criterion, verbose=False)
    loss_test, acc_test = eval_model(model, data_test, target_test, criterion)
    print('Test:', loss_test, acc_test)
    loss_tr_repeat.append(loss_train)
    loss_val_repeat.append(loss_val)
    acc_val_repeat.append(acc_val)
    loss_test_repeat.append(loss_test)
    acc_test_repeat.append(acc_test)

torch.save((loss_tr_repeat, loss_val_repeat, acc_val_repeat, acc_test, loss_test), f'./stats/stats_sgd.pkl' )
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'

if not os.path.exists('./figs/'):
    os.makedirs('./figs/')

loss_train, loss_val, acc_val, acc_test, loss_test=torch.load('stats/stats_sgd.pkl')
loss_train1, loss_val1, acc_val1, acc_test1, loss_test1=torch.load('stats/stats_adam.pkl')
loss_train2, loss_val2, acc_val2, acc_test2, loss_test2=torch.load('stats/stats_adagrad.pkl')
loss_train3, loss_val3, acc_val3, acc_test3, loss_test3=torch.load('stats/stats_rmsprop.pkl')
loss_train4, loss_val4, acc_val4, acc_test4, loss_test4=torch.load('stats/stats_adadelta.pkl')

f, a = plt.subplots(3,1,figsize=(5,6))
a[0].plot(torch.arange(1,num_epochs+1, 9),torch.tensor(loss_train).mean(0)[::9], color='C0')
a[0].plot(torch.arange(1,num_epochs+1, 9),torch.tensor(loss_train1).mean(0)[::9], color='C0', marker ='.', linewidth=1)
a[0].plot(torch.arange(1,num_epochs+1, 9),torch.tensor(loss_train2).mean(0)[::9], color='C0', marker ='x', linewidth=1)
a[0].plot(torch.arange(1,num_epochs+1, 9),torch.tensor(loss_train3).mean(0)[::9], color='C0', marker ='^', linewidth=1, markerfacecolor='none')
a[0].plot(torch.arange(1,num_epochs+1, 9),torch.tensor(loss_train4).mean(0)[::9], color='C0', marker ='s', linewidth=1, markerfacecolor='none')

a[0].set_ylabel('Training Loss', fontsize=16, labelpad=15)
a[0].set_xlabel('Epoch', fontsize=16)
a[0].set_xlim([0,num_epochs])
a[0].legend(['SGD', 'Adam', 'Adagrad', 'RMSProp', 'Adadelta'], ncol=2)
a[1].set_ylabel('Validation Loss', fontsize=16, labelpad=15)
a[1].set_xlabel('Epoch', fontsize=16)
a[1].plot(torch.arange(1,num_epochs+1,9),torch.tensor(loss_val).mean(0)[::9], color='C2', linewidth=1)
a[1].plot(torch.arange(1,num_epochs+1, 9),torch.tensor(loss_val1).mean(0)[::9], color='C2', marker ='.', linewidth=1)
a[1].plot(torch.arange(1,num_epochs+1,9),torch.tensor(loss_val2).mean(0)[::9], color='C2', marker ='x', linewidth=1)
a[1].plot(torch.arange(1,num_epochs+1,9),torch.tensor(loss_val3).mean(0)[::9], color='C2', marker ='^', linewidth=1,markerfacecolor='none')
a[1].plot(torch.arange(1,num_epochs+1,9),torch.tensor(loss_val4).mean(0)[::9], color='C2', marker ='s', linewidth=1,markerfacecolor='none')

a[1].set_xlim([0,num_epochs])


a[2].plot(torch.arange(1,num_epochs+1,9),torch.tensor(acc_val).mean(0)[::9], color='k', linewidth=1)
a[2].plot(torch.arange(1,num_epochs+1,9),torch.tensor(acc_val1).mean(0)[::9], color='k', marker ='.', linewidth=1)
a[2].plot(torch.arange(1,num_epochs+1,9),torch.tensor(acc_val2).mean(0)[::9], color='k', marker ='x', linewidth=1)
a[2].plot(torch.arange(1,num_epochs+1,9),torch.tensor(acc_val3).mean(0)[::9], color='k', marker ='^', linewidth=1,markerfacecolor='none')
a[2].plot(torch.arange(1,num_epochs+1,9),torch.tensor(acc_val4).mean(0)[::9], color='k', marker ='s', linewidth=1,markerfacecolor='none')

a[2].set_ylabel('Val. Accuracy (\%)', fontsize=16)
a[2].set_xlabel('Epoch', fontsize=16)
a[2].set_xlim([0,num_epochs])

a[2].set_ylim([80,100])

f.tight_layout()
# f.savefig('./figs/ce_opt.pdf')
plt.show()
