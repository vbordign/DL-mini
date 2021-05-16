from loss import *
from activation import *
from optimizer import *
from training_parameters import *
from generate_data import *
from train_eval import *

#%%
data_train, target_train = generate_batches(generate_data(train_size), batch_size = batch_size)
data_val, target_val = generate_batches(generate_data(val_size), batch_size = val_size)
data_test, target_test = generate_batches(generate_data(test_size), batch_size = test_size)

model = Sequential(
            Linear(2, 16, bias_flag=True),
            ReLU(),
            Linear(16, 32, bias_flag=True),
            ReLU(),
            Linear(32, 2, bias_flag=True),
)

criterion = CrossEntropy()
optimizer = Adam(model.param() , lr = learning_rate)

loss_train, loss_val, acc_val = train_model(model, data_train, target_train, data_val, target_val, optimizer, criterion, verbose=True)
loss_test, acc_test = eval_model(model, data_test, target_test, criterion)
print('Test:', loss_test, acc_test)



