from loss import *
from activation import *
from optimizer import *
from training_parameters import *
from generate_data import *
from train_eval import *

# %%
if __name__ == "__main__":

    print(f'### Training Model: Loss = Cross Entropy, Optimizer = SGD, Initialization = Uniform ###')
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
    optimizer = SGD(model.param() , lr = learning_rate)

    loss_train, loss_val, acc_val = train_model(
        model, data_train, target_train, data_val, target_val, optimizer, criterion, verbose = True)

    print(f'### Testing Trained Model ###')
    loss_test, acc_test = eval_model(model, data_test, target_test, criterion)

    print(f'Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test.item():.2f}')

    #%%
    print(f'### Training Model: Loss = MSE, Optimizer = SGD, Initialization = Uniform ###')
    data_train, target_train = generate_batches(generate_data(train_size), batch_size = batch_size, one_hot = False)
    data_val, target_val = generate_batches(generate_data(val_size), batch_size = val_size, one_hot = False)
    data_test, target_test = generate_batches(generate_data(test_size), batch_size = test_size, one_hot = False)

    model = Sequential(
                Linear(2, 16, bias_flag=True),
                ReLU(),
                Linear(16, 32, bias_flag=True),
                ReLU(),
                Linear(32, 1, bias_flag=True),
                Sigmoid()
    )

    criterion = LossMSE()
    optimizer = SGD(model.param() , lr = learning_rate)

    loss_train, loss_val, acc_val = train_model(
        model, data_train, target_train, data_val, target_val, optimizer, criterion, verbose = True, one_hot = False)

    print(f'### Testing Trained Model ###')
    loss_test, acc_test = eval_model(model, data_test, target_test, criterion, one_hot = False)

    print(f'Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test.item():.2f}')
