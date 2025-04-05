import numpy as np
import pickle
import os

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        dict_ = pickle.load(f, encoding='bytes')
    return dict_[b'data'], dict_[b'labels']

def load_cifar10(file_folder):
    X_list, y_list = [], []

    for i in range(1, 6):  # data_batch_1 - data_batch_5
        file_train = os.path.join(file_folder, 'data_batch_' + str(i))
        X, y = load_cifar10_batch(file_train)
        X_list.append(X)
        y_list.append(y)

    # train dataset
    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    # test dataset
    file_test = os.path.join(file_folder, 'test_batch')
    X_test, y_test = load_cifar10_batch(file_test)

    # X standardization
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

# # example
# file_folder = f'./data/cifar-10-batches-py'
# X_train, y_train, X_test, y_test = load_cifar10(file_folder)
#
# print('X_train:', X_train,
#       '\ny_train:', y_train,
#       '\nX_test:', X_test,
#       '\ny_test:', y_test)
#
# print('length of X_train:', len(X_train),
#       '\nlength of y_train:', len(y_train),
#       '\nlength of X_test:', len(X_test),
#       '\nlength of y_test:', len(y_test))

