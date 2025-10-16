import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

def load_cifar10_batch(batch_filename):
    with open(batch_filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_cifar10_data(data_dir):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(data_dir, 'data_batch_%d' % b)
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def prepare_data(data_dir):
    x_train, y_train, x_test, y_test = load_cifar10_data(data_dir)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    cifar10_dir = '/Users/Yoshua/cnn-research/cifar-10-batches-py'
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(cifar10_dir)
    print(f'Training data shape: {x_train.shape}, Labels shape: {y_train.shape}')
    print(f'Validation data shape: {x_val.shape}, Labels shape: {y_val.shape}')
    print(f'Test data shape: {x_test.shape}, Labels shape: {y_test.shape}')
