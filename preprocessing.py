''''loading and preprocessing the data for training'''
from tensorflow import keras

def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    return x_train, y_train, x_test, y_test