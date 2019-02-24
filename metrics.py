
from keras.callbacks import BaseLogger
from json import JSONEncoder

class FloydhubKerasCallback(BaseLogger):
    def __init__(self, mode='epoch', stateful_metrics=None):
        super().__init__(stateful_metrics)

        if mode not in ('epoch', 'batch'):
            raise ValueError('Mode parameter should be "epoch" or "batch"')
        self.mode = mode
        self.encoder = JSONEncoder()

    def report(self, metric, value, **kwargs):
        info = {'metric': metric, 'value': value}
        info.update(kwargs)
        print(self.encoder.encode(info))

    def on_batch_end(self, batch, logs):
        if not self.mode == 'batch':
            return
        for metric in frozenset(logs.keys()) - frozenset(['batch', 'size']):
            self.report(metric, logs[metric].item(), step=batch)

    def on_epoch_end(self, epoch, logs):
        if not self.mode == 'epoch':
            return
        for metric in frozenset(logs.keys()):
            self.report(metric, logs[metric].item(), step=epoch)



# Execute this code to test FloydhubKerasCallback
# A simple deep learning model will be created to classify mnist dataset examples

import numpy as np
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ProgbarLogger
import keras.backend as K

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape([X_train.shape[0], 28, 28, 1])
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

K.clear_session()
model = Sequential()
model.add(Conv2D(24, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu' ))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# initial_weights = model.get_weights()

callbacks = [
    EarlyStopping('loss', min_delta=0.01, patience=4),
    FloydhubKerasCallback(mode='batch')
]
model.fit(X_train, y_train, verbose=False, batch_size=32, epochs=1,
          callbacks=callbacks)
