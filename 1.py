from operator import ne

import scipy.io as scio
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

path = 'data_batch_1.mat'
data = scio.loadmat(path)
data_train = data['data']
data_train = data_train.reshape((10000,32,32,3))
data_train = data_train.astype('float32') / 255
data_label = data['labels']
data_label = to_categorical(data_label)


network = models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(32,32,3)))
network.add(layers.MaxPool2D((2,2)))
network.add(layers.Flatten())
network.add(layers.Dense(32*32, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer=optimizers.RMSprop(learning_rate =1e-4), loss='binary_crossentropy', metrics=['acc'])
network.fit(data_train, data_label, epochs=15, batch_size = 128)
network.summary()
