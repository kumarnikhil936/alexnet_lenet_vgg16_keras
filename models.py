# Import the Sequential model type from Keras. This is simply a linear stack of neural network layers.
from keras.models import Sequential

# Import the "core" layers from Keras. These are the layers that are used in almost any neural network.
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape

from keras.layers import ZeroPadding2D

# Import the CNN layers from Keras. These are the convolutional layers that will help us efficiently train on image data
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from keras.constraints import maxnorm

# Import some utilities. This will help us transform our data later
# from keras.utils import np_utils

from keras.layers.normalization import BatchNormalization

from keras import backend as K
K.set_image_data_format('channels_last')  # WARNING : important for images and tensors dimensions ordering

import time 

"""
Models
"""


def simple_model(input_width=32,
          input_height=32,
          feature_maps=32,
          feature_window_size=(5, 5),
          dropout1=0.2,
          dense=128,
          dropout2=0.5,
          use_max_pooling=True,
          pool_size=(2, 2),
          optimizer='rmsprop'):
    """
    Model Architecture:
    
        Convolution
        Convolution
        Pool
        Dropout
        Flatten
        Dense
        Dropout
        Dense

    Input arguments:

        input_width/input_height are size of image, obviously

        feature_maps/feature_window_size determine the number and size of convolution

        dropout1 is number of nodes after pooling layer to set to 0

        dropout2 is number of nodes after flatten/dense layer to set to 0

        dense is number of nodes in dense layer

        use_max_pooling is a boolean; if yes, use MaxPooling, else use AveragePooling

        optimizer is one of 'rmsprop' or 'adam'
    """

    modelA = Sequential()

    modelA.add(ZeroPadding2D(padding=(3, 3), input_shape=(input_width, input_height, 6)))

    # Convolutional input layer:
    # - 20 feature maps (each feature map is a reduced-size convolution that detects a different feature)
    # - 3 pixel square window
    modelA.add(Conv2D(feature_maps,
                      feature_window_size,
                      # input_shape=(input_width, input_height, 6),
                      padding='same',
                      data_format='channels_last'))

    modelA.add(BatchNormalization(axis=3, epsilon=0.00001))

    modelA.add(Activation('relu'))

    modelA.add(ZeroPadding2D(padding=(1, 1)))

    # Pooling layer
    if use_max_pooling:
        modelA.add(MaxPooling2D(pool_size=pool_size, strides=2,
                                data_format='channels_last'))
    else:
        modelA.add(AveragePooling2D(pool_size=pool_size, strides=2,
                                    data_format='channels_last'))

    # Second convolutional layer
    # - 40 feature maps (add more features)
    # - 3 pixel square window
    modelA.add(Conv2D(feature_maps,
                      feature_window_size,
                      padding='same',
                      data_format='channels_last',
                      activation='relu'))

    # Pooling layer
    if (use_max_pooling):
        modelA.add(MaxPooling2D(pool_size=pool_size,
                                data_format='channels_last'))
    else:
        modelA.add(AveragePooling2D(pool_size=pool_size,
                                    data_format='channels_last'))
    modelA.add(ZeroPadding2D(padding=(3, 3), input_shape=(input_width, input_height, 6)))

    # Convolutional input layer:
    # - 20 feature maps (each feature map is a reduced-size convolution that detects a different feature)
    # - 3 pixel square window
    modelA.add(Conv2D(feature_maps,
                      feature_window_size,
                      padding='same',
                      data_format='channels_last'))

    modelA.add(BatchNormalization(axis=3, epsilon=0.00001))

    modelA.add(Activation('relu'))

    modelA.add(ZeroPadding2D(padding=(1, 1)))

    # Pooling layer
    if use_max_pooling:
        modelA.add(MaxPooling2D(pool_size=pool_size, strides=2,
                                data_format='channels_last'))
    else:
        modelA.add(AveragePooling2D(pool_size=pool_size, strides=2,
                                    data_format='channels_last'))

    # Second convolutional layer
    # - 40 feature maps (add more features)
    # - 3 pixel square window
    modelA.add(Conv2D(feature_maps,
                      feature_window_size,
                      padding='same',
                      data_format='channels_last',
                      activation='relu'))

    # Pooling layer
    if (use_max_pooling):
        modelA.add(MaxPooling2D(pool_size=pool_size,
                                data_format='channels_last'))
    else:
        modelA.add(AveragePooling2D(pool_size=pool_size,
                                    data_format='channels_last'))

    # Set X% of units to 0
    modelA.add(Dropout(dropout1))

    # Flatten layer
    modelA.add(Flatten())

    # Fully connected layer with 128 units and a rectifier activation function.
    modelA.add(Dense(dense,
                     activation='relu',
                     kernel_constraint=maxnorm(3)))

    # Dropout 
    modelA.add(Dropout(dropout2))

    # Fully connected output layer with 2 units (Y/N)
    # and a softmax activation function.
    modelA.add(Dense(1, activation='sigmoid'))

    if (optimizer not in ['rmsprop', 'adam', 'adadelta']):
        optimizer = 'rmsprop'
    
    start = time.time()

    modelA.compile(loss='binary_crossentropy',
                   metrics=['binary_accuracy'],
                   optimizer='rmsprop')

    duration = time.time() - start
    print("Time (seconds) to make the Simple model: ", duration)

    return modelA


def alexnet(input_width=32, input_height=32):
    
    model = Sequential()
    
    model.add(ZeroPadding2D((3,3),input_shape=(input_width, input_height, 6), data_format='channels_last'))
    model.add(Conv2D(64, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(64, (3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))
    
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(128, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(128, (3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))
    
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))
    
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(512, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(512, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(512, (3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))
    
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(512, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(512, (3, 3), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1), data_format='channels_last'))
    model.add(Conv2D(512, (3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))
    
    model.add(Flatten(data_format='channels_last'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    
    # Fully connected output layer with 2 units (Y/N)
    # and a softmax activation function.
    model.add(Dense(1, activation='softmax'))

    start = time.time()
    
    model.compile(loss='binary_crossentropy',
                   metrics=['binary_accuracy'],
                   optimizer='adam')

    duration = time.time() - start
    print("Time (seconds) to make the Alexnet model: ", duration)

    return model



def lenet(input_width=32,
          input_height=32):
    
    model = Sequential()
    
    model.add(Conv2D(20, (5,5), input_shape=(input_width, input_height, 6), padding='same', data_format='channels_last', activation='relu', name='conv2d_1'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last', strides=(2,2), name='max_pool_1'))
    
    model.add(Conv2D(50, (5,5), padding='same', data_format='channels_last', activation='relu', name='conv2d_2'))	
    model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last', strides=(2,2), name='max_pool_2'))										
    
    model.add(Flatten(name='flat', data_format='channels_last'))
    model.add(Dense(500, activation='relu'))
    
    # Fully connected output layer with 2 units (Y/N)
    # and a softmax activation function.
    model.add(Dense(1, activation='sigmoid'))

    start = time.time()

    model.compile(loss='binary_crossentropy',
                   metrics=['binary_accuracy'],
                   optimizer='rmsprop')

    duration = time.time() - start
    print("Time (seconds) to make the Lenet model: ", duration)

    return model


def convblock(cdim, nb, bits=3):
    L = []
    for k in range(1, bits + 1):
        convname = 'conv' + str(nb) + '_' + str(k)
        L.append(Conv2D(cdim, (3, 3), padding='same', activation='relu', name=convname, data_format='channels_last'))
    L.append(MaxPooling2D((2, 2), strides=(2, 2)))
    return L


def vgg(input_width=32, input_height=32):
    withDropOut = True

    model = Sequential()

    model.add(Permute((1, 2, 3), input_shape=(input_width, input_height, 6)))

    for l in convblock(64, 1, bits=2):
        model.add(l)

    for l in convblock(128, 2, bits=2):
        model.add(l)

    for l in convblock(256, 3, bits=3):
        model.add(l)

    for l in convblock(512, 4, bits=3):
        model.add(l)

    for l in convblock(512, 5, bits=3):
        model.add(l)

    model.add(Conv2D(4096, (7, 7), padding='same', activation='relu', name='fc6'))
    if withDropOut:
        model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), padding='same', activation='relu', name='fc7'))
    if withDropOut:
        model.add(Dropout(0.5))
    model.add(Conv2D(2622, (1, 1), padding='same', activation='relu', name='fc8'))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))

    # Fully connected output layer with 2 units (Y/N)
    # and a softmax activation function.
    model.add(Dense(1, activation='sigmoid'))

    start = time.time()

    model.compile(loss='binary_crossentropy',
                   metrics=['binary_accuracy'],
                   optimizer='rmsprop')

    duration = time.time() - start
    print("Time (seconds) to make the VGG model: ", duration)

    return model
