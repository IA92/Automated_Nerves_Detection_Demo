#CNN Model
import numpy as np

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda,Reshape
from keras.layers import Input,LeakyReLU, Dense, Layer,LocallyConnected1D,Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras.layers import Conv1D,Conv2D, Conv3D, Conv2DTranspose,ConvLSTM2D, SimpleRNN, LSTM, Permute
from keras.layers import UpSampling2D, merge,Reshape
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras.engine import InputLayer
from keras.callbacks import ModelCheckpoint

#Initialise kernel for nearest neighbour interpolation
def kernel_init(shape):
    """Initialise kernel for nearest neighbour interpolation to be used in the deep learning model architecture  
    shape        : convolution layer filter size.
    """
    kernel = np.ones(shape)
    return kernel


def get_CNN(inputShape, chN=3, lr = 1e-4, loss='binary_crossentropy', metrics=['accuracy'], modelFile=""):
    """Getting CNN model architecture with predefined input shape   
    inputShape   : tupple of input shape [x,y].
    modelFile    : path to the pretrained model file [string]. If left empty, no model will be loaded.
    Returns      : keras model with loaded pretrained model.
    """ 
    inputs = Input((inputShape[0],inputShape[1],chN))
    inputsNormed=BatchNormalization()(inputs)
    conv1a = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(inputsNormed)
    conv1a = BatchNormalization()(conv1a)
    conv1a = Activation('relu')(conv1a)
    conv1b = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1a)
    conv1b = BatchNormalization()(conv1b)
    conv1b = Activation('relu')(conv1b)
    conv1Res1 = keras.layers.Concatenate()([conv1a,conv1b])
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1Res1)

    conv2a = Conv2D(32, 3,  padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2a = BatchNormalization()(conv2a)
    conv2a = Activation('relu')(conv2a)
    conv2b = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2a)
    conv2b = BatchNormalization()(conv2b)
    conv2b = Activation('relu')(conv2b)
    conv2Res1 = keras.layers.Concatenate()([conv2a,conv2b])
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2Res1)


    conv3a = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3a = BatchNormalization()(conv3a)
    conv3a = Activation('relu')(conv3a)
    conv3b = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3a)
    conv3b = BatchNormalization()(conv3b)
    conv3b = Activation('relu')(conv3b)
    conv3Res1 = keras.layers.Concatenate()([conv3a,conv3b])
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3Res1)

    conv4a = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4a = BatchNormalization()(conv4a)
    conv4a = Activation('relu')(conv4a)
    conv4b = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4a)
    conv4b = BatchNormalization()(conv4b)
    conv4b = Activation('relu')(conv4b)
    conv4Res1 = keras.layers.Concatenate()([conv4a,conv4b])
    drop4 = Dropout(0.5)(conv4Res1)
    pool4 = AveragePooling2D(pool_size=(2, 2))(drop4)

    conv5a = Conv2D(256, 3,  padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5a = BatchNormalization()(conv5a)
    conv5a = Activation('relu')(conv5a)
    conv5b = Conv2D(256, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv5a)
    conv5b = BatchNormalization()(conv5b)
    conv5b = Activation('relu')(conv5b)
    conv5Res1 = keras.layers.Concatenate()([conv5a,conv5b])
    drop5 = Dropout(0.5)(conv5Res1)

    fcConv1a = Conv2D(8, 1,  padding = 'same', kernel_initializer = 'he_normal')(drop5)
    fcConv1a = BatchNormalization()(fcConv1a)
    fcConv1a = Activation('relu')(fcConv1a)
    dropFcConv1a = Dropout(0.5)(fcConv1a)
    fc1a = Flatten()(dropFcConv1a)
    fc1b = Dense(256, activation='relu')(fc1a)
    dropFc1 = Dropout(0.5)(fc1b)
    fc1c = Dense(1, activation='sigmoid')(dropFc1)

    up6 = Conv2D(128, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = keras.layers.Concatenate()([conv4Res1,up6])
    conv6a = Conv2D(128, 3,  padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6a = BatchNormalization()(conv6a)
    conv6a = Activation('relu')(conv6a)
    conv6b = Conv2D(128, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv6a)
    conv6b = BatchNormalization()(conv6b)
    conv6b = Activation('relu')(conv6b)
    conv6Res1 = keras.layers.Concatenate()([conv6a,conv6b])

    up7 = Conv2D(64, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6Res1))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = keras.layers.Concatenate()([conv3Res1,up7])
    conv7a = Conv2D(64, 3,  padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7a = BatchNormalization()(conv7a)
    conv7a = Activation('relu')(conv7a)
    conv7b = Conv2D(64, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv7a)
    conv7b = BatchNormalization()(conv7b)
    conv7b = Activation('relu')(conv7b)
    conv7Res1 = keras.layers.Concatenate()([conv7a,conv7b])

    up8 = Conv2D(32, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7Res1))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    merge8 = keras.layers.Concatenate()([conv2Res1,up8])
    conv8a = Conv2D(32, 3,  padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8a = BatchNormalization()(conv8a)
    conv8a = Activation('relu')(conv8a)
    conv8b = Conv2D(32, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv8a)
    conv8b = BatchNormalization()(conv8b)
    conv8b = Activation('relu')(conv8b)
    conv8Res1 = keras.layers.Concatenate()([conv8a,conv8b])

    up9 = Conv2D(16, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8Res1))
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)
    merge9 = keras.layers.Concatenate()([conv1Res1,up9])
    conv9a = Conv2D(16, 3,  padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9a = BatchNormalization()(conv9a)
    conv9a = Activation('relu')(conv9a)
    conv9b = Conv2D(16, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv9a)
    conv9b = BatchNormalization()(conv9b)
    conv9b = Activation('relu')(conv9b)
    conv9Res1 = keras.layers.Concatenate()([conv9a,conv9b])
    conv9 = Conv2D(2, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv9Res1)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('sigmoid')(conv10)


    fc1cFR = Conv2DTranspose(1,inputShape,padding=('valid'),kernel_initializer=kernel_init,trainable=False)(Reshape((1,1,1))(fc1c))
    finalLayer = keras.layers.Multiply()([conv10,fc1cFR])

    model = Model(input = inputs, output = [finalLayer,fc1c])
    model.compile(optimizer = Adam(lr = 1e-4), loss = loss, metrics = metrics)
    if modelFile!="":
        try:
            print("Model loaded: "+str(modelFile))
        except NameError:
            print("Can't find model, no model loaded")
    return model
