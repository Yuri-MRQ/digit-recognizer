#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Input, Dense, MaxPool2D,
                                     BatchNormalization, Activation, 
                                     Add, ZeroPadding2D, Flatten, AveragePooling2D)
from tensorflow.keras.models import Model


# In[2]:


def plain_network(X, filters):
    
#     we have to save x
    x_shortcut = X
    
#     first block
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(X)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
#     second bloack
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
                   
#     shortcut
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
                   
    return x
    
    


# In[3]:


def residual(X, filters):
    
    x_shortcut = X
    
#     first block
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(2, 2))(X)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
#     second block
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
#     shortcut
    x_shortcut = Conv2D(filters=filters, kernel_size=(1,1), padding='valid', strides=(2, 2))(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)
    
#     add

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x


# In[5]:


def resnet18(input_shape):
    
    input_in = Input(shape=(input_shape))
    x = ZeroPadding2D(padding=(3,3))(input_in)
    
#   1st stage
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    
#   2st stage
    x = plain_network(x, 64)
    x = plain_network(x, 64)
    
#   3st stage
    x = residual(x, 128)
    x = plain_network(x, 128)
    
#   4st stage
    x = residual(x, 256)
    x = plain_network(x, 256)
    
#   5st stage
    x = residual(x, 512)
    x = plain_network(x, 512)
    
#   6st stage

    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs = input_in, outputs = x, name='resnet18')
    
    return model

# In[ ]:




