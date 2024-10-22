import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from tensorflow.keras.layers import GRU, BatchNormalization

import os
from dotenv import load_dotenv
load_dotenv()

Tx = int(os.getenv('Tx'))
n_freq = int(os.getenv('n_freq'))

def modelf(input_shape):
    
    X_input = Input(shape = input_shape)

    X = Conv1D(filters = 196, kernel_size = 15, strides = 4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)                                  

    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)                           
    
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)       
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)                                 
    
    X = TimeDistributed(Dense(1,activation = 'sigmoid'))(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

def prepare_model():
    model = modelf(input_shape = (Tx, n_freq))
    model.load_weights('model.h5')
    return model