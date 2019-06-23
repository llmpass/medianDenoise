import os
import numpy as np 
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

n_init_features = 64


def min_pool2d(x):
    min_x = -K.pool2d(-x, pool_size=(2, 2), strides=(1, 1), padding='same')
    return min_x

def find_medians(x, k=3):
    patches = tf.extract_image_patches(
            x, 
            ksizes=[1, k, k, 1],
            strides = [1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
    m_idx = int(k*k/2 + 1)
    top, _ = tf.nn.top_k(patches, m_idx, sorted=True)
    median = tf.slice(top, [0, 0, 0, m_idx-1], [-1, -1, -1, 1])
    return median

def median_pool2d(x, k=3):
    channels = tf.split(x, num_or_size_splits=x.shape[3], axis=3)
    for channel in channels:
        channel = find_medians(channel, k)
    median = merge(channels, mode='concat', concat_axis=-1)
    '''
    r_median = find_medians(r, k)
    g_median = find_medians(g, k)
    b_median = find_medians(b, k)
    median = merge([r_median, g_median, b_median], mode='concat', concat_axis=-1)
    '''
    return median

def min_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)


def fully_conv(pretrained_weights=None, input_size=(None, None, 3)):
    def _residual_block(inputs, feature_dim=64):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])
        return m

    input_img = Input(shape=input_size, name='input_image')
    x5 = Lambda(median_pool2d, arguments={'k': 5}, 
            output_shape=min_pool2d_output_shape)(input_img) 
    x5 = Lambda(median_pool2d, arguments={'k': 5}, 
            output_shape=min_pool2d_output_shape)(x5) 
    '''
    for i in range(0):
        x5 = Conv2D(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x5)
        x5 = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x5)
        x5 = Activation('relu')(x5)  
        x6 = Conv2D(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x6)
        x6 = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x6)
        x6 = Activation('relu')(x6)  
        x7 = Conv2D(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x7)
        x7 = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x7)
        x7 = Activation('relu')(x7)  
        '''
    x = x5 # merge([x7, x9], mode='concat', concat_axis=-1)
    x = Conv2D(n_init_features, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = PReLU(shared_axes=[1, 2])(x)
     
    for i in range(16):
        x = Conv2D(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x)
        x = Activation('relu')(x) 
        x = _residual_block(x, feature_dim=n_init_features)
        if i < 4:
            x = Lambda(median_pool2d, arguments={'k': 5}, output_shape=min_pool2d_output_shape)(x) 
    '''
    for i in range(8):
        x = Conv2DTranspose(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x)
        x = Activation('relu')(x) 
    '''
    x = Conv2D(3, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
    model = Model(input=input_img, output=x)
    # model = Model(input=[input_img, input_loc], output=x)
    model.summary()
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def unet2d(pretrained_weights = None, input_size = (None, None, 3)):
    input_img = Input(shape=input_size)
    x = Lambda(median_pool2d, output_shape=min_pool2d_output_shape)(input_img) 
    '''
    x = Conv2D(n_init_features, (3, 3), padding='same')(x)
    down1 = Conv2D(n_init_features, (3, 3), padding='same', activation='relu', strides=2)(x)
    x = Conv2D(n_init_features*2, (3, 3), activation='relu', padding='same')(down1)
    down2 = Conv2D(n_init_features*2, (3, 3), padding='same', activation='relu', strides=2)(x)
    x = Conv2D(n_init_features*2, (3, 3), activation='relu', padding='same')(down2)
    down3 = Conv2D(n_init_features*2, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(n_init_features*4, (3, 3), activation='relu', padding='same')(down3)
    down4 = Conv2D(n_init_features*4, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(n_init_features*4, (3, 3), activation='relu', padding='same')(down4)
    encoded = Conv2D(n_init_features*4, (2, 2), padding='same', activation='relu', strides=2)(x)
    
    up1 = Conv2DTranspose(n_init_features*4, (3, 3), activation='relu', strides=2, padding='same')(encoded)
    merge1 = merge([down4, up1], mode = 'concat', concat_axis = -1)
    up2 = Conv2DTranspose(n_init_features*2, (3, 3), activation='relu', strides=2, padding='same')(merge1)
    merge2 = merge([down3, up2], mode = 'concat', concat_axis = -1)
    up3 = Conv2DTranspose(n_init_features*2, (3, 3), activation='relu', strides=2, padding='same')(merge2)
    merge3 = merge([down2, up3], mode = 'concat', concat_axis = -1)
    up4 = Conv2DTranspose(n_init_features, (3, 3), activation='relu', strides=2, padding='same')(merge3)
    merge4 = merge([down1, up4], mode = 'concat', concat_axis = -1)
    x = Conv2D(n_init_features, (3, 3), activation='relu', padding='same')(merge4)
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', strides=2, padding='same')(x)
    '''
    model = Model(input_img, x)
    model.summary()
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



