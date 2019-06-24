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
    return median


def median_pool2d_output_shape(input_shape):
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
    x = x5
    x = Conv2D(n_init_features, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = PReLU(shared_axes=[1, 2])(x)
     
    for i in range(32):
        x = Conv2D(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x)
        x = Activation('relu')(x) 
        x = _residual_block(x, feature_dim=n_init_features)
        if i < 16:
            x = Lambda(median_pool2d, arguments={'k': 5}, output_shape=min_pool2d_output_shape)(x) 
    x = Conv2D(3, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
    model = Model(input=input_img, output=x)
    model.summary()
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

