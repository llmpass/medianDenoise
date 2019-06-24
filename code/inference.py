import cv2
import sys
import math
import skimage
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
from skimage.measure import compare_psnr

from model import *

'''
Simply running inference on 1 image.
Usage: 
    python inference.py model_location img_location
Args:
    model_location: str, '../pretrained/median5Res32Decrease-185.hdf5' as default
    img_location: str, '../data/BSD300/' as default
'''

# parse system arguments
model_location = '../pretrained/median5Res32Decrease-185.hdf5'
img_location = '../data/BSD300/102061.jpg'
if len(sys.argv) > 1:
    model_location = str(sys.argv[1])
if len(sys.argv) > 2:
    img_location = str(sys.argv[2]) 
print('Inference on {} using model {}'.format(img_location, model_location))


model = load_model(
        model_location,
        custom_objects={
            'tf':tf, 
            'find_medians': find_medians,
            'merge': merge
            })


src_img = cv2.imread(img_location)
img = np.asarray(src_img / 255.0, np.float)
noisy_img = skimage.util.random_noise(img, mode='s&p', amount=0.7)
gx0 = np.reshape(noisy_img, (1, *noisy_img.shape))
Y = model.predict(gx0, verbose=1)
result = np.asarray(Y[0,:,:,:], np.float)
cv2.imshow('original', src_img)
cv2.imshow('bef', noisy_img)
cv2.imshow('aft', result)
print('psnr original', compare_psnr(img, noisy_img))
print('psnr smoothed', compare_psnr(img, result))
cv2.waitKey(0)


