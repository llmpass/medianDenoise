import os
import cv2
import sys
import math
import skimage
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
from skimage.measure import compare_psnr

from model import find_medians

'''
Generating metrics for evaluation.
Usage: 
    python metrics.py model_location src_img_dir
Args:
    model_location: str, '../pretrained/median5Res32Decrease-185.hdf5' as default
    src_img_dir: str, '../data/kodak/' as default
'''

# parse system arguments
model_location = '../pretrained/median5Res32Decrease-185.hdf5'
src_img_dir = '../data/kodak/'
if len(sys.argv) > 1:
    model_location = str(sys.argv[1])
if len(sys.argv) > 2:
    src_img_dir = str(sys.argv[2]) 
print('Running metrics on {} using model {}'.format(src_img_dir, model_location))

model = load_model(
        model_location, 
        custom_objects={
            'tf':tf, 
            'find_medians': find_medians,
            'merge': merge
            })
print('model loaded')

n_img = 0
sum_psnr3, sum_psnr5, sum_psnr7, sum_psnr9 = 0, 0, 0, 0
for file_name in os.listdir(src_img_dir):
    if file_name.endswith(".png"):
        n_img += 1
        print(file_name)
        fn = os.path.join(src_img_dir, file_name)
        src_img = cv2.imread(fn)
        img = np.asarray(src_img / 255.0, np.float)
        for j, a in enumerate([0.3, 0.5, 0.7, 0.9]):
            noisy_img = skimage.util.random_noise(img, mode='s&p', amount=a)
            gx0 = np.reshape(noisy_img, (1, *noisy_img.shape))
            Y = model.predict(gx0, verbose=0)
            result = np.asarray(Y[0,:,:,:], np.float)
            if j == 0:
                psnr3 = compare_psnr(img, result)
                print('psnr 30%', psnr3)
                sum_psnr3 += psnr3
                if psnr3 < 0:
                    cv2.imshow('bad', result)
                    cv2.waitKey(0)
            elif j == 1:
                psnr5 = compare_psnr(img, result)
                print('psnr 50%', psnr5)
                sum_psnr5 += psnr5
            elif j == 2:
                psnr7 = compare_psnr(img, result)
                print('psnr 70%', psnr7)
                sum_psnr7 += psnr7
            else:
                psnr9 = compare_psnr(img, result)
                print('psnr 90%', psnr9)
                sum_psnr9 += psnr9

print("--------------------------------------")
print("Avg:")
print('psnr 30%', sum_psnr3 / n_img)
print('psnr 50%', sum_psnr5 / n_img)
print('psnr 70%', sum_psnr7 / n_img)
print('psnr 90%', sum_psnr9 / n_img)

