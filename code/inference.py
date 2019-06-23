import cv2
import math
import skimage
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
from skimage.measure import compare_psnr

from model import *

model = load_model(
        # 'pretrained/median5Res32Decrease/unet_2d-185.hdf5', 
        'checkpoints/fully_convM5All-151.h5',
        custom_objects={
            'tf':tf, 
            'find_medians': find_medians,
            'merge': merge
            })

# model = morph_resnet(input_size=(None, None, 3))
# model.load_weights('checkpoints/morph_resnet-151.h5')

src_img = cv2.imread('pic/bad_result/104/102061.png')
img9 = cv2.imread('pic/bad_result/104/104.png')
src_img = np.asarray(src_img / 255.0, np.float)
img9 = np.asarray(img9 / 255.0, np.float)
n2n_img = cv2.imread('pic/bad_result/104/n2n_104.png')
n2n_img = np.asarray(n2n_img / 255.0, np.float)
'''
failure case
src_img = np.zeros((300, 300, 3))
for i in range(0, 12, 2):
    src_img[:, i*20:(i+1)*20, 1] = np.ones((300, 20))
img5 = skimage.util.random_noise(src_img, 's&p', amount=0.1)
'''
#cv2.imwrite('noise5.png', img5*255)
# img5 = cv2.imread('results/156079/noise5.png')
# img5 = cv2.imread('DSC_0355.jpg')
# img5 = np.asarray(img5 / 255.0, np.float)
#img5 = skimage.util.random_noise(img5, 's&p', amount=0.5)
#median5 = cv2.medianBlur(img5, 5)
#median5X2 = np.asarray(cv2.medianBlur(median5, 5)/255.0, np.float)
'''
img3 = cv2.imread('results/lenna/noise3.png')
img3 = np.asarray(img3 / 255.0, np.float)
img5 = cv2.imread('results/lenna/noise5.png')
img5 = np.asarray(img5 / 255.0, np.float)
img7 = cv2.imread('results/lenna/noise7.png')
img7 = np.asarray(img7 / 255.0, np.float)
# cv2.imwrite('noise3.png', img*255)
'''
src_img = cv2.imread('pic/Set14/lenna.bmp')
src_img = np.asarray(src_img / 255.0, np.float)
img3 = cv2.imread('results/lenna/noise7.png')
img3 = np.asarray(img3 / 255.0, np.float)
gx0 = np.reshape(img3, (1, *img3.shape))
Y = model.predict(gx0, verbose=1)
result = np.asarray(Y[0,:,:,:], np.float)
cv2.imshow('bef', img3)
cv2.imshow('aft', result)
#cv2.imshow('median5X2', median5X2)
print('psnr original', compare_psnr(src_img, img3))
print('psnr smoothed', compare_psnr(src_img, result))
# print('psnr n2n', compare_psnr(src_img, n2n_img))
# cv2.imwrite('result3.png', result*255)
cv2.imshow('original', src_img)
#dif = np.abs(src_img-result)
#cv2.imshow('original-aft', dif)
# cv2.imwrite('result9.png', result*255)
#cv2.imwrite('dif5.png', dif*255)
'''
median3 = np.copy(img3)
median5 = np.copy(img5)
median7 = np.copy(img7)
psnr3 = []
psnr5 = []
psnr7 = []
for i in range(30):
    median3 = np.asarray(median3*255, np.uint8)
    median5 = np.asarray(median5*255, np.uint8)
    median7 = np.asarray(median7*255, np.uint8)
    median3 = np.asarray(cv2.medianBlur(median3, 5)/255.0, np.float)
    median5 = np.asarray(cv2.medianBlur(median5, 5)/255.0, np.float)
    median7 = np.asarray(cv2.medianBlur(median7, 5)/255.0, np.float)
    psnr3.append(compare_psnr(src_img, median3))
    psnr5.append(compare_psnr(src_img, median5))
    psnr7.append(compare_psnr(src_img, median7))

import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
plt.plot(psnr3, color='b', label='noise level=30%')
plt.plot(psnr5, color='g', label='noise level=50%')
plt.plot(psnr7, color='r', label='noise level=70%')

plt.xlabel('# Median filter 5x5', fontsize=12)
plt.ylabel('psnr (db)', fontsize=12)
plt.title('Peak Signal to Noise Ratio (PSNR) trending by repeatly appling Median filters', fontsize=16)
plt.grid(True)
plt.legend(loc=4, prop={'size': 12})
plt.savefig("psnr.png")
plt.show()
img = np.copy(img7)
img = np.asarray(img*255, np.uint8)
median3 = np.asarray(cv2.medianBlur(img, 3)/255.0, np.float) 
median5 = np.asarray(cv2.medianBlur(img, 5)/255.0, np.float) 
median7 = np.asarray(cv2.medianBlur(img, 7)/255.0, np.float) 
median5in = np.asarray(median5*255, np.uint8)
#cv2.imwrite('noise7median5.png', median5in)
median5X2 = np.asarray(cv2.medianBlur(median5in, 5)/255.0, np.float) 
median5X2in = np.asarray(median5X2*255, np.uint8)
#cv2.imwrite('noise7median5X2.png', median5X2in)
median5X3 = np.asarray(cv2.medianBlur(median5X2in, 5)/255.0, np.float) 
median5X3in = np.asarray(median5X3*255, np.uint8)
median5X4 = np.asarray(cv2.medianBlur(median5X3in, 5)/255.0, np.float) 
median5X4in = np.asarray(median5X4*255, np.uint8)
median5X5 = np.asarray(cv2.medianBlur(median5X4in, 5)/255.0, np.float) 
#cv2.imwrite('noise7median5X5.png', median5X5*255)
median5X25 = np.copy(median5X5)
for i in range(20):
    median5X25in = np.asarray(median5X25*255, np.uint8)
    median5X25 = np.asarray(cv2.medianBlur(median5X25in, 5)/255.0, np.float) 
    if i==4:
        median5X10 = np.copy(median5X25)
    if i==9:
        median5X15 = np.copy(median5X25)
    if i==14:
        median5X20 = np.copy(median5X25)

cv2.imshow('median3', median3)
#cv2.imwrite('noise7median3.png', median3*255)
cv2.imshow('median5', median5)
cv2.imshow('median7', median7)
#cv2.imwrite('noise7median7.png', median7*255)
cv2.imshow('median5X25', median5X25)
#cv2.imwrite('noise7median5X10.png', median5X10*255)
#cv2.imwrite('noise7median5X25.png', median5X25*255)
print('psnr median3', compare_psnr(src_img, median3))
print('psnr median5', compare_psnr(src_img, median5))
print('psnr median5X2', compare_psnr(src_img, median5X2))
print('psnr median5X3', compare_psnr(src_img, median5X3))
print('psnr median5X4', compare_psnr(src_img, median5X4))
print('psnr median5X5', compare_psnr(src_img, median5X5))
print('psnr median5X10', compare_psnr(src_img, median5X10))
print('psnr median5X15', compare_psnr(src_img, median5X15))
print('psnr median5X25', compare_psnr(src_img, median5X25))
print('psnr median7', compare_psnr(src_img, median7))
'''
cv2.waitKey(0)


