from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)

import cv2
import os
import random
import numpy as np
import skimage
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from impulse_noise import add_impulse_noise

from noise import noisy
from model import find_medians
from data_generator import DataGenerator

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        logs.update({'lr': keras.eval(self.model.optimizer.lr)})
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

# input image dimensions
n1, n2 = 60, 60
params = {'batch_size': 16,
          'dim':(n1,n2),
          'shuffle': True}
train_path = 'BSD100_resize/train/'
test_path = 'BSD100_resize/test/'
train_IDs = range(80)
test_IDs = range(20)
# train_generator   = DataGenerator(data_dir=train_path,data_IDs=train_IDs,**params)
# test_generator    = DataGenerator(data_dir=test_path, data_IDs=test_IDs, **params)

from model_2d import *
model = fully_conv(input_size=(None, None, 3))
model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error', metrics=['accuracy'])
# model.load_weights('checkpoints/morph_resnet-118.h5')
'''
model = load_model(
        'checkpoints/unet_2d-123.hdf5',
        custom_objects={
            'tf':tf, 
            'find_medians': find_medians,
            'merge': merge
            })
model.optimizer = Adam(lr=1e-5)
'''
# model.save_weights('pretrained/good_Pretrain_weights.h5')
model.summary()

# checkpoint
filepath="checkpoints/fully_convM5Only4-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
        # save_weights_only=True, 
        verbose=1, save_best_only=False, mode='max')
logging = TrainValTensorBoard()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-8)
callbacks_list = [checkpoint, logging, reduce_lr]


# Initialization
src_img_dir = './pic/91image/'
n = 91 * 25 * 9
X = np.zeros((n, 70, 70, 3), dtype=np.float)
Y = np.zeros((n, 70, 70, 3), dtype=np.float)
# Generate data

def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name, scales=[1], patch_size=70, stride=20, aug_times=1):
    # read image
    img = cv2.imread(file_name)
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
    h, w, _ = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        n1 = int((h_scaled - patch_size) / stride)
        n2 = int((w_scaled - patch_size) / stride)
        for i2 in range(n2-1):
            for i1 in range(n1-1):
                x = img_scaled[i1*stride:i1*stride+patch_size, 
                        i2*stride:i2*stride+patch_size]
                patches.append(x)
                '''
                x = x.astype('float32') / 255.0
                # data aug
                for k in range(aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
                '''
    return patches

i = 0
for file_name in os.listdir(src_img_dir):
    if file_name.endswith(".bmp"):
        patches = gen_patches(os.path.join(src_img_dir, file_name))
        print(file_name, len(patches))
        for patch in patches:
            for j in range(1, 10):
                noisy_img = skimage.util.random_noise(patch, 
                       mode='s&p', amount=j*0.1)
                #patch_uint = (patch*255).astype(np.uint8)
                # noisy_img = add_impulse_noise(patch, level=j*0.1)
                # noisy_img = np.asarray(noisy_img / 255.0, np.float)
                noisy_img = noisy_img.astype('float32')
                X[i,] = noisy_img
                Y[i,] = np.asarray(patch / 255.0, np.float)
                '''
                cv2.imshow('x', X[i,])
                cv2.imshow('y', Y[i,])
                cv2.waitKey(0)
                '''
                i += 1
                if i==n:
                    break
            if i==n:
                break
        if i==n:
            break

def shuffle_in_unison(a, b, c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(c)

# shuffle_in_unison(X, Y)
print("data prepared, ready to train!")

# Fit the model
'''
model.fit({'input_image': X, 'input_loc': X_loc}, Y, validation_split=0.2, epochs=500, batch_size=16, callbacks=callbacks_list, verbose=1)
'''
history = model.fit(X, Y, validation_split=0.2, epochs=2500, batch_size=16, callbacks=callbacks_list, verbose=1)

'''
model.fit_generator(generator=train_generator,validation_data=test_generator,epochs=500,callbacks=callbacks_list,verbose=1)

# plot_loss_figure(history, 'log/' + str(datetime.now()).split('.')[0].split()[1]+'.jpg')
'''
model.save('unet_fault_2d.hdf5')



