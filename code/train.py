import os
import cv2
import random
import skimage
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from model import find_medians
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='../logs', **kwargs):
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
n1, n2 = 70, 70

from model import *
model = fully_conv(input_size=(None, None, 3))
model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error', metrics=['accuracy'])
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
model.summary()

# checkpoint
if not os.path.exists('../checkpoints'):
    os.mkdir('../checkpoints')
filepath="../checkpoints/fully_convM5Only4-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
if not os.path.exists('../logs'):
    os.mkdir('../logs')
logging = TrainValTensorBoard(log_dir='../logs')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-8)
callbacks_list = [checkpoint, logging, reduce_lr]

# Initialization
src_img_dir = '../data/91image/'
n = 91 * 25 * 9
X = np.zeros((n, n1, n2, 3), dtype=np.float)
Y = np.zeros((n, n1, n2, 3), dtype=np.float)

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
    return patches

i = 0
for file_name in os.listdir(src_img_dir):
    if file_name.endswith(".bmp"):
        patches = gen_patches(os.path.join(src_img_dir, file_name), scales=[1], patch_size=n1)
        for patch in patches:
            for j in range(1, 10):
                noisy_img = skimage.util.random_noise(patch, mode='s&p', amount=j*0.1)
                noisy_img = noisy_img.astype('float32')
                X[i,] = noisy_img
                Y[i,] = np.asarray(patch / 255.0, np.float)
                i += 1
print("data prepared, ready to train!")

# Fit the model
history = model.fit(X, Y, validation_split=0.2, epochs=2500, batch_size=16, callbacks=callbacks_list, verbose=1)

model.save('fullyConvMedian.hdf5')



