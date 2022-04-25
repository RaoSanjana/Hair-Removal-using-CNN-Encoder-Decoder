import math
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers,  Input, losses
from tensorflow.keras.layers import InputLayer,  Flatten
from  keras.layers.core import Dense
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
import glob
from os.path import join, basename
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.layers import merge
from tensorflow.python.ops import math_ops

tf.random.set_seed(42)

BASE_PATH = 'jpeg-melanoma-512x512/train/'
batch_size = 4

def PSNR(y_true, y_pred):
    y_true = y_true[:, :, :, :3]
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

def custom_loss(y_true, y_pred):
     mask_val = y_true[:, :, :, 3:]
     y_true1 = tf.math.multiply(y_true[:, :, :, :3],  mask_val)
     y_pred1 = tf.math.multiply(y_pred,  mask_val)
     
     diff_tot = 0 * math_ops.squared_difference(y_true1, y_pred1) + 1 * math_ops.squared_difference(y_true[:, :, :, :3], y_pred)
     #diff_tot = 0.01 * (1 / PSNR(y_true, y_pred) ) + 1.2 * K.mean(math_ops.squared_difference(y_true1, y_pred1), axis=-1) + 1 * K.mean(math_ops.squared_difference(y_true[:, :, :, :3], y_pred), axis=-1)

     loss = K.mean(diff_tot, axis=-1)
     return loss


all_images = glob.glob(join(BASE_PATH, "*R.jpg"))[:700]
all_images = [basename(x) for x in all_images]
all_images.sort()

X = [fname.split(".")[0][:-1] + ".jpg" for fname in all_images]
Y = all_images

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

class data_generator(keras.utils.Sequence) :
  
  def __init__(self, x_image_filenames, y_image_filenames, batch_size) :
    self.image_filenames = x_image_filenames
    self.labels = y_image_filenames
    self.batch_size = batch_size
    self.mask_fnames = [x.split(".")[0][:-1] + "M.jpg" for x in self.labels]
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    masks_y = self.mask_fnames[idx * self.batch_size : (idx+1) * self.batch_size]

    mask = np.array([resize(imread(join(BASE_PATH, file_name)), (512, 512, 1)) for file_name in masks_y])/255.0
    stacked_mask = np.concatenate([mask, mask, mask], axis=3)
    
    return (np.array([resize(imread(join(BASE_PATH, file_name)), (512, 512, 3)) for file_name in batch_x])/255.0), \
           tf.concat( [(np.array([resize(imread(join(BASE_PATH, file_name)), (512, 512, 3)) for file_name in batch_y])/255.0), stacked_mask ], axis=3)


train_gen = data_generator(X_train, Y_train, batch_size)
val_gen = data_generator(X_val, Y_val, batch_size)
test_gen = data_generator(X_test, Y_test, batch_size)


encodeco_input = keras.Input(shape=(512, 512, 3), name="original_img")
x = layers.Conv2D(128, 3,strides=(1, 1),padding="same", activation="relu")(encodeco_input)
x1 = layers.MaxPooling2D(strides=2)(x)
x2 = layers.Conv2D(256, 3, strides=(1, 1),padding="same", activation="relu")(x1)
x3 = layers.MaxPooling2D(strides=2)(x2)
x4 = layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu")(x3)
x5 = layers.UpSampling2D(2)(x4)
SkipConn1 = layers.concatenate([x2, x5])
x6 = layers.Conv2DTranspose(128, 3, strides=(1, 1), padding="same", activation="relu")(SkipConn1)
SkipConn2 = layers.concatenate([x1, x6])
x7 = layers.UpSampling2D(2)(SkipConn2)
SkipConn3 = layers.concatenate([x, x7])
x8 = layers.Conv2DTranspose(128, 3, strides=(1, 1), padding="same", activation="relu")(SkipConn3)
encodeco_output = layers.Conv2DTranspose(3, 3, strides=(1, 1), padding="same", activation="relu")(x8)
 

encodecoder = keras.Model(encodeco_input, encodeco_output, name="encoderdecoder")
opt = keras.optimizers.Adam(learning_rate=0.0001)
encodecoder.compile(loss=custom_loss, optimizer=opt, metrics=[PSNR])

encodecoder.fit_generator(generator=train_gen, validation_data=val_gen, epochs = 10)
encodecoder.save('./')
