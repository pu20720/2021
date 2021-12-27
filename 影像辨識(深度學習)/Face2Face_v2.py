# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 22:30:05 2021

@author: a2482
"""
from keras_contrib.layers import InstanceNormalization
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import scipy
from glob import glob
import numpy as np
import tensorflow as tf

tf.compat.v1.experimental.output_all_intermediates(True)
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(True)

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def load_batch(data_dir, batch_size = 8):
    pathA = glob(data_dir+r'/trainA/*')
    pathB = glob(data_dir+r'/trainB/*')
    '''
    print('-'*20)
    print("pathA's length = {}, pathB's Length = {} ".format(len(pathA),len(pathB)))
    print('-'*20)
    '''
    
    n_batches = int(min(len(pathA), len(pathB))/ batch_size)
    total_samples = n_batches * batch_size
    index = np.random.choice(range(len(pathA)), total_samples, replace=False)    
    pathA = np.array(pathA)[index]
    pathB = np.array(pathB)[index]
    
    for i in range(n_batches-1):
        batch_A = pathA[i*batch_size:(i+1)*batch_size]
        batch_B = pathB[i*batch_size: (i+1)*batch_size]
        
        imgAs, imgBs = [], []  
        for img_A, img_B in zip(batch_A, batch_B):
            img_A = imread(img_A)
            img_B = imread(img_B)
            imgAs.append(img_A)
            imgBs.append(img_B)
        imgAs_regular = np.array(imgAs)/127.5 - 1.
        imgBs_regular = np.array(imgBs)/127.5 - 1.

        yield imgAs_regular, imgBs_regular


def conv2d(layer_input, filters, f_size=4, normalization=True):
  d = Conv2D(filters, kernel_size=f_size,
             strides=2, padding='same')(layer_input)
  d = LeakyReLU(alpha=0.2)(d)
  if normalization:
      d = InstanceNormalization()(d)
  return d

def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
      u = UpSampling2D(size=2)(layer_input)
      u = Conv2D(filters, kernel_size=f_size, strides=1,
                 padding='same', activation='relu')(u)
      if dropout_rate:
          u = Dropout(dropout_rate)(u)
      u = InstanceNormalization()(u)
      u = Concatenate()([u, skip_input])
      return u    
      
def build_models(img_shape):
       """U-Net Generator"""
       # Image input
       gf = 32
       channels = img_shape[2]
       d0 = Input(shape=img_shape)      # 256*256*3

       # Downsampling
       d1 = conv2d(d0, gf)         # 128*128*32
       d2 = conv2d(d1, gf)         # 64*64*32
       d3 = conv2d(d2, gf * 2)     # 32*32*64
       d4 = conv2d(d3, gf * 4)     # 16*16*128
       d5 = conv2d(d4, gf * 8)     # 8*8*256
       # Upsampling
       u1 = deconv2d(d5, d4, gf * 4)   # 16*16*256
       u2 = deconv2d(u1, d3, gf * 2)   # 32*32*128
       u3 = deconv2d(u2, d2, gf)       # 64*64*64
       u4 = deconv2d(u3, d1, gf)       # 128*128*64
       u5 = UpSampling2D(size=2)(u4)             # 256*256*64
       output_img = Conv2D(channels, kernel_size=4,
                           strides=1, padding='same', activation='tanh')(u5)   # 256*256*3

       return Model(d0, output_img)
   
def mean(data):
    return sum(data)/len(data)

def printing(data_loss):
    epochs = range(1, len(data_loss)+1)
    plt.plot(epochs, data_loss, 'b', label='Training loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def main():
    data_dir = 'train'
    img_shape = (256,256,3)
    
    model = build_models(img_shape)
    #model.summary()
    model.compile(loss='mse', optimizer='adam')

    epochs = 400
    batch_size = 8    
    
    data_loss = []
    for epoch in range(epochs):
        data_batch_loss = []
        for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(data_dir, batch_size)):
            batch_loss = model.train_on_batch(imgs_A, imgs_B)
            data_batch_loss.append(batch_loss)
        dataBatchLoss_mean = mean(data_batch_loss)
        data_loss.append(dataBatchLoss_mean)
        print('{} epoch findished loss = {}'.format(epoch+1, dataBatchLoss_mean))
    model.save('model.h5')

    printing(data_loss)
    
    
if __name__ == '__main__':
    main()