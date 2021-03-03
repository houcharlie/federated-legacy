from tensorflow.keras.layers import Lambda, Conv2D, Softmax, Reshape, Dense, Conv2DTranspose, Permute, Add, BatchNormalization, ReLU, AveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers
from .scale import Scale
from tensorflow_addons.layers.spectral_normalization import SpectralNormalization

import tensorflow as tf 
w_l2 = 1e-4 
weight_decay = 2.5e-5      
class SelfAttention(tf.keras.layers.Layer):
  '''
  Code from: https://github.com/shaoanlu/faceswap-GAN/blob/master/networks/nn_blocks.py
  '''
  def __init__(self, nc, **kwargs):
    super(SelfAttention, self).__init__(**kwargs)
    self.f = SpectralNormalization(Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2)))
    self.g = SpectralNormalization(Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2)))
    self.h = SpectralNormalization(Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2)))
    #self.f = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))
    #self.g = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))
    #self.h = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))
    self.s = Lambda(lambda x: K.batch_dot(x[0], Permute((2,1))(x[1])))
    self.o = Lambda(lambda x: K.batch_dot(x[0], x[1]))

    self.softmax = Softmax(axis=-1)
    self.scale = Scale()
  def call(self, inp):
    x = inp
    shape_x = x.get_shape().as_list()
    f = self.f(x)
    g = self.g(x)
    h = self.h(x)
    shape_f = f.get_shape().as_list()
    shape_g = g.get_shape().as_list()
    shape_h = h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(f)
    flat_g = Reshape((-1, shape_g[-1]))(g)
    flat_h = Reshape((-1, shape_h[-1]))(h) 
    s = self.s([flat_g, flat_f])
    beta = self.softmax(s)
    o = self.o([beta, flat_h])
    o = Reshape(shape_x[1:])(o)
    o = self.scale(o)
    
    out = Add()([o, inp])
    return out
class Up_ResNet(tf.keras.layers.Layer):
  def __init__(self, nc, **kwargs):
    super(Up_ResNet, self).__init__(**kwargs)
    self.deconv1 = SpectralNormalization(Conv2DTranspose(
          filters=nc,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding='same',
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.deconv2 = SpectralNormalization(Conv2DTranspose(
          filters=nc,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding='same',
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.conv1 = SpectralNormalization(Conv2D(
          filters=nc,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.conv2 = SpectralNormalization(Conv2D(
          filters=nc,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.conv3 = SpectralNormalization(Conv2D(
          filters=nc,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.batchnorm1 = BatchNormalization(scale=False)
    self.batchnorm2 = BatchNormalization(scale=False)
  def call(self, input_):
    output = input_
    output = self.batchnorm1(output)
    output = ReLU()(output)
    output = self.deconv1(output)
    output = self.conv1(output)
    output = self.batchnorm2(output)
    output = ReLU()(output)
    output = self.conv2(output)

    output2 = input_
    output2 = self.deconv2(output2)
    output2 = self.conv3(output2)
    return output + output2

class Down_ResNet(tf.keras.layers.Layer):
  def __init__(self, nc, first=False, **kwargs):
    super(Down_ResNet, self).__init__(**kwargs)
    self.first = first
    self.conv1 = SpectralNormalization(Conv2D(
          filters=nc,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.conv2 = SpectralNormalization(Conv2D(
          filters=nc,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
    self.conv3 = SpectralNormalization(Conv2D(
          filters=nc,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
  def call(self, input_):
    output = input_
    if not self.first:
      output = ReLU()(output)
    output = self.conv1(output)
    output = ReLU()(output)
    output = self.conv2(output)
    output = AveragePooling2D(pool_size=(2,2), strides=2, padding="same")(output)

    output2 = input_
    if not self.first:
      output2 = self.conv3(output2)
      output2 = AveragePooling2D(pool_size=(2,2), strides=2, padding="same")(output2)
    else:
      output2 = AveragePooling2D(pool_size=(2,2), strides=2, padding="same")(output2)
      output2 = self.conv3(output2)
    return output + output2
