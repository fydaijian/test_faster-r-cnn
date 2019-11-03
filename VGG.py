import numpy as np
import tensorflow.keras.backend as k
import tensorflow as tf
import math
import random
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax, Flatten,Conv2D
from tensorflow.keras.optimizers import RMSprop, Adam
def VGG():
    inputs = Input(shape=[224, 224, 3])
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    for l in base_model.layers:
        l.trainable = False # 第一步锁定前层，不进行训练
    lasted_layer=base_model.get_layer("block5_conv3")
    return lasted_layer

def rpn(base_layers, num_anchors):

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]
