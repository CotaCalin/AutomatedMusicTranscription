import argparse

import matplotlib.pyplot as plt

#keras utils
from keras.callbacks import Callback
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model


import tensorflow as tf
import sklearn
from sklearn.metrics import precision_recall_fscore_support

import numpy as np

import os

def opt_thresholds(y_true,y_scores):
    othresholds = np.zeros(y_scores.shape[1])
    for label, (label_scores, true_bin) in enumerate(zip(y_scores.T,y_true.T)):
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_bin, label_scores)
        max_f1 = 0
        max_f1_threshold = .5
        for r, p, t in zip(recall, precision, thresholds):
            if p + r == 0: continue
            if (2*p*r)/(p + r) > max_f1:
                max_f1 = (2*p*r)/(p + r)
                max_f1_threshold = t
        othresholds[label] = max_f1_threshold
    return othresholds

class linear_decay(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, initial_lr,epochs):
        super(linear_decay, self).__init__()
        self.initial_lr = initial_lr
        self.decay = initial_lr/epochs

    def on_epoch_begin(self, epoch, logs={}):
        new_lr = self.initial_lr - self.decay*epoch
        print("ld: learning rate is now "+str(new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)

class half_decay(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, initial_lr,period):
        super(half_decay, self).__init__()
        self.init_lr = initial_lr
        self.period = period

    def on_epoch_begin(self, epoch, logs={}):
        factor = epoch // self.period
        lr  = self.init_lr / (2**factor)
        print("hd: learning rate is now "+str(lr))
        K.set_value(self.model.optimizer.lr, lr)

class Threshold(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, val_data):
        super(Threshold, self).__init__()
        self.val_data = val_data
        _,y = val_data
        self.othresholds = np.full(y.shape[1],0.5)

    def on_epoch_end(self, epoch, logs={}):
        #find optimal thresholds on validation data
        x,y_true = self.val_data
        y_scores = self.model.predict(x)
        self.othresholds = opt_thresholds(y_true,y_scores)
        y_pred = y_scores > self.othresholds
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true,y_pred,average='micro')
        print("validation p,r,f,s:")
        print(p,r,f,s)
