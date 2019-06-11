from Utils.logger import LoggerFactory
from Model.data_gen import DataGen
from Model.train import linear_decay, half_decay, Threshold

import matplotlib.pyplot as plt

from keras.callbacks import Callback
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import plot_model

from keras.models import Sequential

import numpy as np
import tensorflow as tf
import sklearn
from sklearn.metrics import precision_recall_fscore_support

import pretty_midi
import Utils.utils as utils

from  skimage.measure import block_reduce

from PIL import Image

import os

class Trainer:
    def __init__(self, factory):
        self.__logger = LoggerFactory.getLogger(__file__)
        self.__modelFactory = factory
        self.__model = None
        self.__cfg = None
        self.__binMultiple = 0
        self.__featureBins = None
        self.__inputShape = None
        self.__inputShapeChannels = 0
        self.__savePath = None
        self.__model_ckpt = None
        self.__batch_size = None
        self.__epochs = None
        self.__trainGen = None
        self.__valGen = None
        self.__testGen = None
        self.__callbacks = None
        self.__noteRange = 0
        self.__init_lr = None

    def predict(self, input):
        if os.path.isfile(self.__model_ckpt):
            self.__logger.logInfo('loading model')
            self.__model = load_model(self.__model_ckpt)
        else:
            self.__logger.logInfo("Model not trained yet")
            return

        self.__model.compile(loss='binary_crossentropy',
            optimizer=SGD(lr=self.__init_lr,momentum=0.9), metrics=[metrics.categorical_accuracy])
        return self.__model.predict(input)

    def setCfg(self, newCfg):
        self.__cfg = newCfg

    def setModel(self, newModel):
        self.__model = newModel

    def train(self):
        x_train, y_train = [], []
        img = []
        i = 0
        for filename in os.listdir(self.__cfg.getValue("OutputDir")):
            if not filename.endswith(".jpg"):
                continue
            print(filename)

            m_fn = filename.replace(".jpg", ".mid")
            if os.path.isfile(os.path.join(self.__cfg.getValue("OutputDir"), m_fn)):
                pm = pretty_midi.PrettyMIDI(os.path.join(self.__cfg.getValue("OutputDir"), m_fn))
                oh = utils.pretty_midi_to_one_hot(pm)
                if type(oh) is not int:
                    oh = utils.slice_to_categories(oh)
                    #oh = oh.reshape(1, 128)
                    y_train.append(oh)

                    im = Image.open(os.path.join(self.__cfg.getValue("OutputDir"), filename))
                    im = im.crop((14, 13, 594, 301))
                    resize = im.resize((49, 145), Image.NEAREST)
                    resize.load()
                    #result = Image.fromarray((visual * 255).astype(numpy.uint8))
                    #resize.save("images/" + str(i) + ".jpg")
                    arr = np.asarray(resize, dtype="float32")
                    #print(arr.shape)
                    #print(arr)
                    #arr = block_reduce(arr, block_size=(2,2,1), func=np.mean)
                    x_train.append(arr)
                    #if len(x_train) > 0:
                    #    break
                    i += 1

        x_train = np.array(x_train)
        #x_train = x_train.reshape(len(x_train), 1)
        y_train = np.array(y_train)
        #print(y_train)
        #print(x_train.shape)
        #print(y_train.shape)
        #print(len(x_train))
        #print(np.shape(x_train))
        #im_array = np.array([np.array
        #x_train = np.array(x_train)
        x_test = np.copy(x_train)
        y_test = np.copy(y_train)
        #x_train, x_test, y_train, y_test = train_test_split(
        #        x_train, y_train, test_size=0.2, random_state=1)
        #print(x_train.shape)
        #print(y_train.shape)
        x_train /= 255.0
        x_test /= 255.0
        self.trainNew(x_train, y_train, x_test, y_test)

    def trainNew(self, x_train, y_train, x_test, y_test):
        if os.path.isfile(self.__model_ckpt):
            self.__logger.logInfo('loading model')
            self.__model = load_model(self.__model_ckpt)
        else:
            self.__logger.logInfo('training new model from scratch')
            type = self.__cfg.getValue("Model")
            if type == "default":
                self.__model = self.__modelFactory.base_model()#self.__modelFactory.baseline_model(self.__inputShape, self.__inputShapeChannels, self.__noteRange)
            elif type == "resnet":
                self.__model == self.__modelFactory.resnet_model(self.__binMultiple, self.__inputShape, self.__inputShapeChannels, self.__noteRange)

        self.__model.compile(loss='binary_crossentropy',
                    optimizer=SGD(lr=self.__init_lr,momentum=0.9), metrics=[metrics.categorical_accuracy])
        self.__model.summary()

        history = self.__model.fit(x_train, y_train,
                batch_size=self.__batch_size,
                epochs=self.__epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=self.__callbacks)

        score = self.__model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #self.printHistory(history)

    def trainOld(self):
        if os.path.isfile(self.__model_ckpt):
            self.__logger.logInfo('loading model')
            self.__model = load_model(self.__model_ckpt)
        else:
            self.__logger.logInfo('training new model from scratch')
            type = self.__cfg.getValue("Model")
            if type == "default":
                self.__model = self.__modelFactory.baseline_model(self.__inputShape, self.__inputShapeChannels, self.__noteRange)
            elif type == "resnet":
                self.__model == self.__modelFactory.resnet_model(self.__binMultiple, self.__inputShape, self.__inputShapeChannels, self.__noteRange)


        self.__model.compile(loss='binary_crossentropy',
                    optimizer=SGD(lr=self.__init_lr,momentum=0.9), metrics=["accuracy"])
        self.__model.summary()

        history = self.__model.fit_generator(self.__trainGen.next(),
                  self.__trainGen.steps(),
                  epochs=self.__epochs,
                  verbose=1,
                  validation_data=self.__valGen.next(),
                  validation_steps=self.__valGen.steps(),
                  callbacks=self.__callbacks
                  )
        res = self.__model.evaluate_generator(self.__testGen.next(),steps=self.__testGen.steps())
        print(self.__model.metrics_names)
        print(res)

        self.printHistory(history)

    def printHistory(self, history):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(self.__savePath,'acc.png'))

        plt.close('all')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(self.__savePath,'loss.png'))
        plt.close('all')

    def setup(self):
        cfg = self.__cfg
        saveRoot = cfg.getValue("OutputDir")
        modelName = cfg.getValue("ModelName")
        self.__savePath = os.path.join(saveRoot, modelName)
        self.__model_ckpt = os.path.join(self.__savePath, 'ckpt.h5')
        self.__logger.logInfo("SavePath for model: {0} is {1}".format(modelName, self.__savePath))

        self.__binMultiple = cfg.getValue("BinMultiple")
        self.__window_size = cfg.getValue("WindowSize")
        self.__minMidi = cfg.getValue("MinMidiValue")
        self.__maxMidi = cfg.getValue("MaxMidiValue")

        self.__noteRange = 128
        self.__featureBins = self.__noteRange * self.__binMultiple
        self.__inputShape = (self.__window_size, self.__featureBins)
        self.__inputShapeChannels = (self.__window_size, self.__featureBins, 1)

        self.__batch_size = cfg.getValue("BatchSize")
        self.__epochs = cfg.getValue("Epochs")
        #self.__trainGen = DataGen(
        #    os.path.join(self.__savePath,'data','train'),
        #    self.__batch_size
        #    )
        #self.__valGen = DataGen(
        #    os.path.join(self.__savePath,'data','val'),
        #    self.__batch_size
        #    )
        #self.__testGen = DataGen(
        #    os.path.join(self.__savePath,'data','test'),
        #    self.__batch_size
        #    )
        self.__init_lr = cfg.getValue("init_lr")
        decay = None

        checkpoint = ModelCheckpoint(self.__model_ckpt, verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(patience=5, verbose=1, mode='min')
        if cfg.getValue("lr_decay") == 'linear':
            decay = linear_decay(self.__init_lr,self.__epochs)
        else:
            decay = half_decay(self.__init_lr,5)
        csv_logger = CSVLogger(os.path.join(self.__savePath,'training.log'))
        self.__callbacks = [checkpoint,early_stop, decay,csv_logger]
