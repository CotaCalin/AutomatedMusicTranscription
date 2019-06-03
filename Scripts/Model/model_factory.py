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

class ModelFactory:
    @staticmethod
    def baseline_model(input_shape, input_shape_channels, note_range):
        inputs = Input(shape=input_shape)
        reshape = Reshape(input_shape_channels)(inputs)

        #normal convnet layer (have to do one initially to get 64 channels)
        conv1 = Conv2D(50,(5,25),activation='tanh')(reshape)
        do1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(1,3))(do1)

        conv2 = Conv2D(50,(3,5),activation='tanh')(pool1)
        do2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(1,3))(do2)

        flattened = Flatten()(pool2)
        fc1 = Dense(1000, activation='sigmoid')(flattened)
        do3 = Dropout(0.5)(fc1)

        fc2 = Dense(200, activation='sigmoid')(do3)
        do4 = Dropout(0.5)(fc2)
        outputs = Dense(note_range, activation='sigmoid')(do4)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def resnet_model(bin_multiple, input_shape, input_shape_channels, note_range):
        #input and reshape
        inputs = Input(shape=input_shape)
        reshape = Reshape(input_shape_channels)(inputs)

        #normal convnet layer (have to do one initially to get 64 channels)
        conv = Conv2D(64,(1,bin_multiple*note_range),padding="same",activation='relu')(reshape)
        pool = MaxPooling2D(pool_size=(1,2))(conv)

        for i in range(int(np.log2(bin_multiple))-1):
            #print i
            #residual block
            bn = BatchNormalization()(pool)
            re = Activation('relu')(bn)
            freq_range = (bin_multiple/(2**(i+1)))*note_range
            #print freq_range
            conv = Conv2D(64,(1,freq_range),padding="same",activation='relu')(re)

            #add and downsample
            ad = add([pool,conv])
            pool = MaxPooling2D(pool_size=(1,2))(ad)

        flattened = Flatten()(pool)
        fc = Dense(1024, activation='relu')(flattened)
        do = Dropout(0.5)(fc)
        fc = Dense(512, activation='relu')(do)
        do = Dropout(0.5)(fc)
        outputs = Dense(note_range, activation='sigmoid')(do)

        model = Model(inputs=inputs, outputs=outputs)

        return model
