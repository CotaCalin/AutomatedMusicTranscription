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
from keras.models import Sequential
import numpy as np

class ModelFactory:
    @staticmethod
    def base_model2():
        img_x, img_y = 145, 49
        input_shape = (img_x, img_y, 3)
        num_classes = 128

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
            activation='relu',
            input_shape=input_shape))
        #model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Final output layer
        #model.add(Conv2D(128, (5,5), activation='sigmoid'))
        #model.add(Flatten())
        model.add(Flatten())
        #model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    @staticmethod
    def base_model():
        img_x, img_y = 145, 49
        input_shape = (img_x, img_y, 3)
        num_classes = 128

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), strides=(2,2),
            activation='relu',
            input_shape=input_shape))
        #model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="valid", activation='relu'))
        # Max Pooling
        #model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding="valid", activation='relu'))

        # 4th Convolutional Layer
        #model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding="valid", activation='relu'))

        # 5th Convolutional Layer
        #model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding="valid", activation='relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, activation="relu"))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096, activation="relu"))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(num_classes, activation="softmax"))

        return model

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
        outputs = Dense(note_range, activation='relu')(do4)

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
            #residual block
            bn = BatchNormalization()(pool)
            re = Activation('relu')(bn)
            freq_range = (bin_multiple/(2**(i+1)))*note_range
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
