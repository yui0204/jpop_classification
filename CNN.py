from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv1D, Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers import Activation, RepeatVector, Flatten, Reshape, Dense
from keras.layers import merge, MaxPooling2D, UpSampling2D, core, GRU, LSTM
from keras.layers.wrappers import Bidirectional

from keras.layers.merge import multiply, dot


def CNN(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 128, 256, 64
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) #64

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) #32
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) #16
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 8, 16, 512

    """
    x = Reshape((8 * 512, 16))(x)

    x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid',
                            return_sequences=True,
                            dropout=0.25, recurrent_dropout=0.25, stateful=False)(x)
    
    x = Conv1D(512, 1, activation='sigmoid')(x)
    print(x)
    x = Reshape((8, -1, 512))(x)
    """                                
            
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
#    x = Dense(4096, activation='relu')(x)
    x = Dense(3, activation='relu')(x)
    x = core.Activation('softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model


def CRNN(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=1)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((4, 1), strides=(4, 1))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((4, 1), strides=(4, 1))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((4, 1), strides=(4, 1))(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((4, 1), strides=(4, 1))(x) # ?, 1, 512, 512

    
    x = Reshape((input_width, 512))(x)
        
    x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(x) 
    x = BatchNormalization()(x)

    x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(x) 
    x = BatchNormalization()(x)

    x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(x) 
    x = BatchNormalization()(x)

    
    x = Conv1D(n_classes, 1, activation='sigmoid')(x)
    x = Reshape((1, -1, 75))(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model





def BiCRNN8(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=1)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2, 1), strides=(2, 1))(x)

    
    x = Reshape((input_width, 512))(x)
        
    x = Bidirectional(GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    
    x = Conv1D(n_classes, 1, activation='sigmoid')(x)
    x = Reshape((1, -1, 75))(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model

