from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.layers.normalization import BatchNormalization
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model


def baseline_model(pixels):
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=(pixels*pixels,)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    return model


def inceptionResNetV2(pixels):

    # Build Xception over a custom input tensor
    input_tensor = Input(shape=(pixels, pixels, 3))
    base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def cnn_model(pixels):

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(11, 11), input_shape=(pixels, pixels, 1), padding='same'))
    model.add(BatchNormalization(momentum=0.7))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=(11, 11), padding='same'))
    model.add(BatchNormalization(momentum=0.7))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(256, kernel_size=(11, 11), padding='same'))
    model.add(BatchNormalization(momentum=0.7))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model
