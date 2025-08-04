import numpy as np
import cv2 as cv
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D

from Classificaltion_Evaluation import ClassificationEvaluation


def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', dilation_rate=(1, 1), use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        block_x = Activation('relu')(block_x)
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', dilation_rate=(1, 1), padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(num_of_class=1):
    dense_block_size = 3
    layers_in_block = 4
    growth_rate = 12
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', dilation_rate=(1, 1), padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(dense_x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(num_of_class, activation='softmax')(dense_x)
    model = Model(input_img, output)

    return model


def Model_Dil_Densenet(Image, Target, Batch_size=None):
    if Batch_size is None:
        Batch_size = 16

    IMG_SIZE = [32, 32, 3]

    Feat = np.zeros((Image.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Image.shape[0]):
        Feat[i, :] = cv.resize(Image[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = dense_net(Target.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Data, Target, steps_per_epoch=10, epochs=10, batch_size=Batch_size)
    # Get the weights and bias
    weights = model.layers[1].get_weights()
    return weights[0]


def Model_Dil_Densenet_cls(Data, Target, mlp_weight, Batch_size, sol=None):
    if sol is None:
        sol = [-20, 20]
    Weight = Model_Dil_Densenet(Data, Target, Batch_size)
    w = Weight + Weight * (sol[0])
    model = mlp_weight

    pred = model.predict(w)
    pred = np.asarray(pred)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = ClassificationEvaluation(pred, Target)
    return Eval, pred
