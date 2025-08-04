import numpy as np
import tensorflow as tf
from keras import layers, models
from Classificaltion_Evaluation import ClassificationEvaluation


def residual_block(x, filters, kernel_size=3, stride=1, use_attention=True):
    # Standard residual block
    res = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    res = layers.BatchNormalization()(res)
    res = layers.Activation('relu')(res)

    # Attention mechanism
    if use_attention:
        att = layers.GlobalAveragePooling2D()(res)
        att = layers.Dense(filters // 16, activation='relu')(att)
        att = layers.Dense(filters, activation='sigmoid')(att)
        att = layers.Reshape((1, 1, filters))(att)
        res = layers.Multiply()([res, att])

    # Skip connection
    shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    output = layers.Add()([res, shortcut])
    output = layers.Activation('relu')(output)
    return output


def build_adaptive_residual_attention_network(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolutional layer
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Stack of residual blocks with attention
    for _ in range(3):
        x = residual_block(x, 10)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x, name='adaptive_residual_attention_network')
    return model


def Model_Ran(Data, Target, Batch_size=None):
    if Batch_size is None:
        Batch_size = 16
    IMG_SIZE = 256
    Train_x = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_x[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = build_adaptive_residual_attention_network([256, 256, 3], 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_x, Target, epochs=250, batch_size=Batch_size, verbose=2)
    # Get the weights and bias
    weights = model.layers[1].get_weights()
    return weights[0]


def Model_Ran_cls(Data, Target, mlp_weight, Batch_size, sol=None):
    if sol is None:
        sol = [-20, 20]
    Weight = Model_Ran(Data, Target, Batch_size)
    w = Weight + Weight * (sol[0])
    model = mlp_weight

    pred = model.predict(w)
    pred = np.asarray(pred)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = ClassificationEvaluation(pred, Target)
    return Eval, pred
