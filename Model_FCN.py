from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.utils import to_categorical
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_FCNs(X, Y, test_x, test_y, Batch_size):
    num_classes = Y.shape[-1]  # Number of classes
    input_shape = X.shape[1:]  # Input shape

    # Define the FCN model using Functional API
    inputs = Input(shape=input_shape)

    # Encoder: Convolutional Layers with MaxPooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Decoder: Transposed Convolutions
    x = Conv2DTranspose(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)  # Output layer for classification

    # Build the model
    model = Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(X, Y, epochs=50, batch_size=Batch_size, validation_data=(test_x, test_y))

    # Predict on test data
    pred = model.predict(test_x)
    return pred


def Model_FCN(train_data, train_target, test_data, test_target, Batch_size):
    IMG_SIZE = 32

    # Resize train data to the required shape
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Resize test data to the required shape
    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Convert targets to categorical (one-hot encoding)
    Train_Y = to_categorical(train_target)
    Test_Y = to_categorical(test_target)

    # Train the FCN model
    pred = Model_FCNs(Train_X, Train_Y, Test_X, Test_Y, Batch_size)

    # Threshold predictions for binary classification (if applicable)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    # Evaluate predictions
    Eval = ClassificationEvaluation(pred, Test_Y)
    return Eval, pred
