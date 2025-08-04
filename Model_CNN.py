import numpy as np
from keras import layers, models
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_Cls(X, Y, test_x, test_y, Batch_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 'relu'
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 'relu'
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 'relu'
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(test_y.shape[-1]))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])  # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.fit(X, Y, epochs=50, batch_size=Batch_size, validation_data=(test_x, test_y))
    pred = model.predict(test_x)
    return pred


def Model_CNN(train_data, train_target, test_data, test_target, Batch_size):
    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    pred = Model_Cls(Train_X, train_target, Test_X, test_target, Batch_size)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = ClassificationEvaluation(pred, test_target)
    return Eval, pred
