import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd


# The step-size for moving the filter across the input is called the stride.

def keras_model(image_x, image_y):
    num_of_classes = 12
    model = Sequential()

    # the output of the convolution may be passed through a so-called Rectified Linear Unit (ReLU), 
    # which merely ensures that the output is positive because negative values are set to zero
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    # For a multi-class classification problem use categorical_crossentropy loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "emojinator.h5"
    # ModelCheckpoint save the model after every epoch.
    #  if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
    # For val_acc, (mode) should be max, for val_loss (mode) should be min
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list


def main():
    data = pd.read_csv("train_foo.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    X = dataset
    Y = dataset
    X = X[:, 1:2501]
    Y = Y[:, 0]

    X_train = X[0:12000, :]
    X_train = X_train / 255.
    X_test = X[12000:13201, :]
    X_test = X_test / 255.

    # Reshape
    Y = Y.reshape(Y.shape[0], 1)
    Y_train = Y[0:12000, :]
    Y_train = Y_train.T
    Y_test = Y[12000:13201, :]
    Y_test = Y_test.T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    image_x = 50
    image_y = 50

    train_y = np_utils.to_categorical(Y_train)
    test_y = np_utils.to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))

    model, callbacks_list = keras_model(image_x, image_y)

    # Keras models are trained on Numpy arrays of input data and labels.
    # For training a model, we will typically use the fit function.
    # Trains the model for a given number of epochs (iterations on a dataset).
    model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64, callbacks=callbacks_list)

    # Returns the loss value & metrics values for the model in test mode.
    # Verbosity mode. 0 = silent, 1 = progress bar.
    scores = model.evaluate(X_test, test_y, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('emojinator.h5')


main()
