import os
import numpy as np
import pandas as pd
from skimage.io import imread 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

NUM_IMG = 60

def get_training_data(train_path, labels_path):
    train_files = []  # gather names of files to train
    for filename in os.listdir(train_path):
        if filename.endswith(".png"):
            train_files.append(train_path + filename)

    features = []  # load images as arrays
    for i, train_file in enumerate(train_files):
        if i >= NUM_IMG: break
        train_image = imread(train_file, True)  # loads image from file
        feature_set = np.asarray(train_image)
        features.append(feature_set)

    labels_df = pd.read_csv(labels_path)  # retrieve labels of images
    labels_df = labels_df["Finding Labels"]

    if NUM_IMG > len(train_files):
        vector_size = len(train_files)
    else:
        vector_size = NUM_IMG

    labels = np.zeros(vector_size)  # labels column vector (0 for no finding, 1 for finding)

    # adjust labels column vector
    for i in range(vector_size):
        if (labels_df[i] == 'No Finding'):
            labels[i] = 0
        else:
            labels[i] = 1
    images = np.expand_dims(np.array(features), axis=3).astype('float32') / 255  # adding single channel
    return images, labels


if __name__ == "__main__":
    x_train, y_train = get_training_data("data/train/", "data/train-labels.csv")
    x_test, y_test = get_training_data("data/test/", "data/test-labels.csv")

    f = open("log.txt","w")
    f.write("--------- Images ----------\n")
    f.write(str(x_train) + "\n")
    f.write("--------- Labels ----------\n")
    f.write(str(y_train) + "\n")
    f.close()

    # linear stack of layers
    model = Sequential()

    # Layer 1
    model.add(
        Conv2D(
            4,  # number of filters (dimensionality of output space)
            (3, 3),  # kernel size (width and height of 2D convolutional window)
            strides=(2,2), 
            activation='relu',  # rectifier linear unit [f(x) = max(0,x)]
            input_shape=(1024, 1024, 1),  # (samples, rows, cols, channels)
            data_format='channels_last'
        )
    )

    # Layer 2
    model.add(
        MaxPooling2D(
            pool_size=(2,2)  # downscaling factors (vert, horiz)
        )
    )

    # Layer 3
    model.add(
        Conv2D(
            8,  # number of filters
            (3, 3),  # kernel size
            strides=(2,2), 
            activation='relu'
        )
    )

    # Layer 4
    model.add(
        Dropout(0.25)  # randomly drops 25% of input to prevent overfitting
    )

    # Layer 5
    model.add(
        Flatten()  # flattens the output shape of the input
    )

    # Layer 6
    # perform classification on features extracted from conv and maxPooling
    model.add(
        Dense(
            1024,  # dimensionality of output space
            activation='relu'
        )
    )

    # Layer 7
    model.add(
        Dense(
            1,  # dimensionality of output space
            activation='sigmoid'
        )
    )

    # Configuration of learning process
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    # Iteration on the training data
    model.fit(
        x_train,
        y_train,
        batch_size=20,  # num of samples propogated through network
        epochs=10,  # num of forward and backward pass of all train samples
        shuffle=True,
        verbose=1
    )

    # Evaluate performance
    score = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )