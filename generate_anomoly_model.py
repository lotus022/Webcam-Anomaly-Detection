from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

from skimage.io import imread
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ANOMOLY_PATH = os.path.join('training', 'anomoly')
NOISE_PATH = os.path.join('training', 'noise')
MODEL_PATH = os.path.join('models', 'anomoly_model.h5')

CHUNK_SIZE = 500
BATCH_SIZE = 6
EPOCHS = 18

input_shape = (720, 1280, 3)

def chunks(l, n):
    """Generator for generating chunks of a list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_images = {}

    for fn in os.listdir(ANOMOLY_PATH):
        if fn.endswith('.jpg'):
            train_images.update({os.path.join(ANOMOLY_PATH, fn) : 1})

    for fn in os.listdir(NOISE_PATH):
        if fn.endswith('.jpg'):
            train_images.update({os.path.join(NOISE_PATH, fn) : 0})

    filenames = list(train_images.keys())
    random.shuffle(filenames) # Randomize Data

    def generate_data(chunk_size):
        """
        Loads data into memory chunk by chunk.

        Parameters
        ----------
        chunk_size : int
            The size of each chunk to generate

        Yields
        ------
        Tuple
            A tuple (X, y) containing a single chunk of training data
        """
        for chunk in chunks(filenames, chunk_size):
            X = np.array([imread(fn) for fn in chunk]) / 255
            y = np.array([train_images[fn] for fn in chunk])
            yield (X, y)

    for X, y in generate_data(CHUNK_SIZE): # Fit the model w/each chunk and save

        model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.2, shuffle=True)
        model.save(MODEL_PATH)

if __name__ == '__main__':
    main()
