import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense, ZeroPadding2D
import random


### Utilities ###

def load_data():
    data = pd.read_csv('../data/fer2013.csv')
    pixels = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape((48, 48)))
    pixels = np.array(pixels.tolist()) / 255.0
    emotion = np.array(data['emotion'])

    return pixels, emotion,  pixels.shape[0]


def plot_select(pixels, emotion):
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pixels[i, :], cmap='gray')
        plt.xlabel(class_names[emotion[i]])
    plt.show()


def augment_data(pixels, emotion, n_elem):
    # mirror images
    pixels = np.concatenate((pixels, pixels), 0)
    emotion = np.concatenate((emotion, emotion), 0)
    for i in range(48):
        pixels[n_elem:, :, i] = pixels[:n_elem, :, 47 - i]
    n_elem *= 2

    # randomly shift left or right by 1 pixel
    pixels = np.concatenate((pixels, np.zeros_like(pixels)), 0)
    emotion = np.concatenate((emotion, emotion), 0)
    for j in range(n_elem):
        if random.random() < 0.5:
            for i in range(47):
                pixels[n_elem+j, :, i] = pixels[j, :, i+1]
        else:
            for i in range(47):
                pixels[n_elem+j, :, i+1] = pixels[j, :, i]
    n_elem *= 2
    
    return pixels, emotion, n_elem


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    x = np.arange(start=1, stop=len(history.history['loss']) + 1, step=1)

    ax1.plot(x, history.history['sparse_categorical_accuracy'], color='r', label='train')
    ax1.plot(x, history.history['val_sparse_categorical_accuracy'], color='g', label='val')
    ax1.set_ylabel('accuracy')
    ax1.legend(loc='upper left')

    ax2.plot(x, history.history['loss'], color='y', label='train')
    ax2.plot(x, history.history['val_loss'], color='b', label='val')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    plt.legend(loc='upper right')

    fig.suptitle('model loss / accuracy')
    plt.show()


### CNN Models ###

def create_model_0():
    # parameters: 1'084'583
    # augmentation: none
    # learn_rate: 0.001
    # batch_size: 32
    # test_accuracy: 47.14%
    # time per epoch: 20 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dense(7)
    ])
    return model


def create_model_1():
    # parameters: 994'871
    # augmentation: none
    # learn_rate: 0.001
    # batch_size: 32
    # test_accuracy: 45.79%
    # time per epoch: 40 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])
    return model


def create_model_2():
    # parameters: 349'559
    # augmentation: none
    # learn_rate: 0.001
    # batch_size: 32
    # test accuracy: 52.65%
    # time per epoch: 52 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])
    return model


def create_model_3():
    # parameters: 126'967
    # augmentation: none
    # learn_rate: 0.001
    # batch_size: 32
    # test accuracy: 55.33%
    # time per epoch: 54 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])
    return model


def create_model_4():
    # parameters: 174'511
    # augmentation: none
    # learn_rate: 0.0005
    # batch_size: 16
    # test accuracy: 54.69%
    # time per epoch: 86 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(24, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(24, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(48, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(48, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])
    return model


def create_model_5():
    # parameters: 145'495
    # augmentation: none
    # learn_rate: 0.001
    # batch_size: 32
    # test accuracy: 55.48%
    # time per epoch:  70 s
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(48, 48, 1)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(48, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        Conv2D(48, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])    
    return model


def create_model_6():
    # parameters: 268'983
    # augmentation: mirror, shift
    # learn_rate: 0.001
    # batch_size: 32
    # test accuracy: 74.27%
    # time per epoch:  380 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(96, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(96, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(80, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])
    return model


def create_model_7():
    # parameters: 100'215
    # learn_rate: 0.001
    # batch_size: 32
    # test accuracy:
    # - base: 50.13%; 60 s
    # - mirror: 59.87%; 110-131 s
    # - shift: 66.64%; 110-131 s
    # - both: 68.77%; 250 s
    model = tf.keras.models.Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])
    return model


model = create_model_7()
model.summary()


# load data set
pixels, emotion, n_elem = load_data()
print('# elements: ' + str(n_elem))
plot_select(pixels, emotion)

# augment data
pixels, emotion, n_elem = augment_data(pixels, emotion, n_elem)

#split into test and training data
x_test,  y_test  = pixels[0:int(n_elem*0.2), :],        emotion[0:int(n_elem*0.2)]
x_train, y_train = pixels[int(n_elem*0.2):n_elem, :],   emotion[int(n_elem*0.2):n_elem]

print('# training samples: ' + str(int(n_elem * 0.8)))


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[callback]
)
plot_history(history)

print('\ntest performance: ')
test_result = model.evaluate(x=x_test, y=y_test)
print(str(model.metrics_names[0]) + ': ' + str(test_result[0]))
print(str(model.metrics_names[1]) + ': ' + str(test_result[1]))


model.save('../models')
