import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense


def load_data():
    data = pd.read_csv('..\\data\\fer2013.csv', nrows=30000)
    pixels = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape((48, 48)))
    pixels = np.array(pixels.tolist())/255.0
    emotion = np.array(data['emotion'])

    return pixels, emotion


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


def augment_data(pixels, emotion):
    pixels = np.concatenate((pixels, pixels), 0)
    emotion = np.concatenate((emotion, emotion), 0)
    for i in range(48):
        pixels[30000:,:,i] = pixels[:30000,:,47-i]
    print('data shape: ', pixels.shape)

    return pixels, emotion


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    x = np.arange(start=1, stop=len(history.history['loss']) + 1, step=1)

    ax1.plot(x, history.history['sparse_categorical_accuracy'], color='r', label='train')
    ax1.plot(x, history.history['val_sparse_categorical_accuracy'], color='g', label='test')
    ax1.set_ylabel('accuracy')
    ax1.legend(loc='upper left')

    ax2.plot(x, history.history['loss'], color='y', label='train')
    ax2.plot(x, history.history['val_loss'], color='b', label='test')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    plt.legend(loc='upper right')

    fig.suptitle('model loss / accuracy')
    plt.show()


'''
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(48, 48, 1)),              # input layer
    tf.keras.layers.Cropping2D(cropping=((1, 1), (1, 1))),
    tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu'),     # stride=(1, 1)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),       # 1024
    tf.keras.layers.Dense(1024, activation='relu'),       # 1024
    tf.keras.layers.Dense(7, activation='sigmoid')                            # output layer
])
'''

''' # close to FER2013 paper
model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(48, 48, 1)),

    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7)
])
'''


''' # close to Alexnet
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(48, 48, 1)),

    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7)
])
'''


def create_model_0():
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


model = create_model_2()

model.summary()

var = input()



pixels, emotion = load_data()
plot_select(pixels, emotion)
pixels, emotion = augment_data(pixels, emotion)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(
    x=pixels,
    y=emotion,
    epochs=30,
    batch_size=32,
    validation_split=0.3,
    callbacks=[callback]
)
plot_history(history)

model.save('../models')