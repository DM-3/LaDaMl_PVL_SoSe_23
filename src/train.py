import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_data():
    data = pd.read_csv('../data/fer2013.csv', nrows=30000)
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

'''
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(48, 48, 1)),              # input layer
    tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(1, 1), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(196, kernel_size=(3, 3), strides=(2, 2), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='sigmoid'),
    tf.keras.layers.Dense(1024, activation='sigmoid'),
    tf.keras.layers.Dense(7, activation='softmax')                            # output layer
])
'''


#### more layers ####
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(48, 48, 1)),

    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(48, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(48, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7)
])

model.summary()

var = input()



pixels, emotion = load_data()
plot_select(pixels, emotion)
# pixels, emotion = augment_data(pixels, emotion)

'''
#### most tests ####
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7)
])
'''

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)
history = model.fit(
    x=pixels,
    y=emotion,
    epochs=20,
    batch_size=32,
    validation_split=0.4,
    callbacks=[callback]
)

plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('../models')
