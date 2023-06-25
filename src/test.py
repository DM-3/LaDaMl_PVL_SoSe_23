import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf


data = pd.read_csv('../data/fer2013.csv', skiprows=lambda x: x in [1, 18000], nrows=12000)
pixels = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape((48, 48)))
pixels = np.array(pixels.tolist())/255.0
emotion = np.array(data['emotion'])


model = tf.keras.models.load_model('../models')
score = model.evaluate(pixels, emotion)

print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
