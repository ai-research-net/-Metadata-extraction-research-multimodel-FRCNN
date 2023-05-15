import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle5 as pickle
import tensorflow as tf
import pandas as pd
import numpy as np

import keras

from keras.layers import Conv1D

from tensorflow.keras import layers

model = keras.models.Sequential()

#model.add(Conv1D(64, 3, activation="relu", input_shape = (114, 1)))
#model.add(layers.MaxPooling1D())
#model.add(Conv1D(16, 3, activation="relu"))
#model.add(layers.MaxPooling1D())
#model.add(Conv1D(64, 3, activation="relu"))
#model.add(layers.MaxPooling1D())


#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dense(16, activation='relu'))
#model.add(layers.Dense(8, activation="sigmoid"))

# MLP_2
'''
model.add(layers.Dense(64, input_dim=114, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(23, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
# MLP_3
'''
model.add(layers.Dense(64, input_dim=114, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
# MLP_4
'''
model.add(layers.Dense(128, input_dim=114, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
# MLP_5
'''
model.add(layers.Dense(128, input_dim=114, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''

#MLP5
'''
model.add(layers.Dense(256, input_dim=114, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
#MLP6
'''
model.add(layers.Dense(256, input_dim=114, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
#MLP7
'''
model.add(layers.Dense(256, input_dim=114, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
#MLP8
'''
model.add(layers.Dense(512, input_dim=114, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
'''
#MLP9
model.add(layers.Dense(512, input_dim=114, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

with open('full_train_df.pkl', 'rb') as f:
    train = pickle.load(f)
    train = train.iloc[1:, :]

from sklearn.metrics import classification_report

out_train = model.predict(X)

print(classification_report(np.argmax(target, axis=1), np.argmax(out_train, axis=1)))