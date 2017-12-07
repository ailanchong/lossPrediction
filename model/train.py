import numpy as np
import pandas as pd
from collections import defaultdict
import re
import pickle
from bs4 import BeautifulSoup
from gensim.models import word2vec
import sys
import os
import gc
import h5py
import json
os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional,TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec



file = h5py.File('../data/samples','r')
x_val = file['val_data'][:]
y_val = file['val_label'][:]
file.close()




DATA_DIR = "../data/"
EMBEDDING_DIM = 25
MAX_SESSION_LENGTH = 150   #the max length of all player single sessions
MAX_SESSIONS = 300   #the max sessions length of all player

with open("../data/embedding_matrix", "rb") as f_in:
    embedding_matrix = pickle.load(f_in)

embedding_layer = Embedding(len(embedding_matrix),
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SESSION_LENGTH,
                            trainable=True)

session_input = Input(shape=(MAX_SESSION_LENGTH,), dtype='int32')
embedded_sessions = embedding_layer(session_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sessions)
sessEncoder = Model(session_input, l_lstm)

usr_input = Input(shape=(MAX_SESSIONS,MAX_SESSION_LENGTH), dtype='int32')
usr_encoder = TimeDistributed(sessEncoder)(usr_input)
l_lstm_sess = Bidirectional(LSTM(100))(usr_encoder)
preds = Dense(2, activation='softmax')(l_lstm_sess)
model = Model(usr_input, preds)

print("model fitting - Bidirectional LSTM")
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])



def process(line):
    data = json.loads(line)
    x = data['data']
    y = data['label']
    return (x, y)


def generate_batch(batch_size):
    while (1):
        curr = 0
        X = []
        Y = []
        with open("../data/train_samples") as file:
            for line in file:
                x, y = process(line)
                X.append(x)
                Y.append(y)
                curr += 1
                if curr == batch_size:
                    curr = 0
                    yield (np.asarray(X), np.asarray(Y))
                    X=[]
                    Y=[]


                




'''
model.fit(x_train, y_train, validation_split = 0.1,
          epochs=5, batch_size=20, verbose=1)
'''
batch_size = 20
total_size = 43526
steps = total_size / batch_size

for i in range(2):
    model.fit_generator(generate_batch(20),steps_per_epoch=steps, epochs=1)
    model.save('my_model.h5')
    loss_and_metrics = model.evaluate(x_val, y_val, batch_size=20)
    print(loss_and_metrics)







