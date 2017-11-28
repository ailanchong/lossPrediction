import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations


EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
DATA_DIR = "../data/"


sessions = []
labels = []
activities = []

'''
json format

sessions: [session_one,session_two,...]
label: 0 or 1  for loss or stay !
'''


MAX_SESSION_LENGTH = 0   #the max length of all player single sessions
MAX_SESSIONS = 0   #the max sessions length of all player

f = open(os.path.join(DATA_DIR, 'player_records'))
for line in f.readlines():
    data = json.loads(line)
    labels.append(data['label'])
    for session in data['sessions']:
        if len(session) > MAX_SESSION_LENGTH:
            MAX_SESSION_LENGTH = len(session)
        activities += session
    sessions.append(data['sessions'])
    if len(data['sessions']) > MAX_SESSIONS:
        MAX_SESSIONS = len(data['sessions'])
f.close()

print("the MAX_SESSION_LENGTH is %s" %MAX_SESSION_LENGTH)
print("the MAX_SESSIONS is %s" %MAX_SESSIONS)


data = np.zeros((len(sessions), MAX_SESSIONS, MAX_SESSION_LENGTH), dtype='int32')

for i, usr_sessions in enumerate(sessions):
    for j, session in enumerate(usr_sessions):
        for k, activity in enumerate(session):
            data[i,j,k] = activity



labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)



embeddings_index = {}
f = open(os.path.join(DATA_DIR, 'activity_embedding'))
for line in f:
    values = line.split()
    activity = values[0]
    embedding = np.asarray(values[1:], dtype='float32')
    embeddings_index[activity] = embedding
f.close()
print('Total %s activity vectors.' % len(embeddings_index))

# padding 0 ,so +1
embedding_matrix = np.random.random((len(embeddings_index) + 1, EMBEDDING_DIM))
for activity, embedding in embeddings_index.items():
    embedding_matrix[activity] = embedding


embedding_layer = Embedding(len(embeddings_index) + 1,
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
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)









