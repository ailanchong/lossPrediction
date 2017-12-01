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



EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.1
TRAIN_SPLIT = 0.7
DATA_DIR = "../data/"


sessions = []
labels = []
activities = []

'''
json format

sessions: [session_one,session_two,...]
label: 0 or 1  for loss or stay !
'''


MAX_SESSION_LENGTH = 150   #the max length of all player single sessions
MAX_SESSIONS = 300   #the max sessions length of all player

f = open(os.path.join(DATA_DIR, 'record'),'rb')
player_record = pickle.load(f)
f.close()

for user_id in player_record:
    labels.append(player_record[user_id]['label'])
    #for session in player_record[user_id]['sessions']:
        #if len(session) > MAX_SESSION_LENGTH:
         #   MAX_SESSION_LENGTH = len(session)
        #activities += session
    sessions.append(player_record[user_id]['sessions'])
    #if len(data['sessions']) > MAX_SESSIONS:
     #   MAX_SESSIONS = len(data['sessions'])

del player_record


print("the MAX_SESSION_LENGTH is %s" %MAX_SESSION_LENGTH)
print("the MAX_SESSIONS is %s" %MAX_SESSIONS)


data = np.zeros((len(sessions), MAX_SESSIONS, MAX_SESSION_LENGTH), dtype='int32')

for i, usr_sessions in enumerate(sessions):
    for j, session in enumerate(usr_sessions):
        if j < MAX_SESSIONS:
            for k, activity in enumerate(session):
                if k < MAX_SESSION_LENGTH:
                    data[i,j,k] = activity



labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
nb_train_samples = int(TRAIN_SPLIT * data.shape[0])


x_train = data[:-nb_train_samples]
y_train = labels[:-nb_train_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))
del data
del labels


embeddings_index = {}
new_model = word2vec.Word2Vec.load('../data/activity_embedding')
for i in range(3640):
    activity = i+1
    embedding = new_model[str(i+1)]
    embeddings_index[activity] = embedding
print('Total %s activity vectors.' % len(embeddings_index))

del new_model

gc.collect() 

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

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(x_train, y_train,
          epochs=1, batch_size=20, verbose=1)

model.save('my_model.h5')

loss_and_metrics = model.evaluate(x_val, y_val, batch_size=20)







