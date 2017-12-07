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



EMBEDDING_DIM = 25
TEST_SPLIT = 0.1
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


MAX_SESSION_LENGTH = 150   #the max length of all player single sessions
MAX_SESSIONS = 300   #the max sessions length of all player

f = open(os.path.join(DATA_DIR, 'record'),'rb')
player_record = pickle.load(f)
f.close()

for user_id in player_record:
    
    #for session in player_record[user_id]['sessions']:
        #if len(session) > MAX_SESSION_LENGTH:
         #   MAX_SESSION_LENGTH = len(session)
        #activities += session
    curr_session = player_record[user_id]['sessions']
    curr_session = np.asarray(curr_session)
    curr_session = curr_session.flatten()
    if (len(curr_session) < 10):
        continue
    labels.append(player_record[user_id]['label'])
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
nb_test_samples = int(TEST_SPLIT * data.shape[0])


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:-nb_test_samples]
y_val = labels[-nb_validation_samples:-nb_test_samples]
x_test = data[-nb_test_samples:]
y_test = data[-nb_test_samples:]

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
'''
train_data = np.asarray([x_train, y_train])
val_data = np.asarray([x_val, y_val])
test_data = np.asarray([x_test, y_test])

del x_train
del x_val
print(gc.collect() )

with open("../data/samples","wb") as f_out:
    pickle.dump(train_data, f_out, protocol=4)
    pickle.dump(val_data, f_out, protocol=4)
    pickle.dump(test_data, f_out, protocol=4)

file = h5py.File('../data/samples','w')
file.create_dataset('train_data', data = x_train)
file.create_dataset('val_data', data = x_val)
file.create_dataset('test_data',data = x_test)
file.create_dataset('train_label', data = y_train)
file.create_dataset('val_label', data = y_val)
file.create_dataset('test_label', data = y_test)
file.close()
'''

with open("../data/train_samples",'w') as file:
    for i in range(x_train.shape[0]):
        data = {}
        data['data'] = x_train[i].tolist()
        data['label'] = y_train[i].tolist()
        data = json.dumps(data)
        file.write(data + "\n")

with open("../data/val_samples",'w') as file:
    for i in range(x_val.shape[0]):
        data = {}
        data['data'] = x_val[i].tolist()
        data['label'] = y_val[i].tolist()
        data = json.dumps(data)
        file.write(data + "\n")
with open("../data/test_samples",'w') as file:
    for i in range(x_test.shape[0]):
        data = {}
        data['data'] = x_test[i].tolist()
        data['label'] = y_test[i].tolist()
        data = json.dumps(data)
        file.write(data + "\n")


with open("../data/embedding_matrix", "wb") as f_out:
    pickle.dump(embedding_matrix, f_out, protocol=4)







