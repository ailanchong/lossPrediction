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
from keras.models import load_model  
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
DATA_DIR = "../data/"

f = open(os.path.join(DATA_DIR, 'samples'),'rb')
#x_train, y_train = pickle.load(f)
x_val, y_val = pickle.load(f)
f.close()

with open("../data/embedding_matrix", "wb") as f_out:
    pickle.dump(embedding_matrix, f_out)

model = load_model("my_model.h5")

loss_and_metrics = model.evaluate(x_val, y_val, batch_size=20)
print(loss_and_metrics)