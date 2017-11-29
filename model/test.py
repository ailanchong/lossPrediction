from gensim.models import word2vec
import numpy as np

embeddings_index = {}
new_model = word2vec.Word2Vec.load('../data/activity_embedding')
for i in range(3640):
    activity = i+1
    embedding = new_model[str(i+1)]
    embeddings_index[activity] = embedding
print('Total %s activity vectors.' % len(embeddings_index))

'''
for line in f:
    values = line.split()
    activity = values[0]
    embedding = np.asarray(values[1:], dtype='float32')
    embeddings_index[activity] = embedding
f.close()
'''
