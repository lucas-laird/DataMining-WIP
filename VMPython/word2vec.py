from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import os
import numpy as np
from nltk.corpus import stopwords
from multiprocessing import cpu_count
from helper_files import *


directory = "reviews_3mo"
data = read_reviews(directory) #We only want the review text
data = [d for d in data]
num_cores = cpu_count()
model = Word2Vec(data, size = 100, window = 5, min_count = 1, workers = num_cores)
model.save('word2vec_test.model')
model.wv.save('wordvecs_test.kv') #Save keyed vectors as well
