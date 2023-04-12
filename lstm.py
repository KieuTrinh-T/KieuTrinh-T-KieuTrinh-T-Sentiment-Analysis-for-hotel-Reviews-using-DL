import ktrain
from keras.models import Sequential,load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

lstm_model = load_model('Models/lstm_model2.h5')
un_lstm_model = load_model('Models/lstm_under_model.h5')

def lstm_predict(review):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=250)
    pred1 = lstm_model.predict(padded)
    pred2 = un_lstm_model.predict(padded)
    labels = ['negative', 'neutral', 'positive']
    return (labels[np.argmax(pred1)],labels[np.argmax(pred2)])
    