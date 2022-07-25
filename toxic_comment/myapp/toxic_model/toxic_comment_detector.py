import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow import keras
import pickle
from itertools import compress
import sys, os, re, csv, codecs, numpy as np, pandas as pd
from tensorflow.keras.utils import pad_sequences

# setting
embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# loading
with open('./myapp/toxic_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model('./myapp/toxic_model/lstm_baseline.h5')
def detect_toxic_comment(text):
    sentence = text if type(text) == list else [text]
    list_tokenized_sentence = tokenizer.texts_to_sequences(sentence)
    X_s = pad_sequences(list_tokenized_sentence, maxlen=maxlen)
    result = model.predict([X_s], verbose=0) > 0.5
    return result.sum() >= 1, [x for y, x in zip(result[0, :], labels) if y]

if __name__ == '__main__':
    text_list = ["Hi.", "Fuck you", "I will kill you.", "You will die."]
    for text in text_list:
        print(f"{text} : {detect_toxic_comment(text)}")