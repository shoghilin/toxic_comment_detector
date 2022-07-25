from ast import arg
import sys, os, re, csv, codecs, numpy as np, pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

def load_data(TRAIN_DATA_FILE, TEST_DATA_FILE):
    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)

    list_sentences_train = train["comment_text"].fillna("_na_").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("_na_").values
    return list_sentences_train, y, list_sentences_test, list_classes


def preprocessing(args, list_sentences_train, list_sentences_test):
    tokenizer = Tokenizer(num_words=args.max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=args.maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=args.maxlen)

    # Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf-8"))

    # Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe.
    all_embs = np.stack(np.array(list(embeddings_index.values())))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    word_index = tokenizer.word_index
    nb_words = min(args.max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, args.embed_size))
    for word, i in word_index.items():
        if i >= args.max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return X_t, X_te, embedding_matrix, tokenizer

def build_model(args, embedding_matrix):
    inp = Input(shape=(args.maxlen,))
    x = Embedding(args.max_features, args.embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    # x = Bidirectional(GRU(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_kaggle_submission(args, X_te):
    y_test = model.predict([X_te], batch_size=1024, verbose=1)
    sample_submission = pd.read_csv(f'{args.path}sample_submission.csv')
    sample_submission[list_classes] = y_test
    sample_submission.to_csv('submission.csv', index=False)
    print("Submission file generated.")

def tweet_evaluation(args, model, tokenizer):
    tweet = pd.read_csv("../Data/toxic_tweet/FinalBalancedDataset.csv", index_col=0)
    list_sentences_tweet = tweet["tweet"].fillna("_na_").values
    y = tweet["Toxicity"].values
    list_tokenized_tweet = tokenizer.texts_to_sequences(list_sentences_tweet)
    X_tw = pad_sequences(list_tokenized_tweet, maxlen=args.maxlen)
    y_tweet = model.predict([X_tw], batch_size=1024, verbose=1)
    y_tweet_predict = np.sum(y_tweet >= 0.5, axis=1) > 0
    acc = (y == y_tweet_predict).mean()
    return acc

def save_result(model, tokenizer):
    model.save('lstm_baseline.h5')
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model and tokenizer has been saved.")

def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='../Data/', type=str, required=False,  help='dataset path')
    parser.add_argument('--embed_size', default=50, type=int, help='how big is each word vector')
    parser.add_argument('--max_features', default=20000, type=int, help='how many unique words to use (i.e num rows in embedding vector)')
    parser.add_argument('--maxlen', default=100, type=int, help='max number of words in a comment to use.')
    parser.add_argument('--kaggle_submit', action='store_true', help='Generate Kaggle Challenge result or not.')
    parser.add_argument('--save_model_tokenizer', action='store_true', help='Save trained model and tokenizer for future inference.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # configuration
    args = set_config()
    EMBEDDING_FILE=f'{args.path}glove6b50d/glove.6B.50d.txt'
    TRAIN_DATA_FILE=f'{args.path}train.csv'
    TEST_DATA_FILE=f'{args.path}test.csv'

    # Load the data
    list_sentences_train, y, list_sentences_test, list_classes = load_data(TRAIN_DATA_FILE, TEST_DATA_FILE)

    # Preprocess the text data
    X_t, X_te, embedding_matrix, tokenizer = preprocessing(args, list_sentences_train, list_sentences_test)

    # Build model
    model = build_model(args, embedding_matrix)

    # Training process
    model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1)

    # Evaluation
    # 1. generate testing prediction for Kaggle submission
    if args.kaggle_submit:
        generate_kaggle_submission(args, X_te)
    # 2. Evaluate on toxic tweet dataset
    acc = tweet_evaluation(args, model, tokenizer)
    print(f"The accuracy on toxic tweet dataset is {acc*100:.2f}%")

    # Save model
    if args.save_model_tokenizer:
        save_result(model, tokenizer)