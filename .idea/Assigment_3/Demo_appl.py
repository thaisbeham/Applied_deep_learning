import tensorflow as tf
import pandas as pd

# from utils import merge_datasets, smaller_set
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras.layers import (
    Embedding,
    Dense,
    LSTM,
    Dropout,
    Flatten,
    BatchNormalization,
    Conv1D,
    GlobalMaxPooling1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
)
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

# from hyperas.distributions import uniform

from keras.utils.np_utils import to_categorical
from keras import regularizers
import string
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras import callbacks
from keras.utils import plot_model
import sys

import click

nltk.download("stopwords")
nltk.download("punkt")

max_words = 1000
max_len = 300
n_batchsize = 128
n_epochs = 2
dropout = 0.2


@click.command()
@click.argument('path_real', default= 'C:/Users/freca/OneDrive/Documentos/GitHub/Applied_deep_learning/Assigment 2/data/True_bisaillon.csv')
@click.argument('path_fake',  default = 'C:/Users/freca/OneDrive/Documentos/GitHub/Applied_deep_learning/Assigment 2/data/fake.csv')
def main(path_real = 'Assigment 2/data/True_bisaillon.csv', path_fake = 'Assigment 2/data/fake.csv' ):


    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = merge_split_datasets(path_real, path_fake)

    trained_model, tok = trainer(X_train, Y_train, max_words, max_len, n_batchsize, n_epochs)

    Y_pred, cm, classif_report = predict_result(X_test, Y_test, trained_model, tok)


    with open("out.txt", "w") as f:
        print(Y_pred, file =f)
        print(cm, file=f)
        print(classif_report, file=f)

    #return Y_pred, cm



def read_news(path_real, path_fake):
    real = pd.read_csv(path_real)
    real["text"] = real["text"].astype(str)
    #if column_text_name != "text":
     #   news.rename(columns={column_text_name: "text"}, inplace=True)
    fake = pd.read_csv(path_fake)
    fake["text"] = fake["text"].astype(str)

    return real, fake


def merge_split_datasets(path_real, path_fake):

    real, fake = read_news(path_real, path_fake)

    n_testsize=0.15
    # add column label
    real["label"] = 1
    fake["label"] = 0

    real = real[["text", "label"]]
    fake = fake[["text", "label"]]

    # merge real and fake
    # use only the sentence column on fake
    merged = pd.DataFrame(real.append(fake, ignore_index=True))
    merged = merged.dropna()

    # shuffle dataset
    merged = merged.sample(frac=1, random_state=1, ignore_index=True)  # .reset_index()

    merged["text"].dropna(inplace=True)
    merged["text"] = merged["text"].astype(str)

    # remove stopwords
    tokens = merged["text"].apply(word_tokenize)
    tokens = tokens.astype(str)
    tokens = tokens.apply(str)
    stop_words = set(stopwords.words("english"))
    filtered_sentence = [w for w in tokens if not w.lower() in stop_words]

    X_train, X_test, Y_train, Y_test = train_test_split(
        filtered_sentence, merged["label"], test_size=n_testsize, random_state=1
    )

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train, test_size=0.15, random_state=42
    )

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def RNN(max_words, max_len, dropout = 0.2):
    inputs = Input(name="inputs", shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name="FC1")(layer)
    layer = Activation("relu")(layer)
    layer = Dropout(dropout)(layer)
    layer = Dense(1, name="out_layer")(layer)
    layer = Activation("sigmoid")(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


def predict_result(X_test, Y_test, trained_model, tok):
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    accr = trained_model.evaluate(test_sequences_matrix, Y_test)
    print("Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))
    predict_labels = trained_model.predict(test_sequences_matrix)

    # convert the range predicition into 0s or 1s
    Y_pred = [1 if i > 0.5 else 0 for i in predict_labels]

    print("Number Real news:", sum(Y_pred))
    print("Number Fake news:", len(Y_pred) - sum(Y_pred))
    print()
    print("Classification Report")
    classif_report = classification_report(Y_test, Y_pred, target_names=["class 0", "class 1"])
    print(classif_report)
    print()
    print("Confusion Matrix")
    matrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    cm = pd.DataFrame(
        matrix,
        index=["class_0 pred", "class_1 pred"],
        columns=["class_0 True", "class_1 True"],
    )
    #print(cm)
    return Y_pred, cm, classif_report





def trainer(X_train, Y_train, max_words, max_len, n_batchsize, n_epochs):
    model = RNN(max_words, max_len, dropout = 0.2)
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", verbose=1, patience=3
    )
    tok = Tokenizer(
        num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True
    )
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    print("Found %s unique tokens." % len(sequences_matrix))
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
    model.fit(
        sequences_matrix,
        Y_train,
        batch_size=n_batchsize,
        epochs=n_epochs,
        validation_split=0.2,
        callbacks=[earlystop],
    )

    return model, tok

if __name__ == '__main__':
    main()